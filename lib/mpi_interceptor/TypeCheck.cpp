// TypeART library
//
// Copyright (c) 2017-2021 TypeART Authors
// Distributed under the BSD 3-Clause license.
// (See accompanying file LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)
//
// Project home: https://github.com/tudasc/TypeART
//
// SPDX-License-Identifier: BSD-3-Clause
//

#include "TypeCheck.h"

#include <algorithm>
#include <cxxabi.h>
#include <filesystem>
#include <memory>
#include <numeric>
#include <sstream>

namespace fs = std::filesystem;

namespace typeart {

void printMPIError(const MPICall& call, const char* fnname, int mpierr) {
  int len;
  std::string mpierrstr;
  mpierrstr.resize(MPI_MAX_ERROR_STRING);
  MPI_Error_string(mpierr, &mpierrstr[0], &len);
  mpierrstr.resize(strlen(mpierrstr.c_str()));
  PRINT_ERRORV(call, "%s failed: %s", fnname, mpierrstr.c_str());
}

std::optional<Buffer> Buffer::create(const MPICall& call, const void* buffer) {
  if (buffer == nullptr) {
    return {Buffer::create(call, 0, nullptr, 0, TYPEART_INVALID_ID)};
  }
  int type_id;
  size_t count          = 0;
  auto typeart_status_v = typeart_get_type(buffer, &type_id, &count);
  if (typeart_status_v != TYPEART_OK) {
    const char* msg = error_message_for(typeart_status_v);
    PRINT_ERRORV(call, "internal runtime error (%s)\n", msg);
    return {};
  }
  return {Buffer::create(call, 0, buffer, count, type_id)};
}

std::optional<Buffer> Buffer::create(const MPICall& call, ptrdiff_t offset, const void* ptr, size_t count,
                                     int type_id) {
  if (ptr == nullptr) {
    return {{0, nullptr, 0, TYPEART_INVALID_ID, "", {}}};
  }
  auto type_name = typeart_get_type_name(type_id);
  typeart_struct_layout struct_layout;
  typeart_status status = typeart_resolve_type_id(type_id, &struct_layout);
  if (status == TYPEART_INVALID_ID) {
    PRINT_ERRORV(call, "Buffer::create received an invalid type_id %d\n", type_id);
    return {};
  }
  if (status == TYPEART_OK) {
    std::vector<Buffer> type_layout = {};
    type_layout.reserve(struct_layout.num_members);
    for (size_t i = 0; i < struct_layout.num_members; ++i) {
      auto buffer =
          Buffer::create(call, static_cast<ptrdiff_t>(struct_layout.offsets[i]), (char*)ptr + struct_layout.offsets[i],
                         struct_layout.count[i], struct_layout.member_types[i]);
      if (!buffer) {
        return {};
      }
      type_layout.push_back(*buffer);
    }
    return {{offset, ptr, count, type_id, type_name, {type_layout}}};
  }
  return {{offset, ptr, count, type_id, type_name, {}}};
}

bool Buffer::hasStructType() const {
  return type_layout.has_value();
}

std::optional<MPICombiner> MPICombiner::create(const MPICall& call, MPI_Datatype type) {
  MPICombiner result;
  int num_integers;
  int num_addresses;
  int num_datatypes;
  int combiner;
  auto mpierr = MPI_Type_get_envelope(type, &num_integers, &num_addresses, &num_datatypes, &combiner);
  if (mpierr != MPI_SUCCESS) {
    printMPIError(call, "MPI_Type_get_envelope", mpierr);
    return {};
  }
  result.id = combiner;
  if (combiner != MPI_COMBINER_NAMED) {
    result.integer_args.resize(num_integers);
    result.address_args.resize(num_addresses);
    std::vector<MPI_Datatype> type_args(num_datatypes);
    mpierr = MPI_Type_get_contents(type, num_integers, num_addresses, num_datatypes, result.integer_args.data(),
                                   result.address_args.data(), type_args.data());
    if (mpierr != MPI_SUCCESS) {
      printMPIError(call, "MPI_Type_get_contents", mpierr);
      return {};
    }
    result.type_args.reserve(num_datatypes);
    for (auto i = size_t{0}; i < num_datatypes; ++i) {
      auto type_arg = MPIType::create(call, type_args[i]);
      if (!type_arg) {
        return {};
      }
      result.type_args.push_back(*type_arg);
    }
  }
  return {result};
}

std::optional<MPIType> MPIType::create(const MPICall& call, MPI_Datatype type) {
  auto combiner = MPICombiner::create(call, type);
  if (!combiner) {
    return {};
  }
  const auto type_id = type_id_for(type);
  auto result        = MPIType{type, type_id, "", *combiner};
  int len;
  result.name.resize(MPI_MAX_OBJECT_NAME);
  int mpierr = MPI_Type_get_name(type, &result.name[0], &len);
  result.name.resize(strlen(result.name.c_str()));
  if (mpierr != MPI_SUCCESS) {
    printMPIError(call, "MPI_Type_get_name", mpierr);
    return {};
  }
  return {result};
}

struct Self {
  std::string exe;

 public:
  Self() {
    exe = fs::canonical("/proc/self/exe");
  }
};

static Self self;

using unique_file = std::unique_ptr<FILE, int (*)(FILE*)>;

struct pipe : public unique_file {
  explicit pipe(const std::string& command) : unique_file(popen(command.c_str(), "r"), &pclose) {
  }

  [[nodiscard]] std::string next_line() const {
    size_t len   = 0;
    char* buffer = nullptr;
    std::string result;
    if (getline(&buffer, &len, get()) != -1) {
      result = {std::string(buffer)};
      free(buffer);
    }
    result.resize(result.size() - 1);
    return result;
  }
};

std::optional<std::string> demangle(const std::string& symbol_name) {
  auto status  = -1;
  auto* buffer = abi::__cxa_demangle(symbol_name.c_str(), nullptr, nullptr, &status);
  // If the status is
  //   "-2: mangled_name is not a valid name under the C++ ABI mangling rules"
  //   Source: https://gcc.gnu.org/onlinedocs/libstdc++/libstdc++-html-USERS-4.3/a01696.html
  // we assume that the mangled string is a C symbol and return it as-is.
  if (status == -2) {
    return symbol_name;
  }
  if (status != 0) {
    return {};
  }
  std::string result{buffer};
  free(buffer);
  return result;
}

std::optional<Caller> Caller::create(const void* addr) {
  Caller result;
  result.addr = addr;
  std::ostringstream command;
  command << "addr2line -e " << self.exe << " -f " << addr;
  auto output = pipe(command.str());
  if (!output) {
    return {};
  }
  auto demangled = demangle(output.next_line());
  if (!demangled) {
    return {};
  }
  result.function    = *demangled;
  auto file_and_line = output.next_line();
  auto delim         = file_and_line.find(':');
  result.line        = file_and_line.substr(delim + 1);
  file_and_line.resize(delim);
  result.file = std::move(file_and_line);
  return result;
}

std::atomic_size_t MPICall::next_trace_id = {0};

std::optional<MPICall> MPICall::create(const char* function_name, const void* called_from, const void* buffer_ptr,
                                       int is_const, int count, MPI_Datatype type) {
  auto rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  auto result = MPICall{next_trace_id++, Caller{}, function_name, is_const, rank, {Buffer{}, count, MPIType{}}};
  auto caller = Caller::create(called_from);
  if (!caller) {
    fprintf(stderr, "R[%d][Error]ID[%ld] couldn't resolve the symbol name for address %p", result.rank, result.trace_id,
            called_from);
    return {};
  }
  auto buffer = Buffer::create(result, buffer_ptr);
  if (!buffer) {
    return {};
  }
  auto mpi_type = MPIType::create(result, type);
  if (!mpi_type) {
    return {};
  }
  result.caller      = *caller;
  result.args.buffer = *buffer;
  result.args.type   = *mpi_type;
  return {result};
}

int MPICall::check_type_and_count() const {
  return check_type_and_count_against(args.buffer);
}

MPICall::CheckResult MPICall::CheckResult::error() {
  return {-1, -1};
}

MPICall::CheckResult MPICall::CheckResult::with_count(int count) {
  return {0, count};
}

MPICall::CheckResult& MPICall::CheckResult::multiply_count_by(int rhs) {
  count *= rhs;
  return *this;
}

// For a given Buffer checks that the type of the buffer fits the MPI type
// `args.type` of this MPICall instance and that the buffer is large enough to
// hold `args.count` elements of the MPI type.
int MPICall::check_type_and_count_against(const Buffer& buffer) const {
  auto result = check_type(buffer, args.type);
  if (result.result != 0) {
    // If the type is a struct type and has a member with offset 0,
    // recursively check against the type of the first member.
    const auto type_layout = buffer.type_layout;
    if (!type_layout) {
      return -1;
    }
    const auto first_member = (*type_layout)[0];
    if (first_member.offset == 0) {
      PRINT_INFOV(*this, "found struct member at offset 0 with type \"%s\", checking with this type...\n",
                  first_member.type_name.c_str());
      return check_type_and_count_against(first_member);
    }
    return -1;
  }
  auto required_count = static_cast<size_t>(args.count) * result.count;
  if (required_count > buffer.count) {
    PRINT_ERRORV(*this, "buffer too small (%ld elements, %d required)\n", buffer.count, required_count);
    return -1;
  }
  return 0;
}

// For a given Buffer and MPIType, checks that the buffer's type matches the
// MPI type.
// The resulting integer `count` is the number of elements of the buffer's type
// required to represent one element of the MPI type.
// (e.g. an MPI_Type_contiguous with a `count` of 4 and an `oldtype` of
// MPI_DOUBLE would require 4 double elements for each element of that type.)
MPICall::CheckResult MPICall::check_type(const Buffer& buffer, const MPIType& type) const {
  switch (type.combiner.id) {
    case MPI_COMBINER_NAMED:
      return check_combiner_named(buffer, type);
    case MPI_COMBINER_DUP:
      // MPI_Type_dup creates an exact duplicate of the type argument to the type
      // combiner, so we can delegate to a check against that type.
      return check_type(buffer, type.combiner.type_args[0]);
    case MPI_COMBINER_CONTIGUOUS:
      return check_combiner_contiguous(buffer, type);
    case MPI_COMBINER_VECTOR:
      return check_combiner_vector(buffer, type);
    case MPI_COMBINER_INDEXED_BLOCK:
      return check_combiner_indexed_block(buffer, type);
    case MPI_COMBINER_STRUCT:
      return check_combiner_struct(buffer, type);
    case MPI_COMBINER_SUBARRAY:
      return check_combiner_subarray(buffer, type);
    default:
      PRINT_ERRORV(*this, "the MPI type combiner %s is currently not supported\n", combiner_name_for(type.combiner.id));
  }
  return CheckResult::error();
}

// See MPICall::check_type(const Buffer&, const MPIType&)
MPICall::CheckResult MPICall::check_combiner_named(const Buffer& buffer, const MPIType& type) const {
  // For named types (like e.g. MPI_DOUBLE) we compare the type id of the
  // buffer with the type id deduced for the MPI type using the type_id_for
  // function from Util.h.
  // As a special case, if the types do not match, but both represent a 128bit
  // floating point type, they are also considered to match.
  if (buffer.type_id != type.type_id && !(buffer.type_id == TYPEART_PPC_FP128 && type.type_id == TYPEART_FP128)) {
    PRINT_ERRORV(*this, "expected a type matching MPI type \"%s\", but found type \"%s\"\n", type.name.c_str(),
                 buffer.type_name.c_str());
    return CheckResult::error();
  }
  return CheckResult::with_count(1);
}

// Type check for the type combiner:
// int MPI_Type_contiguous(int count, MPI_Datatype oldtype,
//     MPI_Datatype *newtype)
//
// See MPICall::check_type(const Buffer&, const MPIType&) for an explanation of
// the arguments and the return type.
MPICall::CheckResult MPICall::check_combiner_contiguous(const Buffer& buffer, const MPIType& type) const {
  // MPI_Type_contiguous has one type argument and a count which denotes the
  // number of consecutive elements of the old type forming one element of the
  // conntiguous type. Therefore, we check that the old type matches the
  // buffer's type and multiply the count required for on element by the first
  // the first integer argument of the type combiner.
  auto count = type.combiner.integer_args[0];
  return check_type(buffer, type.combiner.type_args[0]).multiply_count_by(count);
}

// Type check for the type combiner:
// int MPI_Type_vector(int count, int blocklength, int stride,
//     MPI_Datatype oldtype, MPI_Datatype *newtype)
//
// See MPICall::check_type(const Buffer&, const MPIType&) for an explanation of
// the arguments and the return type.
MPICall::CheckResult MPICall::check_combiner_vector(const Buffer& buffer, const MPIType& type) const {
  const auto count       = type.combiner.integer_args[0];
  const auto blocklength = type.combiner.integer_args[1];
  const auto stride      = type.combiner.integer_args[2];
  if (stride < 0) {
    PRINT_ERROR(*this, "negative strides for MPI_Type_vector are currently not supported\n");
    return {-1, -1};
  }
  // MPI_Type_vector forms a number of `count` blocks of `oldtype` where the
  // start of each consecutive block is `stride` elements of `oldtype` apart
  // and each block consists of `blocklength` elements of oldtype.
  // We therefore check the buffer's type against `oldtype` and multiply the
  // resulting count by `(count - 1) * stride + blocklength`.
  return check_type(buffer, type.combiner.type_args[0]).multiply_count_by((count - 1) * stride + blocklength);
}

// Type check for the type combiner:
// int MPI_Type_create_indexed_block(int count, int blocklength, const int
//     array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype *newtype)
//
// See MPICall::check_type(const Buffer&, const MPIType&) for an explanation of
// the arguments and the return type.
MPICall::CheckResult MPICall::check_combiner_indexed_block(const Buffer& buffer, const MPIType& type) const {
  const auto count                  = type.combiner.integer_args[0];
  const auto blocklength            = type.combiner.integer_args[1];
  const auto array_of_displacements = type.combiner.integer_args.begin() + 2;
  const auto [min_displacement, max_displacement] =
      std::minmax_element(array_of_displacements, array_of_displacements + count);
  if (*min_displacement < 0) {
    PRINT_ERROR(*this, "negative displacements for MPI_Type_create_indexed_block are currently not supported\n");
    return {-1, -1};
  }
  // Similer to MPI_Type_vector but with a separate displacement specified for
  // each block.
  // We therefore check the buffer's type against `oldtype` and multiply the
  // resulting count by `max(array_of_displacements) + blocklength`.
  return check_type(buffer, type.combiner.type_args[0]).multiply_count_by(*max_displacement + blocklength);
}

// Type check for the type combiner:
// int MPI_Type_create_struct(int count, int array_of_blocklengths[],
//     const MPI_Aint array_of_displacements[], const MPI_Datatype array_of_types[],
//     MPI_Datatype *newtype)
//
// See MPICall::check_type(const Buffer&, const MPIType&) for an explanation of
// the arguments and the return type.
MPICall::CheckResult MPICall::check_combiner_struct(const Buffer& buffer, const MPIType& type) const {
  const auto count                 = type.combiner.integer_args[0];
  const auto array_of_blocklenghts = type.combiner.integer_args.begin() + 1;
  // First, check that the buffer's type is a struct type...
  if (!buffer.hasStructType()) {
    PRINT_ERRORV(*this, "expected a struct type, but found type \"%s\"\n", buffer.type_name.c_str());
    return CheckResult::error();
  }
  // ... and that the number of members of the struct matches the argument
  // `count` of the type combiner.
  const auto& type_layout = *(buffer.type_layout);
  if (type_layout.size() != count) {
    PRINT_ERRORV(*this, "expected %d members, but the type \"%s\" has %ld members\n", count, buffer.type_name.c_str(),
                 type_layout.size());
    return CheckResult::error();
  }
  // Then, for each member check that...
  auto result = CheckResult::with_count(1);
  for (size_t i = 0; i < type_layout.size(); ++i) {
    // ... the byte offset of the member matches the respective element in
    // the `array_of_displacements` type combiner argument.
    if (type_layout[i].offset != type.combiner.address_args[i]) {
      PRINT_ERRORV(*this, "expected a byte offset of %ld for member %ld, but the type \"%s\" has an offset of %ld\n",
                   type.combiner.address_args[i], i + 1, buffer.type_name.c_str(), type_layout[i].offset);
      result = CheckResult::error();
    }
  }
  for (size_t i = 0; i < type_layout.size(); ++i) {
    // ... the type of the member matches the respective MPI type in the
    // `array_of_types` type combiner argument.
    auto check_result = check_type(type_layout[i], type.combiner.type_args[i]);
    if (check_result.result != 0) {
      PRINT_ERRORV(*this, "the typechek for member %ld failed\n", i + 1);
      result = CheckResult::error();
      continue;
    }
    // ... the count of elements in the buffer of the member matches the count
    // required to represent `blocklength` elements of the MPI type.
    const auto required_count = static_cast<size_t>(array_of_blocklenghts[i]) * check_result.count;
    if (required_count != type_layout[i].count) {
      PRINT_ERRORV(*this, "expected element count of %ld for member %ld, but the type \"%s\" has a count of %d\n",
                   type_layout[i].count, i + 1, buffer.type_name.c_str(), required_count);
      result = CheckResult::error();
    }
  }
  return result;
}

// Type check for the type combiner:
// int MPI_Type_create_subarray(int ndims, const int array_of_sizes[], const
//     int array_of_subsizes[], const int array_of_starts[], int order, MPI_Datatype
//     oldtype, MPI_Datatype *newtype)
//
// See MPICall::check_type(const Buffer&, const MPIType&) for an explanation of
// the arguments and the return type.
MPICall::CheckResult MPICall::check_combiner_subarray(const Buffer& buffer, const MPIType& type) const {
  const auto ndims               = type.combiner.integer_args[0];
  const auto array_of_sizes      = type.combiner.integer_args.begin() + 1;
  const auto array_element_count = std::accumulate(array_of_sizes, array_of_sizes + ndims, 1, std::multiplies{});
  // As this type combiner specifies a subarray of a larger array, the buffer
  // must be large enough to hold that larger array. We therefore check the
  // buffer's type against `oldtype` and multiply the resulting count with
  // the product of all elements of the `array_of_sizes` (i.e. the element
  // count of the large n-dimensional array).
  return check_type(buffer, type.combiner.type_args[0]).multiply_count_by(array_element_count);
}

}  // namespace typeart
