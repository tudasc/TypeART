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

#include <cxxabi.h>
#include <filesystem>
#include <memory>

namespace fs = std::filesystem;

namespace typeart {

void printMPIError(const MPICall* call, const char* fnname, int mpierr) {
  int len;
  auto mpierrstr = std::string{};
  mpierrstr.resize(MPI_MAX_ERROR_STRING);
  MPI_Error_string(mpierr, &mpierrstr[0], &len);
  mpierrstr.resize(strlen(mpierrstr.c_str()));
  PRINT_ERRORV(call, "%s failed: %s", fnname, mpierrstr.c_str());
}

std::optional<Buffer> Buffer::create(const MPICall* call, const void* buffer) {
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

std::optional<Buffer> Buffer::create(const MPICall* call, ptrdiff_t offset, const void* ptr, size_t count,
                                     int type_id) {
  const auto* type_name = typeart_get_type_name(type_id);
  typeart_struct_layout struct_layout;
  typeart_status status = typeart_resolve_type_id(type_id, &struct_layout);
  if (status == TYPEART_INVALID_ID) {
    PRINT_ERRORV(call, "Buffer::create received an invalid type_id %d\n", type_id);
    return {};
  }
  if (status == TYPEART_OK) {
    auto type_layout = std::vector<Buffer>{};
    type_layout.reserve(struct_layout.num_members);
    for (auto i = size_t{0}; i < struct_layout.num_members; ++i) {
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

std::optional<MPICombiner> MPICombiner::create(const MPICall* call, MPI_Datatype type) {
  auto result = MPICombiner{};
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
    auto type_args = std::vector<MPI_Datatype>(num_datatypes);
    mpierr         = MPI_Type_get_contents(type, num_integers, num_addresses, num_datatypes, result.integer_args.data(),
                                   result.address_args.data(), type_args.data());
    if (mpierr != MPI_SUCCESS) {
      printMPIError(call, "MPI_Type_get_contents", mpierr);
      return {};
    }
    result.type_args.reserve(num_datatypes);
    for (auto i = size_t{0}; i < num_datatypes; ++i) {
      auto type = MPIType::create(call, type_args[i]);
      if (!type) {
        return {};
      }
      result.type_args.push_back(*type);
    }
  }
  return {result};
}

std::optional<MPIType> MPIType::create(const MPICall* call, MPI_Datatype type) {
  auto combiner = MPICombiner::create(call, type);
  if (!combiner) {
    return {};
  }
  const int type_id = type_id_for(type);
  auto result       = MPIType{type, type_id, "", *combiner};
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
    auto result  = std::string{};
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
  if (status != 0) {
    return {};
  }
  auto result = std::string{buffer};
  free(buffer);
  return result;
}

std::optional<Caller> Caller::create(const void* addr) {
  auto result  = Caller{};
  result.addr  = addr;
  auto command = std::ostringstream{};
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
  auto result = MPICall{next_trace_id++, function_name, (Caller){}, is_const, rank, (Buffer){}, count, (MPIType){}};
  auto caller = Caller::create(called_from);
  if (!caller) {
    fprintf(stderr, "[Info, r%d, id%ld] couldn't resolve the symbol name for address %p", result.rank, result.trace_id,
            called_from);
    return {};
  }
  auto buffer = Buffer::create(&result, buffer_ptr);
  if (!buffer) {
    return {};
  }
  auto mpi_type = MPIType::create(&result, type);
  if (!mpi_type) {
    return {};
  }
  result.caller = *caller;
  result.buffer = *buffer;
  result.type   = *mpi_type;
  return {result};
}

int MPICall::check_type_and_count() const {
  return check_type_and_count(&buffer);
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

int MPICall::check_type_and_count(const Buffer* buffer) const {
  auto result = check_type(buffer, &type);
  if (result.result != 0) {
    // If the type is a struct type and has a member with offset 0,
    // recursively check against the type of the first member.
    auto type_layout = buffer->type_layout;
    if (type_layout && (*type_layout)[0].offset == 0) {
      PRINT_INFOV(this, "found struct member at offset 0 with type \"%s\", checking with this type...\n",
                  (*type_layout)[0].type_name.c_str());
      return check_type_and_count(&(*type_layout)[0]);
    }
    return -1;
  }
  if (count * result.count > buffer->count) {
    PRINT_ERRORV(this, "buffer too small (%ld elements, %d required)\n", buffer->count, count * result.count);
    return -1;
  }
  return 0;
}

MPICall::CheckResult MPICall::check_type(const Buffer* buffer, const MPIType* type) const {
  switch (type->combiner.id) {
    case MPI_COMBINER_NAMED:
      return check_combiner_named(buffer, type);
    case MPI_COMBINER_DUP:
      return check_type(buffer, &type->combiner.type_args[0]);
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
      PRINT_ERRORV(this, "the MPI type combiner %s is currently not supported\n", combiner_name_for(type->combiner.id));
  }
  return CheckResult::error();
}

MPICall::CheckResult MPICall::check_combiner_named(const Buffer* buffer, const MPIType* type) const {
  if (buffer->type_id != type->type_id && !(buffer->type_id == TYPEART_PPC_FP128 && type->type_id == TYPEART_FP128)) {
    PRINT_ERRORV(this, "expected a type matching MPI type \"%s\", but found type \"%s\"\n", type->name.c_str(),
                 buffer->type_name.c_str());
    return CheckResult::error();
  }
  return CheckResult::with_count(1);
}

MPICall::CheckResult MPICall::check_combiner_contiguous(const Buffer* buffer, const MPIType* type) const {
  return check_type(buffer, &type->combiner.type_args[0]).multiply_count_by(type->combiner.integer_args[0]);
}

MPICall::CheckResult MPICall::check_combiner_vector(const Buffer* buffer, const MPIType* type) const {
  const auto& integer_args = type->combiner.integer_args;
  if (integer_args[2] < 0) {
    PRINT_ERROR(this, "negative strides for MPI_Type_vector are currently not supported\n");
    return {-1, -1};
  }
  return check_type(buffer, &type->combiner.type_args[0])
      //                      (count - 1)      *     stride      +   blocklength
      .multiply_count_by((integer_args[0] - 1) * integer_args[2] + integer_args[1]);
}

MPICall::CheckResult MPICall::check_combiner_indexed_block(const Buffer* buffer, const MPIType* type) const {
  const auto& integer_args = type->combiner.integer_args;
  auto max_displacement    = 0;
  for (size_t i = 2; i < integer_args.size(); ++i) {
    if (integer_args[i] > max_displacement) {
      max_displacement = integer_args[i];
    }
    if (integer_args[i] < 0) {
      PRINT_ERROR(this, "negative displacements for MPI_Type_create_indexed_block are currently not supported\n");
      return {-1, -1};
    }
  }
  //                                                                    max(array_of_displacements) + blocklength
  return check_type(buffer, &type->combiner.type_args[0]).multiply_count_by(max_displacement + integer_args[1]);
}

MPICall::CheckResult MPICall::check_combiner_struct(const Buffer* buffer, const MPIType* type) const {
  const auto& integer_args = type->combiner.integer_args;
  if (!buffer->hasStructType()) {
    PRINT_ERRORV(this, "expected a struct type, but found type \"%s\"\n", buffer->type_name.c_str());
    return CheckResult::error();
  }
  const auto& type_layout = *(buffer->type_layout);
  if (type_layout.size() != integer_args[0]) {
    PRINT_ERRORV(this, "expected %d members, but the type \"%s\" has %ld members\n", integer_args[0],
                 buffer->type_name.c_str(), type_layout.size());
    return CheckResult::error();
  }
  auto result = CheckResult::with_count(1);
  for (size_t i = 0; i < type_layout.size(); ++i) {
    if (type_layout[i].offset != type->combiner.address_args[i]) {
      PRINT_ERRORV(this, "expected a byte offset of %ld for member %ld, but the type \"%s\" has an offset of %ld\n",
                   type->combiner.address_args[i], i + 1, buffer->type_name.c_str(), type_layout[i].offset);
      result = CheckResult::error();
    }
  }
  for (size_t i = 0; i < type_layout.size(); ++i) {
    auto check_result = check_type(&type_layout[i], &type->combiner.type_args[i]);
    if (check_result.result != 0) {
      PRINT_ERRORV(this, "the typechek for member %ld failed\n", i + 1);
      result = CheckResult::error();
    }
    if (type_layout[i].count * check_result.count != integer_args[i + 1]) {
      PRINT_ERRORV(this, "expected element count of %d for member %ld, but the type \"%s\" has a count of %ld\n",
                   integer_args[i + 1], i + 1, buffer->type_name.c_str(), type_layout[i].count * check_result.count);
      result = CheckResult::error();
    }
  }
  return result;
}

MPICall::CheckResult MPICall::check_combiner_subarray(const Buffer* buffer, const MPIType* type) const {
  const auto& integer_args = type->combiner.integer_args;
  auto array_element_count = 1;
  for (auto i = 0; i < integer_args[0]; ++i) {
    array_element_count *= integer_args[i + 1];
  }
  return check_type(buffer, &type->combiner.type_args[0]).multiply_count_by(array_element_count);
}

}  // namespace typeart
