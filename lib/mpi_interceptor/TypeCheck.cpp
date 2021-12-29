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

#include "RuntimeInterface.h"
#include "TypeInterface.h"
#include "Util.h"

#include <algorithm>
#include <fmt/core.h>
#include <fmt/format.h>
#include <functional>
#include <memory>
#include <numeric>
#include <utility>

namespace typeart {

Result<Buffer> Buffer::create(const void* ptr) {
  int type_id;
  size_t count          = 0;
  auto typeart_status_v = typeart_get_type(ptr, &type_id, &count);
  if (typeart_status_v != TYPEART_OK) {
    return make_internal_error<TypeARTError>(error_message_for(typeart_status_v));
  }
  return Buffer{0, ptr, count, type_id};
}

Buffer Buffer::create(ptrdiff_t offset, const void* ptr, size_t count, int type_id) {
  return Buffer{offset, ptr, count, type_id};
}

std::string error_message_for(int mpierr) {
  int len;
  std::string mpierrstr;
  mpierrstr.resize(MPI_MAX_ERROR_STRING);
  MPI_Error_string(mpierr, &mpierrstr[0], &len);
  mpierrstr.resize(strlen(mpierrstr.c_str()));
  return mpierrstr;
}

Result<MPICombiner> MPICombiner::create(MPI_Datatype type) {
  MPICombiner result;
  int num_integers;
  int num_addresses;
  int num_datatypes;
  int combiner;
  auto mpierr = MPI_Type_get_envelope(type, &num_integers, &num_addresses, &num_datatypes, &combiner);

  if (mpierr != MPI_SUCCESS) {
    return make_internal_error<MPIError>("MPI_Type_get_envelope", error_message_for(mpierr));
  }

  result.id = combiner;

  if (combiner != MPI_COMBINER_NAMED) {
    result.integer_args.resize(num_integers);
    result.address_args.resize(num_addresses);
    std::vector<MPI_Datatype> type_args(num_datatypes);
    mpierr = MPI_Type_get_contents(type, num_integers, num_addresses, num_datatypes, result.integer_args.data(),
                                   result.address_args.data(), type_args.data());

    if (mpierr != MPI_SUCCESS) {
      return make_internal_error<MPIError>("MPI_Type_get_contents", error_message_for(mpierr));
    }

    result.type_args.reserve(num_datatypes);

    for (auto i = size_t{0}; i < num_datatypes; ++i) {
      auto type_arg = MPIType::create(type_args[i]);
      if (!type_arg) {
        return std::move(type_arg).error();
      }
      result.type_args.push_back(std::move(type_arg).value());
    }
  }

  return {result};
}

Result<MPIType> MPIType::create(MPI_Datatype type) {
  auto combiner = MPICombiner::create(type);
  if (!combiner) {
    return std::move(combiner).error();
  }

  const auto type_id = type_id_for(type);
  return MPIType{type, type_id, *combiner};
}

struct Multipliers {
  size_t type;
  size_t buffer;
};

Result<void> check_type_and_count(const Buffer& buffer, const MPIType& type, int count);
Result<Multipliers> check_type(const Buffer& buffer, const MPIType& type);
Result<Multipliers> check_combiner_named(const Buffer& buffer, const MPIType& type);
Result<Multipliers> check_combiner_contiguous(const Buffer& buffer, const MPIType& type);
Result<Multipliers> check_combiner_vector(const Buffer& buffer, const MPIType& type);
Result<Multipliers> check_combiner_indexed_block(const Buffer& buffer, const MPIType& type);
Result<Multipliers> check_combiner_struct(const Buffer& buffer, const MPIType& type);
Result<Multipliers> check_combiner_subarray(const Buffer& buffer, const MPIType& type);

// For a given Buffer checks that the type of the buffer fits the MPI type
// `args.type` of this MPICall instance and that the buffer is large enough to
// hold `args.count` elements of the MPI type.
Result<void> check_buffer(const Buffer& buffer, const MPIType& type, int count) {
  auto result = check_type_and_count(buffer, type, count);
  if (result.has_value()) {
    return result;
  }
  if (result.error()->is<InternalError>()) {
    return result;
  }

  // If the type is a struct type and has a member with offset 0,
  // recursively check against the type of the first member.
  typeart_struct_layout struct_layout;
  auto status = typeart_resolve_type_id(buffer.type_id, &struct_layout);
  if (status == TYPEART_INVALID_ID) {
    auto message = fmt::format("Buffer::create received an invalid type_id {}", buffer.type_id);
    return make_internal_error<InvalidArgument>(message);
  }

  if (status == TYPEART_WRONG_KIND) {
    return result;
  }

  std::vector<StructSubtypeMismatch> subtype_errors;
  int struct_type_id = buffer.type_id;
  while (status == TYPEART_OK && struct_layout.offsets[0] == 0) {
    auto first_member =
        Buffer::create(static_cast<ptrdiff_t>(struct_layout.offsets[0]), (char*)buffer.ptr + struct_layout.offsets[0],
                       struct_layout.count[0], struct_layout.member_types[0]);
    auto subtype_result = check_type_and_count(first_member, type, count);
    if (subtype_result.has_value()) {
      return subtype_result;
    }

    if (subtype_result.error()->is<InternalError>()) {
      return subtype_result;
    } else {
      subtype_errors.push_back(
          StructSubtypeMismatch{struct_type_id, first_member.type_id, first_member.count,
                                std::make_unique<TypeError>(std::move(*subtype_result.error()).get<TypeError>())});
    }

    status = typeart_resolve_type_id(first_member.type_id, &struct_layout);
    if (status == TYPEART_INVALID_ID) {
      auto message = fmt::format("Buffer::create received an invalid type_id {}", buffer.type_id);
      return make_internal_error<InvalidArgument>(message);
    }
    struct_type_id = first_member.type_id;
  }
  auto primary_error = std::make_unique<TypeError>(std::move(*result.error()).get<TypeError>());
  return make_type_error<StructSubtypeErrors>(std::move(primary_error), std::move(subtype_errors));
}

Result<void> check_type_and_count(const Buffer& buffer, const MPIType& type, int count) {
  auto result = check_type(buffer, type);
  if (result.has_error()) {
    return std::move(result).error();
  }
  auto multipliers  = std::move(result).value();
  auto type_count   = static_cast<size_t>(count * multipliers.type);
  auto buffer_count = buffer.count * multipliers.buffer;
  if (type_count > buffer_count) {
    return make_type_error<InsufficientBufferSize>(buffer_count, type_count);
  }
  return {};
}

// For a given Buffer and MPIType, checks that the buffer's type matches the
// MPI type.
// The resulting integer `type_count_multiplier` is the number of elements of
// the buffer's type required to represent one element of the MPI type
// (e.g. an MPI_Type_contiguous with a `count` of 4 and an `oldtype` of
// MPI_DOUBLE would require 4 double elements for each element of that type.)
// Similarly, `buffer_count_multiplier` is the number of elements of the MPI
// type needed to represent one element of the buffer's type. This is used to
// correctly handle MPI_BYTE, where for each given type T, sizeof(T) elements
// of MPI_BYTE are needed to represent one instance of T.
Result<Multipliers> check_type(const Buffer& buffer, const MPIType& type) {
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
      return make_internal_error<UnsupportedCombiner>(combiner_name_for(type.combiner.id));
  }
}

// See MPICall::check_type(const Buffer&, const MPIType&)
Result<Multipliers> check_combiner_named(const Buffer& buffer, const MPIType& type) {
  // We assume MPI_BYTE to be the MPI equivalent of void*.
  if (type.mpi_type == MPI_BYTE) {
    const auto type_size = typeart_get_type_size(buffer.type_id);
    return Multipliers{1, type_size};
  }

  // For named types (like e.g. MPI_DOUBLE) we compare the type id of the
  // buffer with the type id deduced for the MPI type using the type_id_for
  // function from Util.h.
  // As a special case, if the types do not match, but both represent a 128bit
  // floating point type, they are also considered to match.
  if (buffer.type_id != type.type_id && !(buffer.type_id == TYPEART_PPC_FP128 && type.type_id == TYPEART_FP128)) {
    return make_type_error<BuiltinTypeMismatch>(buffer.type_id, type.mpi_type);
  }
  return Multipliers{1, 1};
}

// Type check for the type combiner:
// int MPI_Type_contiguous(int count, MPI_Datatype oldtype,
//     MPI_Datatype *newtype)
//
// See MPICall::check_type(const Buffer&, const MPIType&) for an explanation of
// the arguments and the return type.
Result<Multipliers> check_combiner_contiguous(const Buffer& buffer, const MPIType& type) {
  // MPI_Type_contiguous has one type argument and a count which denotes the
  // number of consecutive elements of the old type forming one element of the
  // conntiguous type. Therefore, we check that the old type matches the
  // buffer's type and multiply the count required for on element by the first
  // the first integer argument of the type combiner.
  auto count = type.combiner.integer_args[0];
  return check_type(buffer, type.combiner.type_args[0]).map([&](auto multipliers) {
    return Multipliers{multipliers.type * count, multipliers.buffer};
  });
}

// Type check for the type combiner:
// int MPI_Type_vector(int count, int blocklength, int stride,
//     MPI_Datatype oldtype, MPI_Datatype *newtype)
//
// See MPICall::check_type(const Buffer&, const MPIType&) for an explanation of
// the arguments and the return type.
Result<Multipliers> check_combiner_vector(const Buffer& buffer, const MPIType& type) {
  const auto count       = type.combiner.integer_args[0];
  const auto blocklength = type.combiner.integer_args[1];
  const auto stride      = type.combiner.integer_args[2];

  if (stride < 0) {
    return make_internal_error<UnsupportedCombinerArgs>(
        "negative strides for MPI_Type_vector are currently not supported");
  }

  // MPI_Type_vector forms a number of `count` blocks of `oldtype` where the
  // start of each consecutive block is `stride` elements of `oldtype` apart
  // and each block consists of `blocklength` elements of oldtype.
  // We therefore check the buffer's type against `oldtype` and multiply the
  // resulting count by `(count - 1) * stride + blocklength`.
  return check_type(buffer, type.combiner.type_args[0]).map([&](auto multipliers) {
    return Multipliers{multipliers.type * ((count - 1) * stride + blocklength), multipliers.buffer};
  });
}

// Type check for the type combiner:
// int MPI_Type_create_indexed_block(int count, int blocklength, const int
//     array_of_displacements[], MPI_Datatype oldtype, MPI_Datatype *newtype)
//
// See MPICall::check_type(const Buffer&, const MPIType&) for an explanation of
// the arguments and the return type.
Result<Multipliers> check_combiner_indexed_block(const Buffer& buffer, const MPIType& type) {
  const auto count                  = type.combiner.integer_args[0];
  const auto blocklength            = type.combiner.integer_args[1];
  const auto array_of_displacements = type.combiner.integer_args.begin() + 2;
  const auto [min_displacement, max_displacement] =
      std::minmax_element(array_of_displacements, array_of_displacements + count);

  if (*min_displacement < 0) {
    return make_internal_error<UnsupportedCombinerArgs>(
        "negative displacements for MPI_Type_create_indexed_block are currently not supported");
  }

  // Similer to MPI_Type_vector but with a separate displacement specified for
  // each block.
  // We therefore check the buffer's type against `oldtype` and multiply the
  // resulting count by `max(array_of_displacements) + blocklength`.
  return check_type(buffer, type.combiner.type_args[0])
      .map([&, max_displacement = *max_displacement](auto multipliers) {
        return Multipliers{multipliers.type * (max_displacement + blocklength), multipliers.buffer};
      });
}

// Type check for the type combiner:
// int MPI_Type_create_struct(int count, int array_of_blocklengths[],
//     const MPI_Aint array_of_displacements[], const MPI_Datatype array_of_types[],
//     MPI_Datatype *newtype)
//
// See MPICall::check_type(const Buffer&, const MPIType&) for an explanation of
// the arguments and the return type.
Result<Multipliers> check_combiner_struct(const Buffer& buffer, const MPIType& type) {
  const auto count                 = type.combiner.integer_args[0];
  const auto array_of_blocklenghts = type.combiner.integer_args.begin() + 1;

  // First, check that the buffer's type is a struct type...
  typeart_struct_layout struct_layout;
  typeart_status status = typeart_resolve_type_id(buffer.type_id, &struct_layout);
  if (status == TYPEART_INVALID_ID) {
    auto message = fmt::format("Buffer::create received an invalid type_id {}", buffer.type_id);
    return make_internal_error<InvalidArgument>(message);
  }
  if (status == TYPEART_WRONG_KIND) {
    return make_type_error<BufferNotOfStructType>(buffer.type_id);
  }
  std::vector<Buffer> type_layout = {};
  if (status == TYPEART_OK) {
    type_layout.reserve(struct_layout.num_members);
    for (size_t i = 0; i < struct_layout.num_members; ++i) {
      auto subtype_buffer =
          Buffer::create(static_cast<ptrdiff_t>(struct_layout.offsets[i]), (char*)buffer.ptr + struct_layout.offsets[i],
                         struct_layout.count[i], struct_layout.member_types[i]);
      type_layout.push_back(subtype_buffer);
    }
  }

  // ... and that the number of members of the struct matches the argument
  // `count` of the type combiner.
  if (type_layout.size() != count) {
    return make_type_error<MemberCountMismatch>(buffer.type_id, type_layout.size(), count);
  }

  // Then, for each member check that...
  for (size_t i = 0; i < type_layout.size(); ++i) {
    // ... the byte offset of the member matches the respective element in
    // the `array_of_displacements` type combiner argument.
    if (type_layout[i].offset != type.combiner.address_args[i]) {
      return make_type_error<MemberOffsetMismatch>(buffer.type_id, i + 1, type_layout[i].offset,
                                                   type.combiner.address_args[i]);
    }
  }

  for (size_t i = 0; i < type_layout.size(); ++i) {
    // ... the type of the member matches the respective MPI type in the
    // `array_of_types` type combiner argument.
    auto result = check_type(type_layout[i], type.combiner.type_args[i]);
    if (result.has_error()) {
      auto error = std::move(result).error();
      if (error->is<InternalError>()) {
        return std::move(error);
      } else {
        return make_type_error<MemberTypeMismatch>(i + 1,
                                                   std::make_unique<TypeError>(std::move(*error).get<TypeError>()));
      }
    }

    // ... the count of elements in the buffer of the member matches the count
    // required to represent `blocklength` elements of the MPI type.
    const auto multipliers  = std::move(result).value();
    const auto type_count   = static_cast<size_t>(array_of_blocklenghts[i]) * multipliers.type;
    const auto buffer_count = type_layout[i].count * multipliers.buffer;
    if (type_count != buffer_count) {
      return make_type_error<MemberElementCountMismatch>(buffer.type_id, i + 1, type_count, buffer_count);
    }
  }
  return Multipliers{1, 1};
}

// Type check for the type combiner:
// int MPI_Type_create_subarray(int ndims, const int array_of_sizes[], const
//     int array_of_subsizes[], const int array_of_starts[], int order, MPI_Datatype
//     oldtype, MPI_Datatype *newtype)
//
// See MPICall::check_type(const Buffer&, const MPIType&) for an explanation of
// the arguments and the return type.
Result<Multipliers> check_combiner_subarray(const Buffer& buffer, const MPIType& type) {
  const auto ndims               = type.combiner.integer_args[0];
  const auto array_of_sizes      = type.combiner.integer_args.begin() + 1;
  const auto array_element_count = std::accumulate(array_of_sizes, array_of_sizes + ndims, 1, std::multiplies{});
  // As this type combiner specifies a subarray of a larger array, the buffer
  // must be large enough to hold that larger array. We therefore check the
  // buffer's type against `oldtype` and multiply the resulting count with
  // the product of all elements of the `array_of_sizes` (i.e. the element
  // count of the large n-dimensional array).
  return check_type(buffer, type.combiner.type_args[0]).map([&](auto multipliers) {
    return Multipliers{multipliers.type * array_element_count, multipliers.buffer};
  });
}

}  // namespace typeart
