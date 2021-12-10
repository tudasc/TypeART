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

#include "Util.h"

#include "TypeInterface.h"

#include <cstdint>
#include <cstdio>

namespace typeart {

const char* error_message_for(typeart_status status) {
  switch (status) {
    case TYPEART_OK:
      return "No errors";
    case TYPEART_UNKNOWN_ADDRESS:
      return "Buffer not registered";
    case TYPEART_BAD_ALIGNMENT:
      return "Buffer access is not aligned correctly";
    case TYPEART_BAD_OFFSET:
      return "Error in offset computation";
    case TYPEART_WRONG_KIND:
      return "Wrong type kind";
    case TYPEART_INVALID_ID:
      return "Invalid type ID";
    default:
      return "Invalid error code";
  }
}

template <class T>
int type_of() {
  static_assert(std::is_integral_v<T>);

  if constexpr (sizeof(T) == sizeof(int8_t)) {
    return TYPEART_INT8;
  } else if constexpr (sizeof(T) == sizeof(int16_t)) {
    return TYPEART_INT16;
  } else if constexpr (sizeof(T) == sizeof(int32_t)) {
    return TYPEART_INT32;
  } else if constexpr (sizeof(T) == sizeof(int64_t)) {
    return TYPEART_INT64;
  } else {
    fprintf(stderr, "[Error] Unsupperted integer width %lu!\n", sizeof(T));
    return TYPEART_UNKNOWN_TYPE;
  }
}

// Given a builtin MPI type, returns the corresponding TypeArt type.
// If the MPI type is a custom type, -1 is returned.
// Note: this function cannot distinguish between TYPEART_FP128 und TYPEART_PPC_TP128,
// therefore TYPEART_FP128 is always returned in case of an 16 byte floating point
// MPI type. This should be considered by the caller for performing typechecks.
int type_id_for(MPI_Datatype mpi_type) {
  if (mpi_type == MPI_BYTE) {
    return TYPEART_INT8;
  } else if (mpi_type == MPI_CHAR) {
    return type_of<char>();
  } else if (mpi_type == MPI_UNSIGNED_CHAR) {
    fprintf(stderr, "[Error] MPI_UNSIGNED_CHAR is currently unsupported!\n");
  } else if (mpi_type == MPI_SIGNED_CHAR) {
    fprintf(stderr, "[Error] MPI_SIGNED_CHAR is currently unsupported!\n");
  } else if (mpi_type == MPI_SHORT) {
    return type_of<short>();
  } else if (mpi_type == MPI_UNSIGNED_SHORT) {
    fprintf(stderr, "[Error] Unsigned integers are currently not supported!\n");
  } else if (mpi_type == MPI_INT) {
    return type_of<int>();
  } else if (mpi_type == MPI_UNSIGNED) {
    fprintf(stderr, "[Error] Unsigned integers are currently not supported!\n");
  } else if (mpi_type == MPI_LONG) {
    return type_of<long int>();
  } else if (mpi_type == MPI_UNSIGNED_LONG) {
    fprintf(stderr, "[Error] Unsigned integers are currently not supported!\n");
  } else if (mpi_type == MPI_LONG_LONG_INT) {
    return type_of<long long int>();
  } else if (mpi_type == MPI_FLOAT) {
    return TYPEART_FLOAT;
  } else if (mpi_type == MPI_DOUBLE) {
    return TYPEART_DOUBLE;
  } else if (mpi_type == MPI_LONG_DOUBLE) {
    if constexpr (sizeof(long double) == sizeof(double)) {
      return TYPEART_DOUBLE;
    } else if constexpr (sizeof(long double) == 10) {
      return TYPEART_X86_FP80;
    } else if constexpr (sizeof(long double) == 16) {
      return TYPEART_FP128;
    } else {
      fprintf(stderr, "[Error] long double has unexpected size %zu!\n", sizeof(long double));
    }
  }

  return TYPEART_UNKNOWN_TYPE;
}

const char* combiner_name_for(int combiner) {
  switch (combiner) {
    case MPI_COMBINER_NAMED:
      return "predefined type";
    case MPI_COMBINER_DUP:
      return "MPI_Type_dup";
    case MPI_COMBINER_CONTIGUOUS:
      return "MPI_Type_contiguous";
    case MPI_COMBINER_VECTOR:
      return "MPI_Type_vector";
    case MPI_COMBINER_HVECTOR:
      return "MPI_Type_hvector";
    case MPI_COMBINER_INDEXED:
      return "MPI_Type_indexed";
    case MPI_COMBINER_HINDEXED:
      return "MPI_Type_hindexed";
    case MPI_COMBINER_INDEXED_BLOCK:
      return "MPI_Type_create_indexed_block";
    case MPI_COMBINER_STRUCT:
      return "MPI_Type_struct";
    case MPI_COMBINER_SUBARRAY:
      return "MPI_Type_create_subarray";
    case MPI_COMBINER_DARRAY:
      return "MPI_Type_create_darray";
    case MPI_COMBINER_F90_REAL:
      return "MPI_Type_create_f90_real";
    case MPI_COMBINER_F90_COMPLEX:
      return "MPI_Type_create_f90_complex";
    case MPI_COMBINER_F90_INTEGER:
      return "MPI_Type_create_f90_integer";
    case MPI_COMBINER_RESIZED:
      return "MPI_Type_create_resized";
    default:
      return "invalid combiner id";
  }
}

}  // namespace typeart
