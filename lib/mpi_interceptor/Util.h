#pragma once

#include "runtime/RuntimeInterface.h"

#include <mpi.h>
#include <stdio.h>

const char* ta_get_error_message(typeart_status status) {
  switch (status) {
    case TA_OK:
      return "No errors";
    case TA_UNKNOWN_ADDRESS:
      return "Buffer not registered";
    case TA_BAD_ALIGNMENT:
      return "Buffer access is not aligned correctly";
    case TA_BAD_OFFSET:
      return "Error in offset computation";
    case TA_WRONG_KIND:
      return "Wrong type kind";
    case TA_INVALID_ID:
      return "Invalid type ID";
    default:
      return "Invalid error code";
  }
}

int ta_mpi_map_int_type(size_t int_sizeof) {
  if (int_sizeof == 1) {
    return TA_INT8;
  } else if (int_sizeof == 2) {
    return TA_INT16;
  } else if (int_sizeof == 4) {
    return TA_INT32;
  } else if (int_sizeof == 8) {
    return TA_INT64;
  } else {
    fprintf(stderr, "[Error] Unsupperted integer width %lu!\n", int_sizeof);
    return TA_UNKNOWN_TYPE;
  }
}

// Given a builtin MPI type, returns the corresponding TypeArt type.
// If the MPI type is a custom type, -1 is returned.
// Note: this function cannot distinguish between TA_FP128 und TA_PPC_TP128,
// therefore TA_FP128 is always returned in case of an 16 byte floating point
// MPI type. This should be considered by the caller for performing typechecks.
int ta_mpi_type_to_type_id(MPI_Datatype mpi_type) {
  if (mpi_type == MPI_CHAR) {
    fprintf(stderr, "[Error] MPI_CHAR is currently unsupported!\n");
  } else if (mpi_type == MPI_UNSIGNED_CHAR) {
    fprintf(stderr, "[Error] MPI_UNSIGNED_CHAR is currently unsupported!\n");
  } else if (mpi_type == MPI_SIGNED_CHAR) {
    fprintf(stderr, "[Error] MPI_SIGNED_CHAR is currently unsupported!\n");
  } else if (mpi_type == MPI_SHORT) {
    return ta_mpi_map_int_type(sizeof(short));
  } else if (mpi_type == MPI_UNSIGNED_SHORT) {
    fprintf(stderr, "[Error] Unsigned integers are currently not supported!\n");
  } else if (mpi_type == MPI_INT) {
    return ta_mpi_map_int_type(sizeof(int));
  } else if (mpi_type == MPI_UNSIGNED) {
    fprintf(stderr, "[Error] Unsigned integers are currently not supported!\n");
  } else if (mpi_type == MPI_LONG) {
    return ta_mpi_map_int_type(sizeof(long int));
  } else if (mpi_type == MPI_UNSIGNED_LONG) {
    fprintf(stderr, "[Error] Unsigned integers are currently not supported!\n");
  } else if (mpi_type == MPI_LONG_LONG_INT) {
    return ta_mpi_map_int_type(sizeof(long long int));
  } else if (mpi_type == MPI_FLOAT) {
    return TA_FLOAT;
  } else if (mpi_type == MPI_DOUBLE) {
    return TA_DOUBLE;
  } else if (mpi_type == MPI_LONG_DOUBLE) {
    if (sizeof(long double) == sizeof(double)) {
      return TA_DOUBLE;
    } else if (sizeof(long double) == 10) {
      return TA_X86_FP80;
    } else if (sizeof(long double) == 16) {
      return TA_FP128;
    } else {
      fprintf(stderr, "[Error] long double has unexpected size %zu!\n", sizeof(long double));
    }
  } else {
    return -1;
  }
  return TA_UNKNOWN_TYPE;
}

const char* ta_mpi_combiner_name(int combiner) {
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

int ta_get_symbol_name(const void* call_adr, char** buffer) {
  const char* exe = getenv("TA_EXE_TARGET");
  if (exe == NULL || exe[0] == '\0') {
    return -1;
  }

  char cmd_buf[512] = {0};
  snprintf(cmd_buf, sizeof(cmd_buf), "addr2line -e %s -f %p", exe, call_adr);

  FILE* fp   = popen(cmd_buf, "r");
  int result = -1;
  if (fp) {
    size_t len = 0;
    if (getline(buffer, &len, fp)) {
      for (size_t i = 0; i < len; ++i) {
        if ((*buffer)[i] == '\n') {
          (*buffer)[i] = '\0';
          break;
        }
      }
      result = 0;
    }
  }
  pclose(fp);
  return result;
}
