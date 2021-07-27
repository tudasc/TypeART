#pragma once

#include "Util.h"
#include "runtime/RuntimeInterface.h"

#include <mpi.h>
#include <stdio.h>

typedef struct {
  const void* ptr;
  int is_const;
  int count;
  int type_id;
  const char* type_name;
} MPIBufferInfo;

typedef struct {
  const char* function_name;
  const void* called_from;
  int rank;
  MPIBufferInfo buffer;
  int count;
  MPI_Datatype type;
} MPICallInfo;

int ta_create_call_info(const char* function_name, const void* called_from, const void* buffer, int is_const, int count,
                        MPI_Datatype type, MPICallInfo* call_info) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int ta_type_id;
  size_t ta_count                 = 0;
  typeart_status typeart_status_v = typeart_get_type(buffer, &ta_type_id, &ta_count);
  if (typeart_status_v != TA_OK) {
    const char* msg = ta_get_error_message(typeart_status_v);
    fprintf(stderr, "R[%d][Error][%d] %s: buffer %p at loc %p - %s\n", rank, is_const, function_name, buffer,
            called_from, msg);
    ta_print_loc(called_from);
    return -1;
  }
  const char* ta_type_name  = typeart_get_type_name(ta_type_id);
  MPIBufferInfo buffer_info = {buffer, is_const, ta_count, ta_type_id, ta_type_name};
  *call_info                = (MPICallInfo){function_name, called_from, rank, buffer_info, count, type};
  return 0;
}

#define PRINT_ERRORV(call, fmt, ...)                                                                                 \
  fprintf(stderr, "R[%d][Error][%d] %s: " fmt, call->rank, call->buffer.is_const, call->function_name, __VA_ARGS__); \
  ta_print_loc(call->called_from);

#define PRINT_ERROR(call, fmt)                                                                          \
  fprintf(stderr, "R[%d][Error][%d] %s: " fmt, call->rank, call->buffer.is_const, call->function_name); \
  ta_print_loc(call->called_from);

  const char* mpi_type_name = typeart_get_type_name(mpi_type_id);
  fprintf(stderr, "R[%d][Info][%d] %s: buffer %p has type %s, MPI type is %s\n", call->rank, call->buffer.is_const,
          call->function_name, call->buffer.ptr, call->buffer.type_name, mpi_type_name);
  if (call->buffer.type_id != mpi_type_id && !(call->buffer.type_id == TA_PPC_FP128 && mpi_type_id == TA_FP128)) {
    PRINT_ERRORV(call, "buffer %p at loc %p has type %s while the MPI type is %s\n", buffer->ptr, call->called_from,
                 buffer->type_name, mpi_type_name);
    return -1;
  }
}

int ta_check_type(const MPICallInfo* call, MPI_Datatype type, int* mpi_count) {
  int num_integers, num_addresses, num_datatypes, combiner;
  MPI_Type_get_envelope(type, &num_integers, &num_addresses, &num_datatypes, &combiner);
  int array_of_integers[num_integers];
  MPI_Aint array_of_addresses[num_addresses];
  MPI_Datatype array_of_datatypes[num_datatypes];
  if (combiner != MPI_COMBINER_NAMED) {
    MPI_Type_get_contents(type, num_integers, num_addresses, num_datatypes, array_of_integers, array_of_addresses,
                          array_of_datatypes);
  }
  switch (combiner) {
    case MPI_COMBINER_NAMED: {
      const int mpi_type_id = ta_mpi_type_to_type_id(type);
      if (mpi_type_id == -1) {
        PRINT_ERROR(call, "couldn't convert builtin type\n");
        return -1;
      }
      *mpi_count = 1;
      return ta_check_builtin_type(call, mpi_type_id);
    }
    case MPI_COMBINER_DUP:
      return ta_check_type(call, array_of_datatypes[0], mpi_count);
    case MPI_COMBINER_CONTIGUOUS: {
      int result = ta_check_type(call, array_of_datatypes[0], mpi_count);
      *mpi_count *= array_of_integers[0];
      return result;
    }
    case MPI_COMBINER_VECTOR: {
      int result = ta_check_type(call, array_of_datatypes[0], mpi_count);
      if (array_of_integers[2] < 0) {
        PRINT_ERROR(call, " negative strides for MPI_Type_vector are currently not supported\n");
        return -1;
      }
      // (count - 1) * stride + blocklength
      *mpi_count *= (array_of_integers[0] - 1) * array_of_integers[2] + array_of_integers[1];
      return result;
    }
    case MPI_COMBINER_INDEXED_BLOCK: {
      int result           = ta_check_type(call, array_of_datatypes[0], mpi_count);
      int max_displacement = 0;
      for (size_t i = 2; i < num_integers; ++i) {
        if (array_of_integers[i] > max_displacement) {
          max_displacement = array_of_integers[i];
        }
        if (array_of_integers[i] < 0) {
          PRINT_ERROR(call, "negative displacements for MPI_Type_create_indexed_block are currently not supported\n");
          return -1;
        }
      }
      // max(array_of_displacements) + blocklength
      *mpi_count *= max_displacement + array_of_integers[1];
      return result;
    }
    default:
      PRINT_ERRORV(call, "the MPI type combiner %s is currently not supported", ta_mpi_combiner_name(combiner));
  }
}

int ta_check_type_and_count(const MPICallInfo* call, MPI_Datatype type) {
  int mpi_type_count;
  if (ta_check_type(call, call->type, &mpi_type_count) == -1) {
    return -1;
  }
  if (call->count * mpi_type_count > call->buffer.count) {
    PRINT_ERRORV(call, "buffer %p too small. The buffer can only hold %d elements (%d required)\n", call->buffer.ptr,
                 (int)call->buffer.count, (int)call->count * mpi_type_count);
    return -1;
  }
}

#undef PRINT_ERROR
#undef PRINT_ERRORV
