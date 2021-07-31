#pragma once

#include "Types.h"
#include "Util.h"
#include "runtime/RuntimeInterface.h"

#include <mpi.h>
#include <stdio.h>

int ta_check_builtin_type(const MPICallInfo* call, const MPIBufferInfo* buffer, int mpi_type_id) {
  const char* mpi_type_name = typeart_get_type_name(mpi_type_id);
  PRINT_INFOV(call, "buffer %p has type %s, MPI type is %s\n", buffer->ptr, buffer->type_name, mpi_type_name);
  if (buffer->type_id != mpi_type_id && !(buffer->type_id == TA_PPC_FP128 && mpi_type_id == TA_FP128)) {
    PRINT_ERRORV(call, "buffer %p has type %s while the MPI type is %s\n", buffer->ptr, buffer->type_name,
                 mpi_type_name);
    return -1;
  }
  return 0;
}

int ta_check_type(const MPICallInfo* call, const MPIBufferInfo* buffer, const MPITypeInfo* type, int* mpi_count) {
  int num_integers, num_addresses, num_datatypes, combiner;
  MPI_Type_get_envelope(type->mpi_type, &num_integers, &num_addresses, &num_datatypes, &combiner);
  int array_of_integers[num_integers];
  MPI_Aint array_of_addresses[num_addresses];
  MPI_Datatype array_of_datatypes[num_datatypes];
  if (combiner != MPI_COMBINER_NAMED) {
    MPI_Type_get_contents(type->mpi_type, num_integers, num_addresses, num_datatypes, array_of_integers,
                          array_of_addresses, array_of_datatypes);
  }
  switch (combiner) {
    case MPI_COMBINER_NAMED: {
      const int mpi_type_id = ta_mpi_type_to_type_id(type->mpi_type);
      if (mpi_type_id == -1) {
        PRINT_ERROR(call, "couldn't convert builtin type\n");
        return -1;
      }
      *mpi_count = 1;
      return ta_check_builtin_type(call, buffer, mpi_type_id);
    }
    case MPI_COMBINER_DUP: {
      MPITypeInfo type_info;
      ta_create_type_info(call, array_of_datatypes[0], &type_info);
      return ta_check_type(call, buffer, &type_info, mpi_count);
    }
    case MPI_COMBINER_CONTIGUOUS: {
      MPITypeInfo type_info;
      ta_create_type_info(call, array_of_datatypes[0], &type_info);
      int result = ta_check_type(call, buffer, &type_info, mpi_count);
      *mpi_count *= array_of_integers[0];
      return result;
    }
    case MPI_COMBINER_VECTOR: {
      MPITypeInfo type_info;
      ta_create_type_info(call, array_of_datatypes[0], &type_info);
      int result = ta_check_type(call, buffer, &type_info, mpi_count);
      if (array_of_integers[2] < 0) {
        PRINT_ERROR(call, "negative strides for MPI_Type_vector are currently not supported\n");
        return -1;
      }
      // (count - 1) * stride + blocklength
      *mpi_count *= (array_of_integers[0] - 1) * array_of_integers[2] + array_of_integers[1];
      return result;
    }
    case MPI_COMBINER_INDEXED_BLOCK: {
      MPITypeInfo type_info;
      ta_create_type_info(call, array_of_datatypes[0], &type_info);
      int result           = ta_check_type(call, buffer, &type_info, mpi_count);
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
      PRINT_ERRORV(call, "the MPI type combiner %s is currently not supported\n", ta_mpi_combiner_name(combiner));
  }
  return -1;
}

int ta_check_type_and_count(const MPICallInfo* call) {
  int mpi_type_count;
  if (ta_check_type(call, &call->buffer, &call->type, &mpi_type_count) != 0) {
    return -1;
  }
  if (call->count * mpi_type_count > call->buffer.count) {
    PRINT_ERRORV(call, "buffer %p too small. The buffer can only hold %d elements (%d required)\n", call->buffer.ptr,
                 (int)call->buffer.count, (int)call->count * mpi_type_count);
    return -1;
  }
}
