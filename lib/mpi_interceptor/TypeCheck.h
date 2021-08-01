#pragma once

#include "Types.h"
#include "Util.h"
#include "runtime/RuntimeInterface.h"

#include <mpi.h>
#include <stdio.h>

int ta_check_builtin_type(const MPICallInfo* call, const MPIBufferInfo* buffer, const MPITypeInfo* type) {
  const int mpi_type_id = ta_mpi_type_to_type_id(type->mpi_type);
  if (mpi_type_id == -1) {
    PRINT_ERROR(call, "couldn't convert builtin type\n");
    return -1;
  }
  if (buffer->type_id != mpi_type_id && !(buffer->type_id == TA_PPC_FP128 && mpi_type_id == TA_FP128)) {
    PRINT_ERRORV(call, "expected a type matching MPI type \"%s\", but found type \"%s\"\n", type->name,
                 buffer->type_name);
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
      *mpi_count = 1;
      return ta_check_builtin_type(call, buffer, type);
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
    case MPI_COMBINER_STRUCT: {
      typeart_struct_layout struct_layout;
      typeart_status status = typeart_resolve_type(buffer->type_id, &struct_layout);
      if (status != TA_OK) {
        PRINT_ERRORV(call, "expected a struct type, but found type \"%s\"\n", buffer->type_name);
        return -1;
      }
      if (struct_layout.len != array_of_integers[0]) {
        PRINT_ERRORV(call, "expected %d members, but the type \"%s\" has %ld members\n", array_of_integers[0],
                     buffer->type_name, struct_layout.len);
        return -1;
      }
      int result = 0;
      for (size_t i = 0; i < struct_layout.len; ++i) {
        if (struct_layout.offsets[i] != array_of_addresses[i]) {
          PRINT_ERRORV(call, "expected a byte offset of %ld for member %ld, but the type \"%s\" has an offset of %ld\n",
                       array_of_addresses[i], i + 1, buffer->type_name, struct_layout.offsets[i]);
          result = -1;
        }
      }
      for (size_t i = 0; i < struct_layout.len; ++i) {
        const void* member_ptr      = buffer->ptr + struct_layout.offsets[i];
        int member_type_id          = struct_layout.member_types[i];
        MPIBufferInfo member_buffer = {member_ptr, struct_layout.count[i], member_type_id,
                                       typeart_get_type_name(member_type_id)};
        MPITypeInfo member_type;
        ta_create_type_info(call, array_of_datatypes[i], &member_type);
        int member_element_count;
        if (ta_check_type(call, &member_buffer, &member_type, &member_element_count) != 0) {
          result = -1;
          PRINT_ERRORV(call, "the typechek for member %ld failed\n", i + 1);
        }
        if (member_buffer.count * member_element_count != array_of_integers[i + 1]) {
          result = -1;
          PRINT_ERRORV(call, "expected element count of %d for member %ld, but the type \"%s\" has a count of %d\n",
                       array_of_integers[i + 1], i + 1, buffer->type_name, member_buffer.count * member_element_count);
        }
      }
      *mpi_count = 1;
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
    PRINT_ERRORV(call, "buffer too small (%d elements, %d required)\n", (int)call->buffer.count,
                 (int)call->count * mpi_type_count);
    return -1;
  }
}
