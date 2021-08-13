#include "TypeCheck.h"

namespace typeart {

void printMPIError(const MPICall* call, const char* fnname, int mpierr) {
  int len;
  char mpierrstr[MPI_MAX_ERROR_STRING];
  MPI_Error_string(mpierr, mpierrstr, &len);
  PRINT_ERRORV(call, "%s failed: %s", fnname, mpierrstr);
}

std::optional<Buffer> Buffer::create(const MPICall* call, const void* buffer) {
  int ta_type_id;
  size_t ta_count       = 0;
  auto typeart_status_v = typeart_get_type(buffer, &ta_type_id, &ta_count);
  if (typeart_status_v != TA_OK) {
    const char* msg = error_message_for(typeart_status_v);
    PRINT_ERRORV(call, "internal runtime error (%s)\n", msg);
    return {};
  }
  const char* ta_type_name = typeart_get_type_name(ta_type_id);
  return {{buffer, ta_count, ta_type_id, ta_type_name}};
}

std::optional<Buffer> Buffer::create(const MPICall* call, const void* ptr, size_t count, int type_id) {
  auto type_name = typeart_get_type_name(type_id);
  return {{ptr, count, type_id, type_name}};
}

std::optional<MPIType> MPIType::create(const MPICall* call, MPI_Datatype type) {
  auto result = MPIType{};
  int len;
  int mpierr = MPI_Type_get_name(type, result.name, &len);
  if (mpierr != MPI_SUCCESS) {
    printMPIError(call, "MPI_Type_get_name", mpierr);
    return {};
  }
  result.mpi_type = type;
  return {result};
}

std::optional<Caller> Caller::create(const void* caller_addr) {
  char* name;
  if (!get_symbol_name(caller_addr, &name)) {
    return {};
  } else {
    return {{caller_addr, name}};
  }
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

int ta_check_builtin_type(const MPICall* call, const Buffer* buffer, const MPIType* type) {
  const int mpi_type_id = type_id_for(type->mpi_type);
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

int ta_check_type(const MPICall* call, const Buffer* buffer, const MPIType* type, int* mpi_count) {
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
      auto type = MPIType::create(call, array_of_datatypes[0]);
      if (!type) {
        return -1;
      }
      return ta_check_type(call, buffer, &*type, mpi_count);
    }
    case MPI_COMBINER_CONTIGUOUS: {
      auto type = MPIType::create(call, array_of_datatypes[0]);
      if (!type) {
        return -1;
      }
      int result = ta_check_type(call, buffer, &*type, mpi_count);
      *mpi_count *= array_of_integers[0];
      return result;
    }
    case MPI_COMBINER_VECTOR: {
      auto type = MPIType::create(call, array_of_datatypes[0]);
      if (!type) {
        return -1;
      }
      int result = ta_check_type(call, buffer, &*type, mpi_count);
      if (array_of_integers[2] < 0) {
        PRINT_ERROR(call, "negative strides for MPI_Type_vector are currently not supported\n");
        return -1;
      }
      // (count - 1) * stride + blocklength
      *mpi_count *= (array_of_integers[0] - 1) * array_of_integers[2] + array_of_integers[1];
      return result;
    }
    case MPI_COMBINER_INDEXED_BLOCK: {
      auto type = MPIType::create(call, array_of_datatypes[0]);
      if (!type) {
        return -1;
      }
      int result           = ta_check_type(call, buffer, &*type, mpi_count);
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
        const void* member_ptr = (char*)buffer->ptr + struct_layout.offsets[i];
        int member_type_id     = struct_layout.member_types[i];
        auto member_buffer     = Buffer::create(call, member_ptr, struct_layout.count[i], member_type_id);
        if (!member_buffer) {
          return -1;
        }
        auto member_type = MPIType::create(call, array_of_datatypes[i]);
        if (!member_type) {
          return -1;
        }
        int member_element_count;
        if (ta_check_type(call, &*member_buffer, &*member_type, &member_element_count) != 0) {
          result = -1;
          PRINT_ERRORV(call, "the typechek for member %ld failed\n", i + 1);
        }
        if (member_buffer->count * member_element_count != array_of_integers[i + 1]) {
          result = -1;
          PRINT_ERRORV(call, "expected element count of %d for member %ld, but the type \"%s\" has a count of %ld\n",
                       array_of_integers[i + 1], i + 1, buffer->type_name, member_buffer->count * member_element_count);
        }
      }
      *mpi_count = 1;
      return result;
    }
    default:
      PRINT_ERRORV(call, "the MPI type combiner %s is currently not supported\n", combiner_name_for(combiner));
  }
  return -1;
}

int MPICall::check_type_and_count() const {
  int mpi_type_count;
  if (ta_check_type(this, &buffer, &type, &mpi_type_count) != 0) {
    return -1;
  }
  if (count * mpi_type_count > buffer.count) {
    PRINT_ERRORV(this, "buffer too small (%ld elements, %d required)\n", buffer.count, count * mpi_type_count);
    return -1;
  }
  return 0;
}

}  // namespace typeart
