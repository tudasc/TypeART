#include "TypeCheck.h"

namespace typeart {

void printMPIError(const MPICall* call, const char* fnname, int mpierr) {
  int len;
  char mpierrstr[MPI_MAX_ERROR_STRING];
  MPI_Error_string(mpierr, mpierrstr, &len);
  PRINT_ERRORV(call, "%s failed: %s", fnname, mpierrstr);
}

std::optional<Buffer> Buffer::create(const MPICall* call, const void* buffer) {
  int type_id;
  size_t count          = 0;
  auto typeart_status_v = typeart_get_type(buffer, &type_id, &count);
  if (typeart_status_v != TA_OK) {
    const char* msg = error_message_for(typeart_status_v);
    PRINT_ERRORV(call, "internal runtime error (%s)\n", msg);
    return {};
  }
  const char* type_name = typeart_get_type_name(type_id);
  return {{buffer, count, type_id, type_name}};
}

std::optional<Buffer> Buffer::create(const MPICall* call, const void* ptr, size_t count, int type_id) {
  auto type_name = typeart_get_type_name(type_id);
  return {{ptr, count, type_id, type_name}};
}

std::optional<MPICombiner> MPICombiner::create(const MPICall* call, MPI_Datatype type) {
  auto result = MPICombiner{};
  int num_integers, num_addresses, num_datatypes, combiner;
  auto mpierr = MPI_Type_get_envelope(type, &num_integers, &num_addresses, &num_datatypes, &combiner);
  if (mpierr != MPI_SUCCESS) {
    printMPIError(call, "MPI_Type_get_envelope", mpierr);
    return {};
  }
  result.id = combiner;
  if (combiner != MPI_COMBINER_NAMED) {
    result.integer_args.resize(num_integers);
    result.address_args.resize(num_addresses);
    MPI_Datatype type_args[num_datatypes];
    mpierr = MPI_Type_get_contents(type, num_integers, num_addresses, num_datatypes, result.integer_args.data(),
                                   result.address_args.data(), type_args);
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
  auto result = MPIType{type, "", *combiner};
  int len;
  int mpierr = MPI_Type_get_name(type, result.name, &len);
  if (mpierr != MPI_SUCCESS) {
    printMPIError(call, "MPI_Type_get_name", mpierr);
    return {};
  }
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

int MPICall::check_type_and_count() const {
  int mpi_type_count;
  if (check_type(&buffer, &type, &mpi_type_count) != 0) {
    return -1;
  }
  if (count * mpi_type_count > buffer.count) {
    PRINT_ERRORV(this, "buffer too small (%ld elements, %d required)\n", buffer.count, count * mpi_type_count);
    return -1;
  }
  return 0;
}

int MPICall::check_type(const Buffer* buffer, const MPIType* type, int* mpi_count) const {
  switch (type->combiner.id) {
    case MPI_COMBINER_NAMED:
      return check_combiner_named(buffer, type, mpi_count);
    case MPI_COMBINER_DUP:
      return check_type(buffer, &type->combiner.type_args[0], mpi_count);
    case MPI_COMBINER_CONTIGUOUS:
      return check_combiner_contiguous(buffer, type, mpi_count);
    case MPI_COMBINER_VECTOR:
      return check_combiner_vector(buffer, type, mpi_count);
    case MPI_COMBINER_INDEXED_BLOCK:
      return check_combiner_indexed_block(buffer, type, mpi_count);
    case MPI_COMBINER_STRUCT:
      return check_combiner_struct(buffer, type, mpi_count);
    default:
      PRINT_ERRORV(this, "the MPI type combiner %s is currently not supported\n", combiner_name_for(type->combiner.id));
  }
  return -1;
}

int MPICall::check_combiner_named(const Buffer* buffer, const MPIType* type, int* mpi_count) const {
  const int mpi_type_id = type_id_for(type->mpi_type);
  if (mpi_type_id == -1) {
    PRINT_ERROR(this, "couldn't convert builtin type\n");
    return -1;
  }
  if (buffer->type_id != mpi_type_id && !(buffer->type_id == TA_PPC_FP128 && mpi_type_id == TA_FP128)) {
    PRINT_ERRORV(this, "expected a type matching MPI type \"%s\", but found type \"%s\"\n", type->name,
                 buffer->type_name);
    return -1;
  }
  *mpi_count = 1;
  return 0;
}

int MPICall::check_combiner_contiguous(const Buffer* buffer, const MPIType* type, int* mpi_count) const {
  auto result = check_type(buffer, &type->combiner.type_args[0], mpi_count);
  *mpi_count *= type->combiner.integer_args[0];
  return result;
}

int MPICall::check_combiner_vector(const Buffer* buffer, const MPIType* type, int* mpi_count) const {
  auto result        = check_type(buffer, &type->combiner.type_args[0], mpi_count);
  auto& integer_args = type->combiner.integer_args;
  if (integer_args[2] < 0) {
    PRINT_ERROR(this, "negative strides for MPI_Type_vector are currently not supported\n");
    return -1;
  }
  // (count - 1) * stride + blocklength
  *mpi_count *= (integer_args[0] - 1) * integer_args[2] + integer_args[1];
  return result;
}

int MPICall::check_combiner_indexed_block(const Buffer* buffer, const MPIType* type, int* mpi_count) const {
  auto& integer_args    = type->combiner.integer_args;
  auto result           = check_type(buffer, &type->combiner.type_args[0], mpi_count);
  auto max_displacement = 0;
  for (size_t i = 2; i < integer_args.size(); ++i) {
    if (integer_args[i] > max_displacement) {
      max_displacement = integer_args[i];
    }
    if (integer_args[i] < 0) {
      PRINT_ERROR(this, "negative displacements for MPI_Type_create_indexed_block are currently not supported\n");
      return -1;
    }
  }
  // max(array_of_displacements) + blocklength
  *mpi_count *= max_displacement + integer_args[1];
  return result;
}

int MPICall::check_combiner_struct(const Buffer* buffer, const MPIType* type, int* mpi_count) const {
  auto& integer_args = type->combiner.integer_args;
  typeart_struct_layout struct_layout;
  typeart_status status = typeart_resolve_type(buffer->type_id, &struct_layout);
  if (status != TA_OK) {
    PRINT_ERRORV(this, "expected a struct type, but found type \"%s\"\n", buffer->type_name);
    return -1;
  }
  if (struct_layout.len != integer_args[0]) {
    PRINT_ERRORV(this, "expected %d members, but the type \"%s\" has %ld members\n", integer_args[0], buffer->type_name,
                 struct_layout.len);
    return -1;
  }
  int result = 0;
  for (size_t i = 0; i < struct_layout.len; ++i) {
    if (struct_layout.offsets[i] != integer_args[i]) {
      PRINT_ERRORV(this, "expected a byte offset of %ld for member %ld, but the type \"%s\" has an offset of %ld\n",
                   type->combiner.address_args[i], i + 1, buffer->type_name, struct_layout.offsets[i]);
      result = -1;
    }
  }
  for (size_t i = 0; i < struct_layout.len; ++i) {
    const void* member_ptr = (char*)buffer->ptr + struct_layout.offsets[i];
    int member_type_id     = struct_layout.member_types[i];
    auto member_buffer     = Buffer::create(this, member_ptr, struct_layout.count[i], member_type_id);
    if (!member_buffer) {
      return -1;
    }
    int member_element_count;
    if (check_type(&*member_buffer, &type->combiner.type_args[i], &member_element_count) != 0) {
      result = -1;
      PRINT_ERRORV(this, "the typechek for member %ld failed\n", i + 1);
    }
    if (member_buffer->count * member_element_count != integer_args[i + 1]) {
      result = -1;
      PRINT_ERRORV(this, "expected element count of %d for member %ld, but the type \"%s\" has a count of %ld\n",
                   integer_args[i + 1], i + 1, buffer->type_name, member_buffer->count * member_element_count);
    }
  }
  *mpi_count = 1;
  return result;
}

}  // namespace typeart
