#pragma once

#include "Util.h"
#include "runtime/RuntimeInterface.h"

#include <mpi.h>

#define MAX_SYMBOL_LENGTH 2048

typedef struct {
  const void* ptr;
  int count;
  int type_id;
  const char* type_name;
} MPIBufferInfo;

typedef struct {
  MPI_Datatype mpi_type;
  char name[MPI_MAX_OBJECT_NAME];
} MPITypeInfo;

typedef struct {
  const void* addr;
  const char* name;
} CallerInfo;

typedef struct {
  size_t trace_id;
  const char* function_name;
  CallerInfo caller;
  int is_send;
  int rank;
  MPIBufferInfo buffer;
  int count;
  MPITypeInfo type;
} MPICallInfo;

#define PRINT_INFOV(call, fmt, ...) fprintf(stderr, "[Info, r%d, id%ld] " fmt, call->rank, call->trace_id, __VA_ARGS__);

#define PRINT_ERRORV(call, fmt, ...) \
  fprintf(stderr, "[Error, r%d, id%ld] " fmt, call->rank, call->trace_id, __VA_ARGS__);

#define PRINT_ERROR(call, fmt) fprintf(stderr, "[Error, r%d, id%ld] " fmt, call->rank, call->trace_id);

int ta_create_caller_info(const void* caller_addr, CallerInfo* caller_info) {
  char* name;
  int result        = ta_get_symbol_name(caller_addr, &name);
  caller_info->name = name;
  caller_info->addr = caller_addr;
  return result;
}

int ta_create_buffer_info(const MPICallInfo* call, const void* buffer, MPIBufferInfo* buffer_info) {
  int ta_type_id;
  size_t ta_count                 = 0;
  typeart_status typeart_status_v = typeart_get_type(buffer, &ta_type_id, &ta_count);
  if (typeart_status_v != TA_OK) {
    const char* msg = ta_get_error_message(typeart_status_v);
    PRINT_ERRORV(call, "internal runtime error (%s)\n", msg);
    return -1;
  }
  const char* ta_type_name = typeart_get_type_name(ta_type_id);
  *buffer_info             = (MPIBufferInfo){buffer, ta_count, ta_type_id, ta_type_name};
  return 0;
}

int ta_create_type_info(const MPICallInfo* call, MPI_Datatype type, MPITypeInfo* type_info) {
  int len;
  int mpierr = MPI_Type_get_name(type, type_info->name, &len);
  if (mpierr != MPI_SUCCESS) {
    char mpierrstr[MPI_MAX_ERROR_STRING];
    MPI_Error_string(mpierr, mpierrstr, &len);
    PRINT_ERRORV(call, "MPI_Type_get_name failed: %s", mpierrstr);
    return -1;
  }
  type_info->mpi_type = type;
  return 0;
}

int ta_create_call_info(size_t trace_id, const char* function_name, const void* called_from, const void* buffer,
                        int is_const, int count, MPI_Datatype type, MPICallInfo* call_info) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPICallInfo result = {trace_id, function_name,     (CallerInfo){}, is_const,
                        rank,     (MPIBufferInfo){}, count,          (MPITypeInfo){}};
  if (ta_create_caller_info(called_from, &result.caller) != 0) {
    fprintf(stderr, "[Info, r%d, id%ld] couldn't resolve the symbol name for address %p", result.rank, result.trace_id,
            called_from);
  }
  if (ta_create_buffer_info(&result, buffer, &result.buffer) != 0) {
    return -1;
  }
  if (ta_create_type_info(&result, type, &result.type) != 0) {
    return -1;
  }
  *call_info = result;
  return 0;
}
