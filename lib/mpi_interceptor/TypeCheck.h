#pragma once

#include "Util.h"
#include "runtime/RuntimeInterface.h"

#include <atomic>
#include <mpi.h>
#include <optional>
#include <stdio.h>
#include <vector>

namespace typeart {

#define MAX_SYMBOL_LENGTH 2048

#define PRINT_INFOV(call, fmt, ...) fprintf(stderr, "R[%d][Info]ID[%ld] " fmt, call->rank, call->trace_id, __VA_ARGS__);

#define PRINT_ERRORV(call, fmt, ...) \
  fprintf(stderr, "R[%d][Error]ID[%ld] " fmt, call->rank, call->trace_id, __VA_ARGS__);

#define PRINT_ERROR(call, fmt) fprintf(stderr, "R[%d][Error]ID[%ld] " fmt, call->rank, call->trace_id);

struct MPICall;

struct Buffer {
  const void* ptr;
  size_t count;
  int type_id;
  const char* type_name;

 public:
  static std::optional<Buffer> create(const MPICall* call, const void* buffer);
  static std::optional<Buffer> create(const MPICall* call, const void* ptr, size_t count, int type_id);
};

struct MPIType {
  MPI_Datatype mpi_type;
  char name[MPI_MAX_OBJECT_NAME];

 public:
  static std::optional<MPIType> create(const MPICall* call, MPI_Datatype type);
};

struct Caller {
  const void* addr;
  const char* name;

 public:
  static std::optional<Caller> create(const void* caller_addr);
};

struct MPICall {
  size_t trace_id;
  const char* function_name;
  Caller caller;
  int is_send;
  int rank;
  Buffer buffer;
  int count;
  MPIType type;

 public:
  static std::optional<MPICall> create(const char* function_name, const void* called_from, const void* buffer,
                                       int is_const, int count, MPI_Datatype type);

  int check_type_and_count() const;

 private:
  static std::atomic_size_t next_trace_id;
};

}  // namespace typeart
