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

#pragma once

#include "Util.h"
#include "runtime/RuntimeInterface.h"

#include <atomic>
#include <cstdio>
#include <mpi.h>
#include <optional>
#include <vector>

namespace typeart {

#define PRINT_INFOV(call, fmt, ...) \
  fprintf(stderr, "R[%d][Info]ID[%ld] " fmt, (call)->rank, (call)->trace_id, __VA_ARGS__)

#define PRINT_ERRORV(call, fmt, ...) \
  fprintf(stderr, "R[%d][Error]ID[%ld] " fmt, (call)->rank, (call)->trace_id, __VA_ARGS__)

#define PRINT_ERROR(call, fmt) fprintf(stderr, "R[%d][Error]ID[%ld] " fmt, (call)->rank, (call)->trace_id)

struct MPICall;
struct MPIType;

struct Buffer {
  ptrdiff_t offset;
  const void* ptr;
  size_t count;
  int type_id;
  std::string type_name;
  std::optional<std::vector<Buffer>> type_layout;

 public:
  static std::optional<Buffer> create(const MPICall* call, const void* buffer);
  static std::optional<Buffer> create(const MPICall* call, ptrdiff_t offset, const void* ptr, size_t count,
                                      int type_id);

  [[nodiscard]] bool hasStructType() const;
};

struct MPICombiner {
  int id;
  std::vector<int> integer_args;
  std::vector<MPI_Aint> address_args;
  std::vector<MPIType> type_args;

 public:
  static std::optional<MPICombiner> create(const MPICall* call, MPI_Datatype type);
};

struct MPIType {
  MPI_Datatype mpi_type;
  int type_id;
  std::string name;
  MPICombiner combiner;

 public:
  static std::optional<MPIType> create(const MPICall* call, MPI_Datatype type);
};

struct Caller {
  const void* addr;
  std::string name;

 public:
  static std::optional<Caller> create(const void* caller_addr);
};

struct MPICall {
  size_t trace_id;
  std::string function_name;
  Caller caller;
  int is_send;
  int rank;
  Buffer buffer;
  int count;
  MPIType type;

 public:
  static std::optional<MPICall> create(const char* function_name, const void* called_from, const void* buffer,
                                       int is_const, int count, MPI_Datatype type);

  [[nodiscard]] int check_type_and_count() const;

 private:
  int check_type_and_count(const Buffer* buffer) const;
  int check_type(const Buffer* buffer, const MPIType* type, int* mpi_count) const;
  int check_combiner_named(const Buffer* buffer, const MPIType* type, int* mpi_count) const;
  int check_combiner_contiguous(const Buffer* buffer, const MPIType* type, int* mpi_count) const;
  int check_combiner_vector(const Buffer* buffer, const MPIType* type, int* mpi_count) const;
  int check_combiner_indexed_block(const Buffer* buffer, const MPIType* type, int* mpi_count) const;
  int check_combiner_struct(const Buffer* buffer, const MPIType* type, int* mpi_count) const;
  int check_combiner_subarray(const Buffer* buffer, const MPIType* type, int* mpi_count) const;

  static std::atomic_size_t next_trace_id;
};

}  // namespace typeart
