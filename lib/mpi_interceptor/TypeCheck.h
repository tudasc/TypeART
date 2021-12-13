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

#ifndef TYPEART_MPI_INTERCEPTOR_TYPE_CHECK_H
#define TYPEART_MPI_INTERCEPTOR_TYPE_CHECK_H

#include "System.h"

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <mpi.h>
#include <optional>
#include <string>
#include <vector>

namespace typeart {

#ifndef NDEBUG
#define PRINT_INFOV(call, fmt, ...) \
  fprintf(stderr, "R[%d][Info]ID[%ld] " fmt, (call).rank, (call).trace_id, __VA_ARGS__)
#else
#define PRINT_INFOV(call, fmt, ...)
#endif

#define PRINT_WARNING(call, fmt) fprintf(stderr, "R[%d][Warning]ID[%ld] " fmt, (call).rank, (call).trace_id)

#define PRINT_ERRORV(call, fmt, ...) \
  fprintf(stderr, "R[%d][Error]ID[%ld] " fmt, (call).rank, (call).trace_id, __VA_ARGS__)

#define PRINT_ERROR(call, fmt) fprintf(stderr, "R[%d][Error]ID[%ld] " fmt, (call).rank, (call).trace_id)

#define PRINT_TRACEV(call, fmt, ...) \
  fprintf(stderr, "R[%d][Trace]ID[%ld] " fmt, (call).rank, (call).trace_id, __VA_ARGS__)

#define PRINT_TRACE(call, fmt) fprintf(stderr, "R[%d][Trace]ID[%ld] " fmt, (call).rank, (call).trace_id)

struct MPICall;
struct MPIType;

struct Type {
  int id;
  std::string name;
  size_t size;

 public:
  static std::optional<Type> create(const MPICall& call, int type_id);
};

struct Buffer {
  ptrdiff_t offset;
  const void* ptr;
  size_t count;
  Type type;
  std::optional<std::vector<Buffer>> type_layout;

 public:
  static std::optional<Buffer> create(const MPICall& call, const void* buffer);
  static std::optional<Buffer> create(const MPICall& call, ptrdiff_t offset, const void* ptr, size_t count,
                                      int type_id);

  [[nodiscard]] bool hasStructType() const;
};

struct MPICombiner {
  int id;
  std::vector<int> integer_args;
  std::vector<MPI_Aint> address_args;
  std::vector<MPIType> type_args;

 public:
  static std::optional<MPICombiner> create(const MPICall& call, MPI_Datatype type);
};

struct MPIType {
  MPI_Datatype mpi_type;
  int type_id;
  std::string name;
  MPICombiner combiner;

 public:
  static std::optional<MPIType> create(const MPICall& call, MPI_Datatype type);
};

struct MPICallArguments {
  Buffer buffer;
  int count;
  MPIType type;
};

struct MPICall {
  size_t trace_id;
  const void* caller;
  std::string function_name;
  int is_send;
  int rank;
  MPICallArguments args;

 public:
  static std::optional<MPICall> create(const char* function_name, const void* called_from, const void* buffer,
                                       int is_const, int count, MPI_Datatype type);

  [[nodiscard]] int check_type_and_count() const;

 private:
  struct [[nodiscard]] CheckResult {
    int result;
    int type_count_multiplier;
    int buffer_count_multiplier;

    static CheckResult error();
    static CheckResult with_multipliers(int type, int buffer);

    CheckResult& multiply_type_count_by(int rhs);
  };

  [[nodiscard]] int check_type_and_count_against(const Buffer& buffer) const;
  CheckResult check_type(const Buffer& buffer, const MPIType& type) const;
  CheckResult check_combiner_named(const Buffer& buffer, const MPIType& type) const;
  CheckResult check_combiner_contiguous(const Buffer& buffer, const MPIType& type) const;
  CheckResult check_combiner_vector(const Buffer& buffer, const MPIType& type) const;
  CheckResult check_combiner_indexed_block(const Buffer& buffer, const MPIType& type) const;
  CheckResult check_combiner_struct(const Buffer& buffer, const MPIType& type) const;
  CheckResult check_combiner_subarray(const Buffer& buffer, const MPIType& type) const;

  static std::atomic_size_t next_trace_id;
};

}  // namespace typeart

#endif  // TYPEART_MPI_INTERCEPTOR_TYPE_CHECK_H
