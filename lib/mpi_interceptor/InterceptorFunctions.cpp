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

#include "Logger.h"
#include "Stats.h"
#include "System.h"
#include "TypeCheck.h"

#include <bits/types/struct_rusage.h>
#include <fmt/printf.h>
#include <mpi.h>
#include <optional>
#include <sys/resource.h>

namespace typeart {

void check_buffer(const char* name, const void* called_from, bool is_send, const void* ptr, int count,
                  MPI_Datatype type);

static CallCounter call_counter;
static MPICounter mpi_counter;

}  // namespace typeart

extern "C" {

void typeart_check_send(const char* name, const void* called_from, const void* sendbuf, int count, MPI_Datatype dtype) {
  ++typeart::call_counter.send;
  typeart::check_buffer(name, called_from, true, sendbuf, count, dtype);
}

void typeart_check_recv(const char* name, const void* called_from, void* recvbuf, int count, MPI_Datatype dtype) {
  ++typeart::call_counter.recv;
  typeart::check_buffer(name, called_from, false, recvbuf, count, dtype);
}

void typeart_check_send_and_recv(const char* name, const void* called_from, const void* sendbuf, int sendcount,
                                 MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype) {
  ++typeart::call_counter.send_recv;
  typeart_check_send(name, called_from, sendbuf, sendcount, sendtype);
  typeart_check_recv(name, called_from, recvbuf, recvcount, recvtype);
}

void typeart_unsupported_mpi_call(const char* name, const void* /*called_from*/) {
  ++typeart::call_counter.unsupported;
}

void typeart_exit() {
  // Called at MPI_Finalize time
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  struct rusage end;
  getrusage(RUSAGE_SELF, &end);
  typeart::logger::call_counter(typeart::call_counter, end.ru_maxrss);
  typeart::logger::mpi_counter(typeart::mpi_counter);
}

}  // extern "C"

namespace typeart {

void check_buffer(const char* name, const void* called_from, bool is_send, const void* ptr, int count,
                  MPI_Datatype type) {
  const bool count_is_zero     = count <= 0;
  const bool buffer_is_nullptr = ptr == nullptr;

  if (count_is_zero) {
    ++mpi_counter.null_count;
    return;
  }

  if (buffer_is_nullptr) {
    ++mpi_counter.null_buff;
    logger::null_buffer();
    return;
  }

  auto buffer = Buffer::create(ptr);
  if (buffer.has_error()) {
    ++mpi_counter.error;
    logger::error(name, called_from, *std::move(buffer).error());
    return;
  }
  auto mpi_type = MPIType::create(type);
  if (mpi_type.has_error()) {
    ++mpi_counter.error;
    logger::error(name, called_from, *std::move(mpi_type).error());
    return;
  }

  auto result = check_buffer(*buffer, *mpi_type, count);
  if (result.has_error()) {
    ++mpi_counter.type_error;
  }
  logger::result(name, called_from, is_send, *buffer, *mpi_type, result);
}

}  // namespace typeart