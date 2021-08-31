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

#include "TypeCheck.h"
#include "runtime/RuntimeInterface.h"

#include <atomic>
#include <mpi.h>
#include <sys/resource.h>

int typeart_check_buffer(const typeart::MPICall& call);

struct CallCounter {
  std::atomic_size_t send        = {0};
  std::atomic_size_t recv        = {0};
  std::atomic_size_t send_recv   = {0};
  std::atomic_size_t unsupported = {0};
};

static CallCounter counter;

struct MPICounter {
  std::atomic_size_t null_count = {0};
  std::atomic_size_t null_buff  = {0};
  std::atomic_size_t type_error = {0};
  std::atomic_size_t error      = {0};
};

static MPICounter mcounter;

extern "C" {

void typeart_check_send(const char* name, const void* called_from, const void* sendbuf, int count, MPI_Datatype dtype) {
  ++counter.send;
  auto call = typeart::MPICall::create(name, called_from, sendbuf, 1, count, dtype);
  if (!call) {
    ++mcounter.error;
    return;
  }
  typeart_check_buffer(*call);
}

void typeart_check_recv(const char* name, const void* called_from, void* recvbuf, int count, MPI_Datatype dtype) {
  ++counter.recv;
  auto call = typeart::MPICall::create(name, called_from, recvbuf, 0, count, dtype);
  if (!call) {
    ++mcounter.error;
    return;
  }
  typeart_check_buffer(*call);
}

void typeart_check_send_and_recv(const char* name, const void* called_from, const void* sendbuf, int sendcount,
                                 MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype) {
  ++counter.send_recv;
  typeart_check_send(name, called_from, sendbuf, sendcount, sendtype);
  typeart_check_recv(name, called_from, recvbuf, recvcount, recvtype);
}

void typeart_unsupported_mpi_call(const char* name, const void* called_from) {
  ++counter.unsupported;
  fprintf(stderr, "[Error] The MPI function %s is currently not checked by TypeArt", name);
  // exit(0);
}

void typeart_exit() {
  // Called at MPI_Finalize time
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  struct rusage end;
  getrusage(RUSAGE_SELF, &end);
  fprintf(stderr, "R[%i][Info] CCounter { Send: %zu Recv: %zu Send_Recv: %zu Unsupported: %zu MAX RSS[KBytes]: %ld }\n",
          rank, counter.send.load(), counter.recv.load(), counter.send_recv.load(), counter.unsupported.load(),
          end.ru_maxrss);
  fprintf(stderr, "R[%i][Info] MCounter { Error: %zu Null_Buf: %zu Null_Count: %zu Type_Error: %zu }\n", rank,
          mcounter.error.load(), mcounter.null_buff.load(), mcounter.null_count.load(), mcounter.type_error.load());
}
}

int typeart_check_buffer(const typeart::MPICall& call) {
  PRINT_INFOV(call, "%s[%p] at %s:%s: %s: checking %s-buffer %p of type \"%s\" against MPI type \"%s\"\n",
              call.caller.function.c_str(), call.caller.addr, call.caller.file.c_str(), call.caller.line.c_str(),
              call.function_name.c_str(), call.is_send ? "send" : "recv", call.args.buffer.ptr,
              call.args.buffer.type_name.c_str(), call.args.type.name.c_str());

  const bool count_is_zero     = call.args.count <= 0;
  const bool buffer_is_nullptr = call.args.buffer.ptr == nullptr;

  if (count_is_zero) {
    ++mcounter.null_count;
  }

  if (buffer_is_nullptr) {
    ++mcounter.null_buff;
    PRINT_ERROR(call, "buffer is NULL\n");
  }

  if (count_is_zero || buffer_is_nullptr) {
    return -1;
  }

  if (call.check_type_and_count() != 0) {
    ++mcounter.type_error;
    return -1;
  }
  return 0;
}
