/*
 * InterceptorFunctions.h
 *
 *  Created on: Jul 16, 2018
 *      Author: ahueck
 */

#ifndef TEST_MPI_INTERCEPTOR_INTERCEPTORFUNCTIONS_H_
#define TEST_MPI_INTERCEPTOR_INTERCEPTORFUNCTIONS_H_

#include "TypeCheck.h"
#include "Util.h"
#include "runtime/RuntimeInterface.h"

#include <mpi.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/time.h>

int ta_check_buffer(const MPICallInfo* mpi_call, int const_adr);

static _Atomic size_t trace_id;

typedef struct CallCounter {
  _Atomic size_t send;
  _Atomic size_t recv;
  _Atomic size_t send_recv;
  _Atomic size_t unsupported;
} CCounter;

static CCounter counter = {0, 0, 0, 0};

typedef struct MPISemCounter {
  _Atomic size_t null_count;
  _Atomic size_t null_buff;
  _Atomic size_t error;
} MPICounter;

static MPICounter mcounter = {0, 0, 0};

void ta_check_send(const char* name, const void* called_from, const void* sendbuf, int count, MPI_Datatype dtype) {
  ++counter.send;
  MPICallInfo call_info;
  if (ta_create_call_info(trace_id++, name, called_from, sendbuf, 1, count, dtype, &call_info) != 0) {
    ++mcounter.error;
    return;
  }
  ta_check_buffer(&call_info, 1);
}

void ta_check_recv(const char* name, const void* called_from, void* recvbuf, int count, MPI_Datatype dtype) {
  ++counter.recv;
  MPICallInfo call_info;
  if (ta_create_call_info(trace_id++, name, called_from, recvbuf, 0, count, dtype, &call_info) != 0) {
    ++mcounter.error;
    return;
  }
  ta_check_buffer(&call_info, 0);
}

void ta_check_send_and_recv(const char* name, const void* called_from, const void* sendbuf, int sendcount,
                            MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype) {
  ++counter.send_recv;
  ta_check_send(name, called_from, sendbuf, sendcount, sendtype);
  ta_check_recv(name, called_from, recvbuf, recvcount, recvtype);
}

void ta_unsupported_mpi_call(const char* name, const void* called_from) {
  ++counter.unsupported;
  fprintf(stderr, "[Error] The MPI function %s is currently not checked by TypeArt", name);
  // exit(0);
}

int ta_check_buffer(const MPICallInfo* call, int const_adr) {
  PRINT_INFOV(call, "%s at %p in function %s: checking %s-buffer %p of type \"%s\" against MPI type \"%s\"\n",
              call->function_name, call->caller.addr, call->caller.name, call->is_send ? "send" : "recv",
              call->buffer.ptr, call->buffer.type_name, call->type.name);
  if (call->count <= 0) {
    ++mcounter.null_count;
    return 1;
  }
  if (call->buffer.ptr == NULL) {
    ++mcounter.null_buff;
    PRINT_ERRORV(call, "buffer %p is NULL\n", call->buffer.ptr);
    return -1;
  }
  return ta_check_type_and_count(call);
}

void ta_exit() {
  // Called at MPI_Finalize time
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  struct rusage end;
  getrusage(RUSAGE_SELF, &end);
  fprintf(stderr, "R[%i][Info] CCounter { Send: %zu Recv: %zu Send_Recv: %zu Unsupported: %zu MAX RSS[KBytes]: %ld }\n",
          rank, counter.send, counter.recv, counter.send_recv, counter.unsupported, end.ru_maxrss);
  fprintf(stderr, "R[%i][Info] MCounter { Error: %zu Null_Buf: %zu Null_Count: %zu }\n", rank, mcounter.error,
          mcounter.null_buff, mcounter.null_count);
}

#endif /* TEST_MPI_INTERCEPTOR_INTERCEPTORFUNCTIONS_H_ */
