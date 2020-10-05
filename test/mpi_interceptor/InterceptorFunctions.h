/*
 * InterceptorFunctions.h
 *
 *  Created on: Jul 16, 2018
 *      Author: ahueck
 */

#ifndef TEST_MPI_INTERCEPTOR_INTERCEPTORFUNCTIONS_H_
#define TEST_MPI_INTERCEPTOR_INTERCEPTORFUNCTIONS_H_

#include "RuntimeInterface.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/time.h>

int ta_check_buffer(const char* mpi_name, const void* called_from, const void* buf, int mpi_count, int const_adr);
void ta_print_loc(const void* call_adr);

typedef struct CallCounter {
  size_t send;
  size_t recv;
  size_t send_recv;
  size_t unsupported;
} CCounter;

static CCounter counter = {0, 0, 0, 0};

typedef struct MPISemCounter {
  size_t null_count;
  size_t null_buff;
  size_t error;
} MPICounter;

static MPICounter mcounter = {0, 0, 0};

void ta_check_send(const char* name, const void* called_from, const void* sendbuf, int count, MPI_Datatype dtype) {
  ++counter.send;
  ta_check_buffer(name, called_from, sendbuf, count, 1);
}

void ta_check_recv(const char* name, const void* called_from, void* recvbuf, int count, MPI_Datatype dtype) {
  ++counter.recv;
  ta_check_buffer(name, called_from, recvbuf, count, 0);
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
  ta_print_loc(called_from);
  // exit(0);
}

const char* ta_get_error_message(typeart_status status) {
  switch (status) {
    case TA_OK:
      return "No errors";
    case TA_UNKNOWN_ADDRESS:
      return "Buffer not registered";
    case TA_BAD_ALIGNMENT:
      return "Buffer access is not aligned correctly";
    case TA_BAD_OFFSET:
      return "Error in offset computation";
    case TA_WRONG_KIND:
      return "Wrong type kind";
    case TA_INVALID_ID:
      return "Invalid type ID";
    default:
      return "Invalid error code";
  }
}

int ta_check_buffer(const char* mpi_name, const void* called_from, const void* buf, int mpi_count, int const_adr) {
  if (mpi_count <= 0) {
    ++mcounter.null_count;
    return 1;
  }
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (buf == NULL) {
    ++mcounter.null_buff;
    printf("R[%d][Error][%d] %s: buffer %p is NULL\n", rank, const_adr, mpi_name, buf);
    ta_print_loc(called_from);
    return -1;
  }
  int typeId;
  size_t count                    = 0;
  typeart_status typeart_status_v = typeart_get_type(buf, &typeId, &count);
  if (typeart_status_v != TA_OK) {
    ++mcounter.error;
    const char* msg = ta_get_error_message(typeart_status_v);
    printf("R[%d][Error][%d] %s: buffer %p at loc %p - %s\n", rank, const_adr, mpi_name, buf, called_from, msg);
    ta_print_loc(called_from);
    return 0;
  }
  // if (mpi_count > count) {
  // TODO: Count check not really sensible without taking the MPI type into account
  //  printf("R[%d][Error][%d] Call '%s' buffer %p too small\n", rank, const_adr, mpi_name, buf);
  //  printf("The buffer can only hold %d elements (%d required)\n", (int) count, (int) mpi_count);
  //  ta_print_loc(called_from);
  //  return 0;
  //}
  return 1;
}

void ta_print_loc(const void* call_adr) {
  const char* exe = getenv("TA_EXE_TARGET");
  if (exe == NULL || exe[0] == '\0') {
    return;
  }

  char cmd_buf[512] = {0};
  snprintf(cmd_buf, sizeof(cmd_buf), "addr2line -e %s -f %p", exe, call_adr);

  FILE* fp = popen(cmd_buf, "r");
  if (fp) {
    char read_buf[512] = {0};
    while (fgets(read_buf, sizeof(read_buf), fp)) {
      printf("    %s", read_buf);
    }
  }
  pclose(fp);
}

void ta_exit() {
  // Called at MPI_Finalize time
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  struct rusage end;
  getrusage(RUSAGE_SELF, &end);
  printf("CCounter (%i) { Send: %i Recv: %i Send_Recv: %i Unsupported: %i MAX RSS[KBytes]: %ld }\n", rank, counter.send,
         counter.recv, counter.send_recv, counter.unsupported, end.ru_maxrss);
  printf("MCounter (%i) { Error: %i Null_Buf: %i Null_Count: %i }\n", rank, mcounter.error, mcounter.null_buff,
         mcounter.null_count);
}

#endif /* TEST_MPI_INTERCEPTOR_INTERCEPTORFUNCTIONS_H_ */
