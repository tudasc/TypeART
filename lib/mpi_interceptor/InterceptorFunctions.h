/*
 * InterceptorFunctions.h
 *
 *  Created on: Jul 16, 2018
 *      Author: ahueck
 */

#ifndef TEST_MPI_INTERCEPTOR_INTERCEPTORFUNCTIONS_H_
#define TEST_MPI_INTERCEPTOR_INTERCEPTORFUNCTIONS_H_

#include "runtime/RuntimeInterface.h"

#include <mpi.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/time.h>

int ta_mpi_type_to_type_id(MPI_Datatype mpi_type);
int ta_check_buffer(const char* mpi_name, const void* called_from, const void* buf, MPI_Datatype mpi_type,
                    int mpi_count, int const_adr);
void ta_print_loc(const void* call_adr);

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
  ta_check_buffer(name, called_from, sendbuf, dtype, count, 1);
}

void ta_check_recv(const char* name, const void* called_from, void* recvbuf, int count, MPI_Datatype dtype) {
  ++counter.recv;
  ta_check_buffer(name, called_from, recvbuf, dtype, count, 0);
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

int ta_mpi_map_int_type(size_t int_sizeof) {
  if (int_sizeof == 1) {
    return TA_INT8;
  } else if (int_sizeof == 2) {
    return TA_INT16;
  } else if (int_sizeof == 4) {
    return TA_INT32;
  } else if (int_sizeof == 8) {
    return TA_INT64;
  } else {
    fprintf(stderr, "[Error] Unsupperted integer width %lu!\n", int_sizeof);
    return TA_UNKNOWN_TYPE;
  }
}

// Given an MPI type, returns the corresponding TypeArt type.
// Note: this function cannot distinguish between TA_FP128 und TA_PPC_TP128,
// therefore TA_FP128 is always returned in case of an 16 byte floating point
// MPI type. This should be considered by the caller for performing typechecks.
int ta_mpi_type_to_type_id(MPI_Datatype mpi_type) {
  if (mpi_type == MPI_CHAR) {
    fprintf(stderr, "[Error] MPI_CHAR is currently unsupported!\n");
  } else if (mpi_type == MPI_UNSIGNED_CHAR) {
    fprintf(stderr, "[Error] MPI_UNSIGNED_CHAR is currently unsupported!\n");
  } else if (mpi_type == MPI_SIGNED_CHAR) {
    fprintf(stderr, "[Error] MPI_SIGNED_CHAR is currently unsupported!\n");
  } else if (mpi_type == MPI_SHORT) {
    return ta_mpi_map_int_type(sizeof(short));
  } else if (mpi_type == MPI_UNSIGNED_SHORT) {
    fprintf(stderr, "[Error] Unsigned integers are currently not supported!\n");
  } else if (mpi_type == MPI_INT) {
    return ta_mpi_map_int_type(sizeof(int));
  } else if (mpi_type == MPI_UNSIGNED) {
    fprintf(stderr, "[Error] Unsigned integers are currently not supported!\n");
  } else if (mpi_type == MPI_LONG) {
    return ta_mpi_map_int_type(sizeof(long int));
  } else if (mpi_type == MPI_UNSIGNED_LONG) {
    fprintf(stderr, "[Error] Unsigned integers are currently not supported!\n");
  } else if (mpi_type == MPI_LONG_LONG_INT) {
    return ta_mpi_map_int_type(sizeof(long long int));
  } else if (mpi_type == MPI_FLOAT) {
    return TA_FLOAT;
  } else if (mpi_type == MPI_DOUBLE) {
    return TA_DOUBLE;
  } else if (mpi_type == MPI_LONG_DOUBLE) {
    if (sizeof(long double) == sizeof(double)) {
      return TA_DOUBLE;
    } else if (sizeof(long double) == 10) {
      return TA_X86_FP80;
    } else if (sizeof(long double) == 16) {
      return TA_FP128;
    } else {
      fprintf(stderr, "[Error] long double has unexpected size %zu!\n", sizeof(long double));
    }
  } else {
    fprintf(stderr, "[Error] Unsupported MPI datatype found!\n");
  }
  return TA_UNKNOWN_TYPE;
}

int ta_check_buffer(const char* mpi_name, const void* called_from, const void* buf, MPI_Datatype mpi_type,
                    int mpi_count, int const_adr) {
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
  const char* type_name     = typeart_get_type_name(typeId);
  const int mpi_type_id     = ta_mpi_type_to_type_id(mpi_type);
  const char* mpi_type_name = typeart_get_type_name(mpi_type_id);
  printf("R[%d][Info][%d] %s: buffer %p has type %s, MPI type is %s\n", rank, const_adr, mpi_name, buf, type_name,
         mpi_type_name);
  if (typeId != mpi_type_id && !(typeId == TA_PPC_FP128 && mpi_type_id == TA_FP128)) {
    printf("R[%d][Error][%d] %s: buffer %p at loc %p has type %s while the MPI type is %s\n", rank, const_adr, mpi_name,
           buf, called_from, type_name, mpi_type_name);
    ta_print_loc(called_from);
    return -1;
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
  printf("CCounter (%i) { Send: %zu Recv: %zu Send_Recv: %zu Unsupported: %zu MAX RSS[KBytes]: %ld }\n", rank,
         counter.send, counter.recv, counter.send_recv, counter.unsupported, end.ru_maxrss);
  printf("MCounter (%i) { Error: %zu Null_Buf: %zu Null_Count: %zu }\n", rank, mcounter.error, mcounter.null_buff,
         mcounter.null_count);
}

#endif /* TEST_MPI_INTERCEPTOR_INTERCEPTORFUNCTIONS_H_ */
