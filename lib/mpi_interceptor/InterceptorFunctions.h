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

typedef struct {
  const void* ptr;
  int is_const;
  int count;
  int type_id;
  const char* type_name;
} MPIBufferInfo;

typedef struct {
  const char* function_name;
  const void* called_from;
  int rank;
  MPIBufferInfo buffer;
  int count;
  MPI_Datatype type;
} MPICallInfo;

int ta_create_call_info(const char* name, const void* called_from, const void* buffer, int is_const, int count,
                        MPI_Datatype type, MPICallInfo* call_info);
int ta_check_buffer(const MPICallInfo* mpi_call, int const_adr);
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
  MPICallInfo call_info;
  if (ta_create_call_info(name, called_from, sendbuf, 1, count, dtype, &call_info) != 0) {
    return;
  }
  ta_check_buffer(&call_info, 1);
}

void ta_check_recv(const char* name, const void* called_from, void* recvbuf, int count, MPI_Datatype dtype) {
  ++counter.recv;
  MPICallInfo call_info;
  if (ta_create_call_info(name, called_from, recvbuf, 0, count, dtype, &call_info) != 0) {
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

// Given a builtin MPI type, returns the corresponding TypeArt type.
// If the MPI type is a custom type, -1 is returned.
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
    return -1;
  }
  return TA_UNKNOWN_TYPE;
}

int ta_create_call_info(const char* function_name, const void* called_from, const void* buffer, int is_const, int count,
                        MPI_Datatype type, MPICallInfo* call_info) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int ta_type_id;
  size_t ta_count                 = 0;
  typeart_status typeart_status_v = typeart_get_type(buffer, &ta_type_id, &ta_count);
  if (typeart_status_v != TA_OK) {
    ++mcounter.error;
    const char* msg = ta_get_error_message(typeart_status_v);
    fprintf(stderr, "R[%d][Error][%d] %s: buffer %p at loc %p - %s\n", rank, is_const, function_name, buffer,
            called_from, msg);
    ta_print_loc(called_from);
    return -1;
  }
  const char* ta_type_name  = typeart_get_type_name(ta_type_id);
  MPIBufferInfo buffer_info = {buffer, is_const, ta_count, ta_type_id, ta_type_name};
  *call_info                = (MPICallInfo){function_name, called_from, rank, buffer_info, count, type};
  return 0;
}

void ta_log_buffer_too_small(const MPICallInfo* call, int actual_count, int required_count) {
  fprintf(stderr, "R[%d][Error][%d] %s: buffer %p too small. The buffer can only hold %d elements (%d required)\n",
          call->rank, call->buffer.is_const, call->function_name, call->buffer.ptr, actual_count, required_count);
  ta_print_loc(call->called_from);
}

int ta_check_builtin_type(const MPICallInfo* call, int mpi_type_id, int* mpi_count) {
  const char* mpi_type_name = typeart_get_type_name(mpi_type_id);
  fprintf(stderr, "R[%d][Info][%d] %s: buffer %p has type %s, MPI type is %s\n", call->rank, call->buffer.is_const,
          call->function_name, call->buffer.ptr, call->buffer.type_name, mpi_type_name);
  if (call->buffer.type_id != mpi_type_id && !(call->buffer.type_id == TA_PPC_FP128 && mpi_type_id == TA_FP128)) {
    fprintf(stderr, "R[%d][Error][%d] %s: buffer %p at loc %p has type %s while the MPI type is %s\n", call->rank,
            call->buffer.is_const, call->function_name, call->buffer.ptr, call->called_from, call->buffer.type_name,
            mpi_type_name);
    ta_print_loc(call->called_from);
    return -1;
  }
}

const char* ta_mpi_combiner_name(int combiner) {
  switch (combiner) {
    case MPI_COMBINER_NAMED:
      return "predefined type";
    case MPI_COMBINER_DUP:
      return "MPI_Type_dup";
    case MPI_COMBINER_CONTIGUOUS:
      return "MPI_Type_contiguous";
    case MPI_COMBINER_VECTOR:
      return "MPI_Type_vector";
    case MPI_COMBINER_HVECTOR:
      return "MPI_Type_hvector";
    case MPI_COMBINER_INDEXED:
      return "MPI_Type_indexed";
    case MPI_COMBINER_HINDEXED:
      return "MPI_Type_hindexed";
    case MPI_COMBINER_INDEXED_BLOCK:
      return "MPI_Type_create_indexed_block";
    case MPI_COMBINER_STRUCT:
      return "MPI_Type_struct";
    case MPI_COMBINER_SUBARRAY:
      return "MPI_Type_create_subarray";
    case MPI_COMBINER_DARRAY:
      return "MPI_Type_create_darray";
    case MPI_COMBINER_F90_REAL:
      return "MPI_Type_create_f90_real";
    case MPI_COMBINER_F90_COMPLEX:
      return "MPI_Type_create_f90_complex";
    case MPI_COMBINER_F90_INTEGER:
      return "MPI_Type_create_f90_integer";
    case MPI_COMBINER_RESIZED:
      return "MPI_Type_create_resized";
    default:
      return "invalid combiner id";
  }
  *mpi_count = 1;
}

int ta_check_type(const MPICallInfo* call, MPI_Datatype type, int* mpi_count) {
  int num_integers, num_addresses, num_datatypes, combiner;
  MPI_Type_get_envelope(type, &num_integers, &num_addresses, &num_datatypes, &combiner);
  int array_of_integers[num_integers];
  MPI_Aint array_of_addresses[num_addresses];
  MPI_Datatype array_of_datatypes[num_datatypes];
  if (combiner != MPI_COMBINER_NAMED) {
    MPI_Type_get_contents(type, num_integers, num_addresses, num_datatypes, array_of_integers, array_of_addresses,
                          array_of_datatypes);
  }
  switch (combiner) {
    case MPI_COMBINER_NAMED: {
      const int mpi_type_id = ta_mpi_type_to_type_id(type);
      if (mpi_type_id == -1) {
        fprintf(stderr, "R[%d][Error][%d] %s: couldn't convert builtin type\n", call->rank, call->buffer.is_const,
                call->function_name);
        ta_print_loc(call->called_from);
        return -1;
      }
      return ta_check_builtin_type(call, mpi_type_id, mpi_count);
    }
    case MPI_COMBINER_DUP:
      return ta_check_type(call, array_of_datatypes[0], mpi_count);
    case MPI_COMBINER_CONTIGUOUS: {
      int result = ta_check_type(call, array_of_datatypes[0], mpi_count);
      *mpi_count *= array_of_integers[0];
      return result;
    }
    case MPI_COMBINER_VECTOR: {
      int result = ta_check_type(call, array_of_datatypes[0], mpi_count);
      if (array_of_integers[2] < 0) {
        fprintf(stderr, "R[%d][Error][%d] %s: negative strides for MPI_Type_vector are currently not supported\n",
                call->rank, call->buffer.is_const, call->function_name);
        ta_print_loc(call->called_from);
        return -1;
      }
      // (count - 1) * stride + blocklength
      *mpi_count *= (array_of_integers[0] - 1) * array_of_integers[1] + array_of_integers[2];
      return result;
    }
    default:
      fprintf(stderr, "R[%d][Error][%d] %s: the MPI type combiner %s is currently not supported", call->rank,
              call->buffer.is_const, call->function_name, ta_mpi_combiner_name(combiner));
      ta_print_loc(call->called_from);
  }
}

int ta_check_buffer(const MPICallInfo* call, int const_adr) {
  if (call->count <= 0) {
    ++mcounter.null_count;
    return 1;
  }
  if (call->buffer.ptr == NULL) {
    ++mcounter.null_buff;
    fprintf(stderr, "R[%d][Error][%d] %s: buffer %p is NULL\n", call->rank, const_adr, call->function_name,
            call->buffer.ptr);
    ta_print_loc(call->called_from);
    return -1;
  }
  int mpi_type_count;
  if (ta_check_type(call, call->type, &mpi_type_count) == -1) {
    return -1;
  }
  if (call->count * mpi_type_count > call->buffer.count) {
    ta_log_buffer_too_small(call, (int)call->buffer.count, (int)call->count * mpi_type_count);
    return -1;
  }
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
  fprintf(stderr, "CCounter (%i) { Send: %zu Recv: %zu Send_Recv: %zu Unsupported: %zu MAX RSS[KBytes]: %ld }\n", rank,
          counter.send, counter.recv, counter.send_recv, counter.unsupported, end.ru_maxrss);
  fprintf(stderr, "MCounter (%i) { Error: %zu Null_Buf: %zu Null_Count: %zu }\n", rank, mcounter.error,
          mcounter.null_buff, mcounter.null_count);
}

#endif /* TEST_MPI_INTERCEPTOR_INTERCEPTORFUNCTIONS_H_ */
