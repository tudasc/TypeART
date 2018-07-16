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

int ta_check_exists(const char* mpi_name, const void* called_from, const void* buf);
void ta_print_loc(const void* call_adr);

void ta_check_const_void(const char* name, const void* called_from, const void* buf, MPI_Datatype dtype) {
  ta_check_exists(name, called_from, buf);
}

void ta_check_void(const char* name, const void* called_from, const void* buf) {
  ta_check_exists(name, called_from, buf);
}

int ta_check_exists(const char* mpi_name, const void* called_from, const void* buf) {
  if (buf == NULL) {
    return -1;
  }
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  typeart_type_info type;
  size_t count = 0;
  typeart_status typeart_status_v = typeart_get_type(buf, &type, &count);
  if (typeart_status_v != TA_OK && rank == 0) {
    printf("R[%d][Error] Call '%s' buffer %p at loc %p status: %d\n", rank, mpi_name, buf, called_from,
           (int)typeart_status_v);
    ta_print_loc(called_from);
    return 0;
  }
  return 1;
}

void ta_print_loc(const void* call_adr) {
  const char* exe = getenv("TA_EXE_TARGET");
  if (exe == NULL || exe[0] == '\0') {
    return;
  }

  char cmd_buf[512];
  snprintf(cmd_buf, sizeof(cmd_buf), "addr2line -e %s -f %p", exe, call_adr);

  FILE* fp = popen(cmd_buf, "r");
  if (fp) {
    char read_buf[512];
    while (fgets(read_buf, sizeof(read_buf), fp)) {
      printf("    %s", read_buf);
    }
  }
  fclose(fp);
}

#endif /* TEST_MPI_INTERCEPTOR_INTERCEPTORFUNCTIONS_H_ */
