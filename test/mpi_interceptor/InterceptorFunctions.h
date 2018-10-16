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

int ta_check_exists(const char* mpi_name, const void* called_from, const void* buf, int const_adr);
void ta_print_loc(const void* call_adr);

void ta_check_const_void(const char* name, const void* called_from, const void* buf, MPI_Datatype dtype) {
  ta_check_exists(name, called_from, buf, 1);
}

void ta_check_void(const char* name, const void* called_from, const void* buf) {
  ta_check_exists(name, called_from, buf, 0);
}

int ta_check_exists(const char* mpi_name, const void* called_from, const void* buf, int const_adr) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (buf == NULL) {
    printf("R[%d][Error][%d] Call '%s' buffer NULL\n", rank, const_adr, mpi_name);
    ta_print_loc(called_from);
    return -1;
  }
  typeart_type_info type;
  size_t count = 0;
  typeart_status typeart_status_v = typeart_get_type(buf, &type, &count);
  if (typeart_status_v != TA_OK) {
    printf("R[%d][Error][%d] Call '%s' buffer %p at loc %p status: %d\n", rank, const_adr, mpi_name, buf, called_from,
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

#endif /* TEST_MPI_INTERCEPTOR_INTERCEPTORFUNCTIONS_H_ */
