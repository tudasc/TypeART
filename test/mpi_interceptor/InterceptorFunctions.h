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

void ta_check_const_void(const void* buf, MPI_Datatype dtype) {
  typeart_type_info type;
  size_t count = 0;
  typeart_status typeart_status_v = typeart_get_type(buf, &type, &count);
  if (typeart_status_v != TA_OK) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("R[%d][Error] Input buffer status: %d\n", rank, (int)typeart_status_v);
  }
}

void ta_check_void(const void* buf) {
}

#endif /* TEST_MPI_INTERCEPTOR_INTERCEPTORFUNCTIONS_H_ */
