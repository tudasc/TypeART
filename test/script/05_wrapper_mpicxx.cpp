// RUN: echo --- > types.yaml
// RUN: %wrapper-mpicxx -O1 %s -o %s.exe
// RUN: %mpi-exec -np 1 %s.exe 2>&1 | FileCheck %s

// REQUIRES: mpicxx

#include <mpi.h>
#include "../../lib/runtime/CallbackInterface.h"


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    __typeart_alloc((const void *) 2, 7, 1);  // OK
    MPI_Finalize();
    return 0;
}

// CHECK: [Trace] Alloc 0x2 7 float128 16 1
