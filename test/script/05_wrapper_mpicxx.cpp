// RUN: echo --- > types.yaml
// RUN: %wrapper-mpicxx -g %s -o %s.exe
// RUN: %mpi-exec -np 1 %s.exe 2>&1 | %filecheck %s

// RUN: %wrapper-mpicxx -g -c %s -o %s.o
// RUN: %wrapper-mpicxx %s.o -o %s.exe
// RUN: %mpi-exec -np 1 %s.exe 2>&1 | %filecheck %s

// REQUIRES: mpicxx
// UNSUPPORTED: sanitizer

#include "../../lib/runtime/CallbackInterface.h"

#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  __typeart_alloc((const void*)2, 7, 1);  // OK
  MPI_Finalize();
  return 0;
}

// CHECK: [Trace] Alloc 0x2 7 float128 16 1
