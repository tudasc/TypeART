// RUN: echo --- > typeart-types.yaml
// RUN: %wrapper-mpicc -O1 %s -o %s.exe
// RUN: %mpi-exec -np 1 %s.exe 2>&1 | %filecheck %s

// RUN: %wrapper-mpicc -O1 -c %s -o %s.o
// RUN: %wrapper-mpicc %s.o -o %s.exe
// RUN: %mpi-exec -np 1 %s.exe 2>&1 | %filecheck %s

// REQUIRES: mpicc
// UNSUPPORTED: sanitizer

#include "../../lib/runtime/CallbackInterface.h"
#include "../../lib/typelib/TypeInterface.h"

#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  __typeart_alloc((const void*)2, TYPEART_FLOAT_128, 1);  // OK
  MPI_Finalize();
  return 0;
}

// CHECK: [Trace] Alloc 0x2 25 long double 16 1
