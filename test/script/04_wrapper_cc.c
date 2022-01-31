// RUN: echo --- > types.yaml
// RUN: %wrapper-cc -O1 %s -o %s.exe
// RUN: %s.exe 2>&1 | %filecheck %s

// RUN: %wrapper-cc -O1 -c %s -o %s.o
// RUN: %wrapper-cc %s.o -o %s.exe
// RUN: %mpi-exec -np 1 %s.exe 2>&1 | %filecheck %s

#include "../../lib/runtime/CallbackInterface.h"

int main(int argc, char** argv) {
  __typeart_alloc((const void*)2, 7, 1);  // OK
  return 0;
}

// CHECK: [Trace] Alloc 0x2 7 float128 16 1
