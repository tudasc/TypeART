// RUN: echo --- > types.yaml
// RUN: %wrapper-cxx -O1 %s -o %s.exe
// RUN: %s.exe 2>&1 | %filecheck %s

#include "../../lib/runtime/CallbackInterface.h"

int main(int argc, char** argv) {
  __typeart_alloc((const void*)2, 7, 1);  // OK
  return 0;
}

// CHECK: [Trace] Alloc 0x2 7 float128 16 1
