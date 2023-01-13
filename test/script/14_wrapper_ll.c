// RUN: %wrapper-cc -S -emit-llvm -O1 %s -o %s.ll
// RUN: cat %s.ll 2>&1 | %filecheck %s

// RUN: %wrapper-cc -emit-llvm -O1 %s -o %s.bc
// RUN: cat %s.bc 2>&1 | %opt -S | %filecheck %s

// RUN: TYPEART_WRAPPER=OFF %wrapper-cc -S -emit-llvm -O1 %s -o %s-van.ll
// RUN: cat %s-van.ll 2>&1 | %filecheck %s --check-prefixes vanilla-CHECK

// RUN: TYPEART_WRAPPER=OFF %wrapper-cc -c -emit-llvm -O1 %s -o %s-van.bc
// RUN: cat %s-van.bc 2>&1 | %opt -S | %filecheck %s --check-prefixes vanilla-CHECK

#include <stdlib.h>

int main(int argc, char** argv) {
  int* p = malloc(argc * sizeof(int));
  return 0;
}

// CHECK: __typeart_alloc
// vanilla-CHECK-NOT: __typeart_alloc
