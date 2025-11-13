// clang-format off
// RUN: export TYPEART_TYPE_SERIALIZATION=inline
// RUN: %wrapper-cc -O1 %s -o %s.exe
// RUN: %s.exe 2>&1 | %filecheck %s

// REQUIRES: llvm-18 || llvm-19
// clang-format on

#include <stdlib.h>
int main(void) {
  int* p = (int*)malloc(42 * sizeof(int));
  free(p);
  return 0;
}

// CHECK: Allocation type detail (heap, stack, global)
// CHECK-NEXT: 13 :   1 ,    0 ,    0 , int