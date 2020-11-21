// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -S 2>&1 | FileCheck %s
// clang-format on
#include <stdlib.h>
void test() {
  int* p     = (int*)malloc(42 * sizeof(int));
  double* pd = (double*)malloc(42 * sizeof(double));
}

// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}2
// CHECK-NEXT: Free
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0
