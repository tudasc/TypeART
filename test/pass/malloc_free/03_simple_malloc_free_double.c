// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -S 2>&1 | FileCheck %s
// clang-format on
#include <stdlib.h>
void test() {
  double* p = (double*)malloc(42 * sizeof(double));
  free(p);
}

// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK-NEXT: Free{{[ ]*}}:{{[ ]*}}1
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0
