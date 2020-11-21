// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -S 2>&1 | FileCheck %s
// clang-format on
#include <stdlib.h>

int main() {
  double* pd = calloc(10, sizeof(double));

  pd = realloc(pd, 20 * sizeof(double));

  return 0;
}

// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}2
// CHECK-NEXT: Free
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0
