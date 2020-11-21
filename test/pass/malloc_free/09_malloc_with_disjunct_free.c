// clang-format off
// RUN: %c-to-llvm %s | opt -load %pluginpath/analysis/meminstfinderpass.so -load %pluginpath/%pluginname %pluginargs -S 2>&1 | FileCheck %s
// clang-format on
#include <stdlib.h>

void foo(double* ptr);

void test() {
  double* p = (double*)malloc(42 * sizeof(double));
  foo(p);
}

void foo(double* ptr) {
  free(ptr);
  ptr = NULL;
}

// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK-NEXT: Free{{[ ]*}}:{{[ ]*}}1
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0
