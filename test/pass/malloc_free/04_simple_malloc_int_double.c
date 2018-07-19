// clang-format off
// RUN: clang -S -emit-llvm %s -o - | opt -load %pluginpath/analysis/meminstfinderpass.so -load %pluginpath/%pluginname %pluginargs -S 2>&1 | FileCheck %s
// clang-format on
#include <stdlib.h>
void test() {
  int* p = (int*)malloc(42 * sizeof(int));
  double* pd = (double*)malloc(42 * sizeof(double));
}

// CHECK: Malloc{{[ ]*}}:{{[ ]*}}2
// CHECK: Alloca{{[ ]*}}:{{[ ]*}}0
