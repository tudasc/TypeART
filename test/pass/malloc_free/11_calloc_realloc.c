// clang-format off
// RUN: clang -S -emit-llvm %s -o - | opt -load %pluginpath/analysis/meminstfinderpass.so -load %pluginpath/%pluginname %pluginargs -S 2>&1 | FileCheck %s
// clang-format on
#include <stdlib.h>

int main() {
  double* pd = calloc(10, sizeof(double));

  pd = realloc(pd, 20 * sizeof(double));

  return 0;
}

// CHECK: Malloc{{[ ]*}}:{{[ ]*}}2
// CHECK: Alloca{{[ ]*}}:{{[ ]*}}0
