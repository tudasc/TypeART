// RUN: clang -S -emit-llvm %s -o - | opt -load %pluginpath/analysis/MemInstFinderPass.so -load %pluginpath/%pluginname
// %pluginargs -S 2>&1 | FileCheck %s
#include <stdlib.h>
void test() {
  double* p = (double*)malloc(42 * sizeof(double));
  free(p);
}

// CHECK: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK: Free{{[ ]*}}:{{[ ]*}}1
// CHECK: Alloca{{[ ]*}}:{{[ ]*}}0
