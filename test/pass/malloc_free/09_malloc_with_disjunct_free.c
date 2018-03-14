// RUN: clang -S -emit-llvm %s -o - | opt -load %pluginpath/%pluginname %pluginargs -S 2>&1 | FileCheck %s
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

// CHECK: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK: Free{{[ ]*}}:{{[ ]*}}1
// CHECK: Alloca{{[ ]*}}:{{[ ]*}}0