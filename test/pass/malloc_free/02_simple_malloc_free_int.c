// RUN: clang -S -emit-llvm %s -o - | opt -load %pluginpath/%pluginname %pluginargs -S 2>&1 | FileCheck %s
#include <stdlib.h>
void test() {
  int* p = (int*)malloc(42 * sizeof(int));
  free(p);
}

// CHECK: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK: Free{{[ ]*}}:{{[ ]*}}1
// CHECK: Alloca{{[ ]*}}:{{[ ]*}}0
