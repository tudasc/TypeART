// RUN: clang -S -emit-llvm %s -o - | opt -load %pluginpath/%pluginname %pluginargs -S 2>&1 | FileCheck %s
#include <stdlib.h>
void test() {
  void* p = malloc(42 * sizeof(int));
  int* pi = (int*)p;
  short* ps = (short*)p;
}

// CHECK: [Warning]
// CHECK: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK: Alloca{{[ ]*}}:{{[ ]*}}0
