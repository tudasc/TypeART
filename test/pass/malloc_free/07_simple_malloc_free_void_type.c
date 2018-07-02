// RUN: clang -S -emit-llvm %s -o - | opt -load %pluginpath/analysis/meminstfinderpass.so -load %pluginpath/%pluginname
// %pluginargs -S 2>&1 | FileCheck %s
#include <stdlib.h>
void test() {
  void* p = malloc(42 * sizeof(int));  // LLVM-IR: lacks a bitcast
  free(p);
}
// CHECK: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK: Alloca{{[ ]*}}:{{[ ]*}}0
