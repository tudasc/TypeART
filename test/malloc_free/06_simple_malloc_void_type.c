// RUN: clang -S -emit-llvm %s -o - | opt -load %pluginpath/%pluginname %pluginargs -S 2>&1 | FileCheck %s
#include <stdlib.h>
void test() {
  void* p = malloc(42 * sizeof(int));  // LLVM-IR: lacks a bitcast
}
// CHECK: Malloc{{[ ]*}}:{{[ ]*}}1
// Also required (TBD): Alloca{{[ ]*}}:{{[ ]*}}0
