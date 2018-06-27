// RUN: clang -S -emit-llvm %s -o - | opt -load %pluginpath/analysis/MemInstFinderPass.so -load %pluginpath/%pluginname
// %pluginargs -S 2>&1 | FileCheck %s
void test() {
  int a[100];
}

// CHECK: Malloc{{[ ]*}}:{{[ ]*}}0
// CHECK: Free{{[ ]*}}:{{[ ]*}}0
// CHECK: Alloca{{[ ]*}}:{{[ ]*}}1
