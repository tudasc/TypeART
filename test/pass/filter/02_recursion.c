// Template for recursion.ll.in
// clang-format off
// RUN: clang -S -emit-llvm %s -o - | opt -load %pluginpath/analysis/meminstfinderpass.so -load %pluginpath/%pluginname %pluginargs -typeart-alloca -alloca-array-only=false -call-filter -filter-impl=deprecated::default -S 2>&1 | FileCheck %s
// RUN: clang -S -emit-llvm %s -o - | opt -load %pluginpath/analysis/meminstfinderpass.so -load %pluginpath/%pluginname %pluginargs -typeart-alloca -alloca-array-only=false -call-filter -S 2>&1 | FileCheck %s
// clang-format on
void bar(int* x) {
}

void foo(int x) {
  bar(&x);
  if (x > 1) {
    foo(x);
  }
}

// CHECK: MemInstFinderPass
// CHECK: Stack call filtered %{{[ :]+}}100.0
