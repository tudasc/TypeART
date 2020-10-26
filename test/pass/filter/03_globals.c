// clang-format off
// RUN: clang -S -emit-llvm %s -o - | opt -load %pluginpath/analysis/meminstfinderpass.so -load %pluginpath/%pluginname %pluginargs -typeart-alloca -alloca-array-only=false -call-filter -S 2>&1 | FileCheck %s
// RUN: clang -S -emit-llvm %s -o - | opt -load %pluginpath/analysis/meminstfinderpass.so -load %pluginpath/%pluginname %pluginargs -typeart-alloca -alloca-array-only=false -call-filter -filter-impl=experimental::default  -S 2>&1 | FileCheck %s
// clang-format on

int a;
double x[3];

extern void bar(int* v);
void foo() {
  bar(&a);
}

// CHECK: MemInstFinderPass
// Global                 :     2
// Global filter total    :     1
// Global call filtered % : 50.00
// Global filtered %      : 50.00