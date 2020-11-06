// clang-format off
// RUN: clang -S -emit-llvm %s -o - | opt -load %pluginpath/analysis/meminstfinderpass.so -load %pluginpath/%pluginname %pluginargs -typeart-alloca -alloca-array-only=false -call-filter -S 2>&1 | FileCheck %s
// RUN: clang -S -emit-llvm %s -o - | opt -load %pluginpath/analysis/meminstfinderpass.so -load %pluginpath/%pluginname %pluginargs -typeart-alloca -alloca-array-only=false -call-filter  -filter-impl=experimental::default -S 2>&1 | FileCheck %s
// clang-format on

extern int d;

void empty() {
  int a = 1;
  int b = 2;
  int c = 3;

  if (d > c) {
    b = a * c;
  } else {
    b = c * c;
  }
}

// Standard filter
// CHECK: > Stack Memory
// CHECK-NEXT: Alloca                 :  3.00
// CHECK-NEXT: Stack call filtered %  :  100.00