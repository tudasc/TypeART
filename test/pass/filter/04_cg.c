// clang-format off
// RUN: clang -S -emit-llvm %s -o - | opt -load %pluginpath/analysis/meminstfinderpass.so -load %pluginpath/%pluginname %pluginargs -typeart-alloca -alloca-array-only=false -call-filter -filter-impl=CG -cg-file=%p/04_cg.ipcg -S 2>&1 | FileCheck %s
// RUN: clang -S -emit-llvm %s -o - | opt -load %pluginpath/analysis/meminstfinderpass.so -load %pluginpath/%pluginname %pluginargs -typeart-alloca -alloca-array-only=false -call-filter -filter-impl=experimental::cg -cg-file=%p/04_cg.ipcg -S 2>&1 | FileCheck %s
// RUN: clang -S -emit-llvm %s -o - | opt -load %pluginpath/analysis/meminstfinderpass.so -load %pluginpath/%pluginname %pluginargs -typeart-alloca -alloca-array-only=false -call-filter -filter-impl=default -S 2>&1 | FileCheck %s --check-prefix=CHECK-default
// RUN: clang -S -emit-llvm %s -o - | opt -load %pluginpath/analysis/meminstfinderpass.so -load %pluginpath/%pluginname %pluginargs -typeart-alloca -alloca-array-only=false -call-filter -filter-impl=experimental::default -S 2>&1 | FileCheck %s --check-prefix=CHECK-default
// clang-format on

extern void bar(int* ptr);  // reaches MPI, see 04_cg.ipcg
extern void aar(int* ptr);  // does not reach MPI

void foo() {
  int a, b;
  bar(&a);
  aar(&b);
}
// CG:
// CHECK: > Stack Memory
// CHECK-NEXT: Alloca                 :  2.00
// CHECK-NEXT: Stack call filtered %  : 50.00

// Standard filter
// CHECK-default: > Stack Memory
// CHECK-default-NEXT: Alloca                 :  2.00
// CHECK-default-NEXT: Stack call filtered %  :  0.00