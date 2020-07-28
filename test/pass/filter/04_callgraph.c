// clang-format off
// RUN: clang -S -emit-llvm %s -o - | opt -load %pluginpath/analysis/meminstfinderpass.so -load %pluginpath/%pluginname %pluginargs -typeart-alloca -alloca-array-only=false -call-filter -cg-file=04.ipcg -S 2>&1 | FileCheck %s
// XFAIL: *
// clang-format on

int a;
int b;

extern void bar(int* a);  // reaches MPI
extern void aar(int* a);  // does not reach MPI

void foo() {
  bar(&a);
  aar(&b);
}

// CHECK: Global                      :   2.0
// CHECK: Global total filtered       :   1.0
