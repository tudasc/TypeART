// clang-format off
// RUN: clang -S -emit-llvm %s -o - | opt -load %pluginpath/analysis/meminstfinderpass.so -load %pluginpath/%pluginname %pluginargs -typeart-alloca -alloca-array-only=false -call-filter -S 2>&1 | FileCheck %s
// clang-format on

int a;
double x[3];

extern void bar(int* v);
void foo() {
  bar(&a);
}

//CHECK: Global                      :   2.0
//CHECK: Global Filtered             :   1.0
//CHECK: % global call filtered      :  50.0
