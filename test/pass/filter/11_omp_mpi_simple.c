// clang-format off
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %apply-typeart -typeart-alloca -call-filter -S 2>&1 | FileCheck %s
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | opt -O2 -S | %apply-typeart -typeart-alloca -call-filter -S 2>&1 | FileCheck %s
// REQUIRES: openmp
// XFAIL: *
// clang-format on

#include "omp.h"

void MPI_test(void*);
void foo(int* x) {
#pragma omp parallel  // transformed to @__kmpc_fork_call
  { MPI_test(x); }
}

void bar() {
  int x;
  foo(&x);
}
// CHECK: TypeArtPass [Heap & Stack]
// CHECK-NEXT: Malloc :   0
// CHECK-NEXT: Free   :   0
// CHECK-NEXT: Alloca :   1
// CHECK-NEXT: Global :   0