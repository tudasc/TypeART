// clang-format off
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %apply-typeart -typeart-stack -typeart-call-filter -typeart-filter-pointer-alloca=false -S 2>&1 | %filecheck %s --check-prefix CHECK-alloca-pointer
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %opt -O2 -S | %apply-typeart -typeart-stack -typeart-call-filter -typeart-filter-pointer-alloca=false -S 2>&1 | %filecheck %s

// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %apply-typeart -typeart-stack -typeart-call-filter -typeart-filter-pointer-alloca=true -S 2>&1 | %filecheck %s --check-prefix CHECK-alloca-pointer
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %opt -O2 -S | %apply-typeart -typeart-stack -typeart-call-filter -typeart-filter-pointer-alloca=true -S 2>&1 | %filecheck %s --check-prefix CHECK-alloca-pointer
// clang-format on

// REQUIRES: openmp

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

// FIXME: the opt pass tracks 2 allocs in bar (alloca x and alloca x.addr (which is passed to the outlined region)):
// CHECK: TypeArtPass [Heap & Stack]
// CHECK-NEXT: Malloc :   0
// CHECK-NEXT: Free   :   0
// CHECK-NOT: Alloca :   1
// CHECK: Global :   0

// CHECK-alloca-pointer: TypeArtPass [Heap & Stack]
// CHECK-alloca-pointer-NEXT: Malloc :   0
// CHECK-alloca-pointer-NEXT: Free   :   0
// CHECK-alloca-pointer-NEXT: Alloca :   1
// CHECK-alloca-pointer-NEXT: Global :   0
