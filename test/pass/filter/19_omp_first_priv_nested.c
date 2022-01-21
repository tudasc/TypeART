// clang-format off
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %apply-typeart -typeart-stack -typeart-call-filter -S 2>&1 | %filecheck %s
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %opt -O2 -S | %apply-typeart -typeart-stack -typeart-call-filter -S 2>&1 | %filecheck %s

// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %apply-typeart -typeart-stack -typeart-call-filter -S | %filecheck %s --check-prefix=check-inst
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %opt -O2 -S | %apply-typeart -typeart-stack -typeart-call-filter -S | %filecheck %s --check-prefix=check-inst
// REQUIRES: openmp
// clang-format on

#include "omp.h"

extern void MPI_Send(void*, int);

void func(int* x, int* e) {
  // firstprivate > every thread has a private copy of addr(!) x
  // check-inst: define {{.*}} @func
  // check-inst-NOT: call void @__typeart_alloc_stack
#pragma omp parallel for firstprivate(x), shared(e)
  for (int i = 0; i < 10; ++i) {
    // Analysis should not filter x, but e...
    MPI_Send((void*)x, *e);
  }
}

void foo() {
  // check-inst: define {{.*}} @foo
  // check-inst: call void @__typeart_alloc_stack(i8* %0, i32 2, i64 1)
  int x = 1;
  int y = 2;
#pragma omp parallel
  { func(&x, &y); }
}

void func_other(int* x, int* e) {
  // firstprivate > every thread has a private copy of addr(!) x
  // check-inst: define {{.*}} @func_other
  // check-inst-NOT: call void @__typeart_alloc_stack
#pragma omp parallel for firstprivate(x), shared(e)
  for (int i = 0; i < 10; ++i) {
    // Analysis should not filter x, but e...
    MPI_Send(x, *e);
  }
  MPI_Send(x, *e);
}

void bar(int x_other) {
  // check-inst: define {{.*}} @bar
  // check-inst: call void @__typeart_alloc_stack(i8* %0, i32 2, i64 1)
  int x = x_other;
  int y = 2;
#pragma omp parallel
  { func_other(&x, &y); }
}

// CHECK: TypeArtPass [Heap & Stack]
// CHECK-NEXT: Malloc :   0
// CHECK-NEXT: Free   :   0
// CHECK-NEXT: Alloca :   2
// CHECK-NEXT: Global :   0
