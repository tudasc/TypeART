// clang-format off
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %apply-typeart -typeart-stack -typeart-call-filter -typeart-filter-pointer-alloca=false -S 2>&1 | %filecheck %s
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %opt -O2 -S | %apply-typeart -typeart-stack -typeart-call-filter -typeart-filter-pointer-alloca=false -S 2>&1 | %filecheck %s --check-prefix=check-opt

// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %apply-typeart -typeart-stack -typeart-call-filter -typeart-filter-pointer-alloca=false -S | %filecheck %s --check-prefix=check-inst
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %opt -O2 -S | %apply-typeart -typeart-stack -typeart-call-filter -typeart-filter-pointer-alloca=false -S | %filecheck %s --check-prefix=check-opt-inst
// REQUIRES: openmp
// clang-format on

#include "omp.h"

// NOTE: with opt, the compiler passes the address until the MPI_Send, hence
// only the initial allocation is tracked.

extern void MPI_Send(void*, int);

void func(int* x, int* e) {
  // check-inst: define {{.*}} @func
  // check-inst-NOT: call void @__typeart_alloc_stack

  // check-opt-inst: define {{.*}} @func
  // check-opt-inst-NOT: call void @__typeart_alloc_stack

  // check-inst: define {{.*}} @.omp_outlined
  // check-inst: call void @__typeart_alloc_stack_omp(i8* %{{[0-9]}}, i32 10, i64 1)

  // check-opt-inst: define {{.*}} @.omp_outlined
  // check-opt-inst-NOT: call void @__typeart_alloc_stack_omp
#pragma omp parallel for firstprivate(x), lastprivate(x), shared(e)
  for (int i = 0; i < 10; ++i) {
    // Analysis should not filter x, but e...
    MPI_Send((void*)x, *e);
  }
}

void foo() {
  // check-inst: define {{.*}} @foo
  // check-inst: call void @__typeart_alloc_stack(i8* %0, i32 2, i64 1)

  // check-opt-inst: define {{.*}} @foo
  // check-opt-inst: call void @__typeart_alloc_stack(i8* %0, i32 2, i64 1)
  int x = 1;
  int y = 2;
#pragma omp parallel
  { func(&x, &y); }
}

void func_other(int* x, int* e) {
  // check-inst: define {{.*}} @func_other
  // check-inst-NOT: call void @__typeart_alloc_stack

  // check-opt-inst: define {{.*}} @func_other
  // check-opt-inst-NOT: call void @__typeart_alloc_stack

  // check-inst: define {{.*}} @.omp_outlined
  // check-inst: call void @__typeart_alloc_stack_omp(i8* %{{[0-9]}}, i32 10, i64 1)

  // check-opt-inst: define {{.*}} @.omp_outlined
  // check-opt-inst-NOT: call void @__typeart_alloc_stack_omp
#pragma omp parallel for firstprivate(x), lastprivate(x), shared(e)
  for (int i = 0; i < 10; ++i) {
    // Analysis should not filter x, but e...
    MPI_Send(x, *e);
  }
  MPI_Send(x, *e);
}

void bar(int x_other) {
  // check-inst: define {{.*}} @bar
  // check-inst: call void @__typeart_alloc_stack(i8* %{{[0-9]}}, i32 2, i64 1)

  // check-opt-inst: define {{.*}} @bar
  // check-opt-inst: call void @__typeart_alloc_stack(i8* %{{[0-9]}}, i32 2, i64 1)
  int x = x_other;
  int y = 2;
#pragma omp parallel
  { func_other(&x, &y); }
}

// CHECK: TypeArtPass [Heap & Stack]
// CHECK-NEXT: Malloc :   0
// CHECK-NEXT: Free   :   0
// CHECK-NEXT: Alloca :   4
// CHECK-NEXT: Global :   0

// check-opt: TypeArtPass [Heap & Stack]
// check-opt: Malloc :   0
// check-opt: Free   :   0
// check-opt: Alloca :   2
// check-opt: Global :   0