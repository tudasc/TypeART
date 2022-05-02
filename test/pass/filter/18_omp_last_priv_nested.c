// clang-format off
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %apply-typeart -typeart-stack -typeart-call-filter -typeart-filter-pointer-alloca=false -S 2>&1 | %filecheck %s

// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %apply-typeart -typeart-stack -typeart-call-filter -typeart-filter-pointer-alloca=false -S | %filecheck %s --check-prefix=check-inst
// REQUIRES: openmp
// clang-format on

// NOTE: This test has limited applicability in this scenario:
// lastprivate(x) copies address used in the MPI_send, and subsequently copies the result back to "x*".
// The data flow tracker detects only the usage of the copy in the context of MPI (see "foo() and func(...)")
// NOTE 2: with optimization the parameter "x" of MPI_Send mock call in the parallel loop gets "undef"

#include "omp.h"

extern void MPI_Send(void*, int);

void func(int* x, int* e) {
  // lastprivate - addr(!) value of x is copied to "private_val" (which is tracked) in outlined region
  // , and "int x=1;" is thus not tracked.
  // check-inst: define {{.*}} @func
  // check-inst: define {{.*}} @.omp_outlined
  // check-inst: call void @__typeart_alloc_stack_omp(i8* %{{[0-9]}}, i32 10, i64 1)
#pragma omp parallel for lastprivate(x), shared(e)
  for (int i = 0; i < 10; ++i) {
    // Analysis should not filter x, but e...
    MPI_Send((void*)x, *e);
  }
}

void foo() {
  // check-inst: define {{.*}} @foo
  // check-inst-NOT: call void @__typeart_alloc_stack
  int x = 1;
  int y = 2;
#pragma omp parallel
  { func(&x, &y); }
}

void func_other(int* x, int* e) {
  // lastprivate - addr(!) value of x is copied to "private_val" (which is tracked) in outlined region
  // check-inst: define {{.*}} @func_other
  // check-inst: define {{.*}} @.omp_outlined
  // check-inst: call void @__typeart_alloc_stack_omp(i8* %{{[0-9]}}, i32 10, i64 1)
#pragma omp parallel for lastprivate(x), shared(e)
  for (int i = 0; i < 10; ++i) {
    // Analysis should not filter x, but e...
    MPI_Send(x, *e);
  }
  MPI_Send(x, *e);
}

void bar(int x_other) {
  // check-inst: define {{.*}} @bar
  // check-inst: call void @__typeart_alloc_stack(i8* %{{[0-9]}}, i32 2, i64 1)
  int x = x_other;
  int y = 2;
#pragma omp parallel
  { func_other(&x, &y); }
}

// CHECK: TypeArtPass [Heap & Stack]
// CHECK-NEXT: Malloc :   0
// CHECK-NEXT: Free   :   0
// CHECK-NEXT: Alloca :   3
// CHECK-NEXT: Global :   0
