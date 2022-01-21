// clang-format off
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %apply-typeart -typeart-stack -typeart-call-filter -S 2>&1 | %filecheck %s
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %opt -O2 -S | %apply-typeart -typeart-stack -typeart-call-filter -S 2>&1 | %filecheck %s

// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %apply-typeart -typeart-stack -typeart-call-filter -S | %filecheck %s --check-prefix=check-inst
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %opt -O2 -S | %apply-typeart -typeart-stack -typeart-call-filter -S | %filecheck %s --check-prefix=check-inst
// REQUIRES: openmp
// clang-format on

#include "omp.h"

extern void MPI_Send(void*, int);

void foo(int count) {
  // firstprivate > every thread has a private copy (d.addr) of d (which is passes to outlined region for copy)
  // check-inst: define {{.*}} @foo
  // check-inst-NOT: call void @__typeart_alloc_stack
  int d = 3;
  int e = 4;
  // check-inst: define {{.*}} @.omp_outlined
  // check-inst: %d.addr = alloca i64, align 8
  // check-inst-NEXT: %0 = bitcast i64* %d.addr to i8*
  // check-inst-NEXT: call void @__typeart_alloc_stack_omp(i8* %0, i32 3, i64 1)
#pragma omp parallel for schedule(dynamic, 1) firstprivate(d) shared(e)
  for (int i = 0; i < count; ++i) {
    // Analysis should not filter d, but e...
    MPI_Send((void*)&d, e);
  }
}

void bar(int count) {
  // lastprivate - value of d is copied to "private_val" (which is tracked) in outlined region and thus not tracked.
  // --> see "void bar2()" for different scenario with tracking of "d"
  // check-inst: define {{.*}} @bar
  // check-inst-NOT: call void @__typeart_alloc_stack
  int d = 3;
  int e = 4;
  // check-inst: define {{.*}} @.omp_outlined
  // check-inst: %d{{[0-9]}} = alloca i32
  // check-inst-NEXT: %0 = bitcast i32* %d{{[0-9]}} to i8*
  // check-inst-NEXT: call void @__typeart_alloc_stack_omp(i8* %0, i32 2, i64 1)
#pragma omp parallel for schedule(dynamic, 1) lastprivate(d) shared(e)
  for (int i = 0; i < count; ++i) {
    // Analysis should not filter d, but e...
    MPI_Send((void*)&d, e);
  }
}

void bar2(int count) {
  // check-inst: define {{.*}} @bar2
  // check-inst: %d = alloca
  // check-inst-NEXT: %0 = bitcast i32* %d to i8*
  // check-inst-NEXT: call void @__typeart_alloc_stack(i8* %0, i32 2, i64 1)
  int d = 3;
  int e = 4;
  // check-inst: define {{.*}} @.omp_outlined
  // check-inst: %d{{[0-9]}} = alloca i32
  // check-inst-NEXT: %0 = bitcast i32* %d{{[0-9]}} to i8*
  // check-inst-NEXT: call void @__typeart_alloc_stack_omp(i8* %0, i32 2, i64 1)
#pragma omp parallel for schedule(dynamic, 1) lastprivate(d) shared(e)
  for (int i = 0; i < count; ++i) {
    // Analysis should not filter d, but e...
    MPI_Send((void*)&d, e);
  }

  MPI_Send((void*)&d, e);
}

void foo_bar(int count) {
  // private: d, e are "randomly" initialised values inside outlined region (outer d,e are not passed)
  // check-inst: define {{.*}} @foo_bar
  // check-inst-NOT: call void @__typeart_alloc_stack
  int d = 3;
  int e = 4;
  // check-inst: define {{.*}} @.omp_outlined
  // check-inst: %d = alloca
  // check-inst-NEXT: %0 = bitcast i32* %d to i8*
  // check-inst-NEXT: call void @__typeart_alloc_stack_omp(i8* %0, i32 2, i64 1)
#pragma omp parallel for schedule(dynamic, 1) private(d, e)
  for (int i = 0; i < count; ++i) {
    MPI_Send((void*)&d, e);
  }
}

// CHECK: TypeArtPass [Heap & Stack]
// CHECK-NEXT: Malloc :   0
// CHECK-NEXT: Free   :   0
// CHECK-NEXT: Alloca :   5
// CHECK-NEXT: Global :   0
