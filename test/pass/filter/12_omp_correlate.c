// clang-format off
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %apply-typeart -typeart-stack -typeart-call-filter -typeart-filter-pointer-alloca=false -S 2>&1 | %filecheck %s
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %opt -O2 -S | %apply-typeart -typeart-stack -typeart-call-filter -typeart-filter-pointer-alloca=false -S 2>&1 | %filecheck %s --check-prefix=CHECK-opt
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %opt -O2 -S | %apply-typeart -typeart-stack -typeart-call-filter -typeart-call-filter-impl=cg -typeart-call-filter-cg-file=%p/05_cg.ipcg -typeart-filter-pointer-alloca=false -S 2>&1 | %filecheck %s --check-prefix=CHECK-exp-cg

// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %apply-typeart -typeart-stack -typeart-call-filter -typeart-filter-pointer-alloca=false -S | %filecheck %s --check-prefix=check-inst
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %opt -O2 -S | %apply-typeart -typeart-stack -typeart-call-filter -typeart-filter-pointer-alloca=false -S | %filecheck %s --check-prefix=check-inst
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %opt -O2 -S | %apply-typeart -typeart-stack -typeart-call-filter -typeart-call-filter-impl=cg -typeart-call-filter-cg-file=%p/05_cg.ipcg -typeart-filter-pointer-alloca=false -S | %filecheck %s --check-prefix=check-inst
// REQUIRES: openmp
// clang-format on

#include "omp.h"

extern void MPI_Mock(int, int, int);
extern void MPI_Send(void*, int);

void foo() {
  int a = 0;
  int b = 1;
  int c = 2;
  // check-inst: define {{.*}} @foo
  // check-inst: %d = alloca
  // check-inst: [[POINTER:%[0-9a-z]+]] = bitcast i32* %d to i8*
  // check-inst: call void @__typeart_alloc_stack(i8* [[POINTER]], i32 2, i64 1)
  // check-inst-not: __typeart_alloc_stack_omp
  int d = 3;
  int e = 4;
#pragma omp parallel
  {
    // no (void*), so we assume benign (with deep analysis)
    MPI_Mock(a, b, c);
    // Analysis should not filter d, but e...
    MPI_Send((void*)d, e);
  }
}

// Standard filter
// CHECK: > Stack Memory
// CHECK-NEXT: Alloca                 :  12.00
// CHECK-NEXT: Stack call filtered %  :  91.67

// with opt only "d" in foo is tracked
// CHECK-opt: > Stack Memory
// CHECK-opt-NEXT: Alloca                 :  5.00
// CHECK-opt-NEXT: Stack call filtered %  :  80.00

// CG experimental filter
// CHECK-exp-cg: > Stack Memory
// CHECK-exp-cg-NEXT: Alloca                 :  5.00
// CHECK-exp-cg-NEXT: Stack call filtered %  :  80.00