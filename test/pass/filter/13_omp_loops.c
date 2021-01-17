// clang-format off
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %apply-typeart -typeart-alloca -call-filter -S 2>&1 | FileCheck %s
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | opt -O2 -S | %apply-typeart -typeart-alloca -call-filter -S 2>&1 | FileCheck %s --check-prefix=CHECK-opt
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | opt -O2 -S | %apply-typeart -typeart-alloca -call-filter -call-filter-impl=cg -call-filter-cg-file=%p/05_cg.ipcg -S 2>&1 | FileCheck %s --check-prefix=CHECK-exp-cg
// REQUIRES: openmp
// clang-format on

#include "omp.h"

extern void MPI_Mock(int, int, int);
extern void MPI_Send(void*, int);

void foo(int count) {
  int a = 0;
  int b = 1;
  int c = 2;
  int d = 3;
  int e = 4;
#pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < count; ++i) {
    // no (void*), so we assume benign (with deep analysis)
    MPI_Mock(a, b, c);
    for (int j = 0; j < count; ++j) {
      // Analysis should not filter d, but  e...
      MPI_Send((void*)&d, e);
    }
  }
}

// Standard filter (2 are kept: d in foo and d.addr in outlined region)
// CHECK: > Stack Memory
// CHECK-NEXT: Alloca                 :  25.00
// CHECK-NEXT: Stack call filtered %  :  92.00

// with opt only "d" in foo is tracked
// CHECK-opt: > Stack Memory
// CHECK-opt-NEXT: Alloca                 :  10.00
// CHECK-opt-NEXT: Stack call filtered %  :  90.00

// CG experimental filter
// CHECK-exp-cg: > Stack Memory
// CHECK-exp-cg-NEXT: Alloca                 :  10.00
// CHECK-exp-cg-NEXT: Stack call filtered %  :  90.00