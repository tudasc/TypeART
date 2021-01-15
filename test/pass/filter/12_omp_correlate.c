// clang-format off
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %apply-typeart -typeart-alloca -call-filter -S 2>&1 | FileCheck %s
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | opt -O2 -S | %apply-typeart -typeart-alloca -call-filter -S 2>&1 | FileCheck %s --check-prefix=CHECK-opt
// REQUIRES: openmp
// clang-format on

#include "omp.h"

extern void MPI_Mock(int, int, int);
extern void MPI_Send(void*, int);

void foo() {
  int a = 0;
  int b = 1;
  int c = 2;
  int d = 3;
  int e = 4;
#pragma omp parallel
  {
    // no (void*), so we assume benign (with deep analysis)
    MPI_Mock(a, b, c);
    // Analysis should filter d, but not e...
    MPI_Send((void*)d, e);
  }
}

// Standard filter (2 are kept: d in foo and d.addr in outlined region
// CHECK: > Stack Memory
// CHECK-NEXT: Alloca                 :  12.00
// CHECK-NEXT: Stack call filtered %  :  83.33

// with opt only "d" in foo is tracked
// CHECK-opt: > Stack Memory
// CHECK-opt-NEXT: Alloca                 :  5.00
// CHECK-opt-NEXT: Stack call filtered %  :  80.00