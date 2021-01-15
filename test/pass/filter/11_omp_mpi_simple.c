// clang-format off
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %apply-typeart -typeart-alloca -call-filter  -call-filter-deep=true -S 2>&1 | FileCheck %s
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | opt -O2 -S | %apply-typeart -typeart-alloca -call-filter  -call-filter-deep=true -S 2>&1 | FileCheck %s --check-prefix=CHECK-opt
// REQUIRES: openmp
// clang-format on

#include "omp.h"

void MPI_test(void*);
void foo(int* x) {
#pragma omp parallel  // transformed to @__kmpc_fork_call
  { MPI_test(x); }
}

// Standard filter (keeps int* x in foo and the corresponding alloca value in the outlined region)
// CHECK: > Stack Memory
// CHECK-NEXT: Alloca                 :   4.0
// CHECK-NEXT: Stack call filtered %  :  50.0

// with opt we keep the int* x and nothing else
// CHECK-opt: > Stack Memory
// CHECK-opt-NEXT: Alloca                 :  1.00
// CHECK-opt-NEXT: Stack call filtered %  :  0.00