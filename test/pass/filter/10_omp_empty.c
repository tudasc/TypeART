// clang-format off
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %apply-typeart -typeart-stack -typeart-call-filter  -S 2>&1 | %filecheck %s
// REQUIRES: openmp
// clang-format on

#include "omp.h"

// CHECK-NOT: {{.*}} __typeart_alloc

void foo(int* x) {
#pragma omp parallel  // transformed to @__kmpc_fork_call
  { *x = -1; }

#pragma omp parallel for
  for (int i = 0; i < x[10]; ++i) {
    x[i] = i;
  }
}

// Standard filter
// CHECK: > Stack Memory
// CHECK-NEXT: Alloca                 :
// CHECK-NEXT: Stack call filtered %  :  100.00