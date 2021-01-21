// clang-format off
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %apply-typeart -typeart-alloca -call-filter -S 2>&1 | FileCheck %s
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | opt -O2 -S | %apply-typeart -typeart-alloca -call-filter -S 2>&1 | FileCheck %s --check-prefix=CHECK-opt
// REQUIRES: openmp
// clang-format on
extern void MPI_call(void*);

void func(int* x) {
#pragma omp parallel
  { MPI_call(x); }
}

void foo() {
  int x;
#pragma omp parallel
  { func(&x); }
}

// TODO filter stack vars that get assigned from function argument, or have no explicit source loc (only in omp context
// or generally?)

// CHECK: > Stack Memory
// CHECK-NEXT: Alloca                 :   8.00
// CHECK-NEXT: Stack call filtered %  :  50.00

// CHECK-opt: > Stack Memory
// CHECK-opt-NEXT: Alloca                 :  3.00
// CHECK-opt-NEXT: Stack call filtered %  :  0.00