// clang-format off
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %apply-typeart -typeart-alloca -call-filter -S 2>&1 | FileCheck %s
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | opt -O2 -S | %apply-typeart -typeart-alloca -call-filter -S 2>&1 | FileCheck %s
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

// CHECK: TypeArtPass [Heap & Stack]
// CHECK-NEXT: Malloc :   0
// CHECK-NEXT: Free   :   0
// CHECK-NEXT: Alloca :   1
// CHECK-NEXT: Global :   0