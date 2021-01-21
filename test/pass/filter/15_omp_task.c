// clang-format off
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %apply-typeart -typeart-alloca -call-filter -S 2>&1 | FileCheck %s
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | opt -O2 -S | %apply-typeart -typeart-alloca -call-filter -S 2>&1 | FileCheck %s --check-prefix=CHECK-opt
// REQUIRES: openmp
// clang-format on
extern void MPI_call(void*);

void foo() {
  int x;
#pragma omp parallel
  {
#pragma omp task
    { MPI_call(&x); }
  }
}

// CHECK: TypeArtPass [Heap & Stack]
// CHECK-NEXT: Malloc :   0
// CHECK-NEXT: Free   :   0
// CHECK-NOT: Alloca :   0

// CHECK-opt: TypeArtPass [Heap & Stack]
// CHECK-opt-NEXT: Malloc :   0
// CHECK-opt-NEXT: Free   :   0
// CHECK-opt-NEXT: Alloca :   1
// CHECK-opt-NEXT: Global :   0