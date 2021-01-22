// clang-format off
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %apply-typeart -typeart-alloca -call-filter -S 2>&1 | FileCheck %s
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | opt -O2 -S | %apply-typeart -typeart-alloca -call-filter -S 2>&1 | FileCheck %s
// REQUIRES: openmp
// clang-format on

extern void MPI_send(void*);

float sum(const float* a, int n) {
  float total = 0.;
#pragma omp parallel for reduction(+ : total)
  for (int i = 0; i < n; i++) {
    total += a[i];
  }
  return total;
}

void foo() {
  const int n    = 10;
  float array[n] = {0};
  float loc      = sum(array, n);
  MPI_send((void*)&loc);
}

// CHECK: TypeArtPass [Heap & Stack]
// CHECK: Malloc :   0
// CHECK: Free   :   0
// CHECK: Alloca :   1
// CHECK: Global :   0
