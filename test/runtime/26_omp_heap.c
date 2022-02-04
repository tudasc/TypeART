// clang-format off
// RUN: %run %s --omp 2>&1 | %filecheck %s --check-prefix=CHECK-TSAN
// RUN: %run %s --omp 2>&1 | %filecheck %s
// REQUIRES: openmp && softcounter
// clang-format on

#include <stdlib.h>

void repeat_alloc_free(unsigned n) {
  for (int i = 0; i < n; i++) {
    double* d = (double*)malloc(sizeof(double) * n);
    free(d);
  }
}

int main(int argc, char** argv) {
  const int n = 1000;
  // CHECK: [Trace] TypeART Runtime Trace

#pragma omp parallel sections
  {
#pragma omp section
    repeat_alloc_free(n);
#pragma omp section
    repeat_alloc_free(n);
#pragma omp section
    repeat_alloc_free(n);
  }

  // CHECK-TSAN-NOT: ThreadSanitizer

  // CHECK-NOT: Error
  // CHECK: Allocation type detail (heap, stack, global)
  // CHECK: 6   : 3000 ,    0 ,    0 , double
  // CHECK: Free allocation type detail (heap, stack)
  // CHECK: 6   : 3000 ,    0 , double

  return 0;
}