// clang-format off
// RUN: %run %s --omp -typeart-call-filter 2>&1 | %filecheck %s --check-prefix=CHECK-TSAN
// RUN: %run %s -o -O2 --omp -typeart-call-filter 2>&1 | %filecheck %s --check-prefix=CHECK-TSAN

// RUN: %run %s -o -O2 --omp -typeart-call-filter 2>&1 | %filecheck %s
// RUN: %run %s --omp -typeart-call-filter 2>&1 | %filecheck %s
// REQUIRES: openmp && softcounter
// clang-format on

#include <stdlib.h>

void ptr(const int n) {
  // Sections can sometimes cause Max. Heap Allocs to be 1 (instead of more likely 2), if
  // thread execution order always frees one pointer before malloc of other.
#pragma omp parallel sections num_threads(2)
  {
#pragma omp section
    for (int i = 1; i <= n; i++) {
      double* d = (double*)malloc(sizeof(double) * n);
      free(d);
    }
#pragma omp section
    for (int i = 1; i <= n; i++) {
      double* e = (double*)malloc(2 * sizeof(double) * n);
      free(e);
    }
  }
}

int main(int argc, char** argv) {
  const int n = 100;

  ptr(n);

  // CHECK-TSAN-NOT: ThreadSanitizer

  // CHECK: [Trace] TypeART Runtime Trace
  // CHECK-NOT: [Error]
  // CHECK: Alloc Stats from softcounters
  // CHECK-NEXT: Total heap                 : 200 ,  200 ,    -
  // CHECK-NEXT: Total stack                :   0 ,    0 ,    -
  // CHECK-NEXT: Total global               :   0 ,    0 ,    -
  // CHECK-NEXT: Max. Heap Allocs           :   {{[1-2]}} ,    - ,    -
  // CHECK-NEXT: Max. Stack Allocs          :   0 ,    - ,    -
  // CHECK-NEXT: Addresses checked          :   0 ,    - ,    -
  // CHECK-NEXT: Distinct Addresses checked :   0 ,    - ,    -
  // CHECK-NEXT: Addresses re-used          :   0 ,    - ,    -
  // CHECK-NEXT: Addresses missed           :   0 ,    - ,    -
  // CHECK-NEXT: Distinct Addresses missed  :   0 ,    - ,    -
  // CHECK-NEXT: Total free heap            : 200 ,  200 ,    -
  // CHECK-NEXT: Total free stack           :   0 ,    0 ,    -
  // CHECK-NEXT: OMP Stack/Heap/Free        :   0 ,  200 ,  200
  // CHECK-NEXT: Null/Zero/NullZero Addr    :   0 ,    0 ,    0
  // CHECK-NEXT: User-def. types            :   0 ,    - ,    -
  // CHECK-NEXT: Estimated memory use (KiB) :   {{[0-9]+}} ,    - ,    -
  // CHECK-NEXT: Bytes per node map/stack   :  96 ,    8 ,    -
  // CHECK-NEXT: {{(#|-)+}}
  // CHECK-NEXT: Allocation type detail (heap, stack, global)
  // CHECK: {{(#|-)+}}
  // CHECK-NEXT: Free allocation type detail (heap, stack)
  // CHECK-NEXT: 6 : 200 ,    0 , double
  // CHECK: Per-thread counter values (2 threads)
  // CHECK-NEXT: Thread Heap Allocs       : 100 ,  100
  // CHECK-NEXT: Thread Heap Arrays       : 100 ,  100
  // CHECK-NEXT: Thread Heap Allocs Free  : 100 ,  100
  // CHECK-NEXT: Thread Heap Arrays Free  : 100 ,  100
  // CHECK-NEXT: Thread Stack Allocs      :   0 ,    0
  // CHECK-NEXT: Thread Stack Arrays      :   0 ,    0
  // CHECK-NEXT: Thread Max. Stack Allocs :   0 ,    0
  // CHECK-NEXT: Thread Stack Allocs Free :   0 ,    0
  // CHECK-NEXT: Thread Stack Array Free  :   0 ,    0

  return 0;
}