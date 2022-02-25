// clang-format off
// RUN: %run %s -o -O0 --omp -typeart-filter-pointer-alloca=false 2>&1 | %filecheck %s --check-prefix=CHECK-TSAN
// RUN: %run %s -o -O0 --omp -typeart-filter-pointer-alloca=false 2>&1 | %filecheck %s --check-prefixes=CHECK,ERROR
// REQUIRES: openmp && softcounter
// clang-format on

#include <stdlib.h>

void foo() {
  double d[32];
  float f[32];
}

void ptr(const int n) {
#pragma omp parallel sections num_threads(2)
  {
#pragma omp section
    for (int i = 1; i <= n; i++) {
      foo();
    }
#pragma omp section
    for (int i = 1; i <= n; i++) {
      foo();
    }
  }
}

int main(int argc, char** argv) {
  const int n = 100;

  ptr(n);
  // CHECK-TSAN-NOT: ThreadSanitizer
  // ERROR-NOT: [Error]

  // CHECK: [Trace] TypeART Runtime Trace
  // CHECK: Alloc Stats from softcounters
  // CHECK-NEXT: Total heap                 :   0 ,    0 ,    -
  // CHECK: Total stack                     :   {{[0-9]+}} ,   400 ,    -
  // CHECK-NEXT: Total global               :   {{[0-9]+}} ,    {{[0-9]+}} ,    -
  // CHECK-NEXT: Max. Heap Allocs           :   0 ,    - ,    -
  // CHECK-NEXT: Max. Stack Allocs          :  16 ,    - ,    -
  // CHECK-NEXT: Addresses checked          :   0 ,    - ,    -
  // CHECK-NEXT: Distinct Addresses checked :   0 ,    - ,    -
  // CHECK-NEXT: Addresses re-used          :   0 ,    - ,    -
  // CHECK-NEXT: Addresses missed           :   0 ,    - ,    -
  // CHECK-NEXT: Distinct Addresses missed  :   0 ,    - ,    -
  // CHECK-NEXT: Total free heap            :   0 ,    0 ,    -
  // CHECK-NEXT: Total free stack           : 423 ,  400 ,    -
  // CHECK-NEXT: OMP Stack/Heap/Free        :  {{[0-9]+}} ,    0 ,    0
  // CHECK-NEXT: Null/Zero/NullZero Addr    :   0 ,    0 ,    0
  // CHECK-NEXT: User-def. types            :   0 ,    - ,    -
  // CHECK-NEXT: Estimated memory use (KiB) :   {{[0-9]+}} ,    - ,    -
  // CHECK-NEXT: Bytes per node map/stack   :  96 ,    8 ,    -
  // CHECK-NEXT: {{(#|-)+}}
  // CHECK-NEXT: Allocation type detail (heap, stack, global)
  // CHECK: {{(#|-)+}}
  // CHECK-NEXT: Free allocation type detail (heap, stack)
  // CHECK: 5 : 0 ,    200 , float
  // CHECK: 6 : 0 ,    200 , double
  // CHECK: Per-thread counter values (2 threads)
  // CHECK-NEXT: Thread Heap Allocs       : 0 ,  0
  // CHECK-NEXT: Thread Heap Arrays       : 0 ,  0
  // CHECK-NEXT: Thread Heap Allocs Free  : 0 ,  0
  // CHECK-NEXT: Thread Heap Arrays Free  : 0 ,  0
  // CHECK-NEXT: Thread Stack Allocs      : {{[0-9]+}} ,   {{[0-9]+}}
  // CHECK-NEXT: Thread Stack Arrays      : 200 ,    200
  // CHECK-NEXT: Thread Max. Stack Allocs : {{[0-9]+}} ,   {{[0-9]+}}
  // CHECK-NEXT: Thread Stack Allocs Free : {{[0-9]+}} ,   {{[0-9]+}}
  // CHECK-NEXT: Thread Stack Array Free  : 200 ,    200

  return 0;
}