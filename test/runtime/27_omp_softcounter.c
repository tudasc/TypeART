// clang-format off
// RUN: %run %s -o -O2 --omp 2>&1 | FileCheck %s
// REQUIRES: openmp && softcounter
// clang-format on

#include <stdlib.h>

void ptr (const int n){
#pragma omp parallel sections
  {
#pragma omp section
    for (int i = 1; i <= n; i++) {
      double* d = (double*) malloc(sizeof(double) * n);
      free(d);
    }
#pragma omp section
    for (int i = 1; i <= n; i++) {
      double* d = (double*) malloc(sizeof(double) * n);
      free(d);
    }
  }
}

int main(int argc, char** argv) {
  const int n = 100;

  ptr(n);

  // CHECK: [Trace] TypeART Runtime Trace
  // CHECK-NOT: [Error]
  // CHECK: Alloc Stats from softcounters
  // CHECK-NEXT: Total heap                 : 200 ,  200 ,    -
  // CHECK-NEXT: Total stack                :  {{[0-9]+}} ,    0 ,    -
  // CHECK-NEXT: Total global               :   0 ,    0 ,    -
  // CHECK-NEXT: Max. Heap Allocs           :   2 ,    - ,    -
  // CHECK-NEXT: Max. Stack Allocs          :  {{[0-9]+}} ,    - ,    -
  // CHECK-NEXT: Addresses checked          :   0 ,    - ,    -
  // CHECK-NEXT: Distinct Addresses checked :   0 ,    - ,    -
  // CHECK-NEXT: Addresses re-used          :   0 ,    - ,    -
  // CHECK-NEXT: Addresses missed           :   0 ,    - ,    -
  // CHECK-NEXT: Distinct Addresses missed  :   0 ,    - ,    -
  // CHECK-NEXT: Total free heap            : 200 ,  200 ,    -
  // CHECK-NEXT: Total free stack           :  {{[0-9]+}} ,    0 ,    -
  // CHECK-NEXT: OMP Stack/Heap/Free        :  {{[0-9]+}} ,  200 ,  200
  // CHECK-NEXT: Null/Zero/NullZero Addr    :   0 ,    0 ,    0
  // CHECK-NEXT: User-def. types            :   0 ,    - ,    -
  // CHECK-NEXT: Estimated memory use (KiB) :   {{[0-9]+}} ,    - ,    -
  // CHECK-NEXT: Bytes per node map/stack   :  96 ,    8 ,    -
  // CHECK-NEXT: {{(#|-)+}}
  // CHECK-NEXT: Allocation type detail (heap, stack, global)
  // CHECK: {{(#|-)+}}
  // CHECK-NEXT: Free allocation type detail (heap, stack)
  // CHECK: 6 : 200 ,    0 , double


  return 0;
}