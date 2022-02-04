// RUN: %run %s -o -O3 2>&1 | %filecheck %s
// REQUIRES: softcounter

#include <stdlib.h>

int main(void) {
  for (int i = 1; i <= 5; ++i) {
    // 5 heap alloc and free: one single double, and then arrays
    // max heap (concurrently) 1
    double* d = (double*)malloc(i * sizeof(double));
    free(d);
  }
  return 0;
}

// CHECK: Alloc Stats from softcounters
// CHECK-NEXT: Total heap                 :   5 ,    4 ,    -
// CHECK-NEXT: Total stack                :   0 ,    0 ,    -
// CHECK-NEXT: Total global               :   0 ,    0 ,    -
// CHECK-NEXT: Max. Heap Allocs           :   1 ,    - ,    -
// CHECK-NEXT: Max. Stack Allocs          :   0 ,    - ,    -
// CHECK-NEXT: Addresses checked          :   0 ,    - ,    -
// CHECK-NEXT: Distinct Addresses checked :   0 ,    - ,    -
// CHECK-NEXT: Addresses re-used          :   0 ,    - ,    -
// CHECK-NEXT: Addresses missed           :   0 ,    - ,    -
// CHECK-NEXT: Distinct Addresses missed  :   0 ,    - ,    -
// CHECK-NEXT: Total free heap            :   5 ,    4 ,    -
// CHECK-NEXT: Total free stack           :   0 ,    0 ,    -
// CHECK-NEXT: OMP Stack/Heap/Free        :   0 ,    0 ,    0
// CHECK-NEXT: Null/Zero/NullZero Addr    :   0 ,    0 ,    0
// CHECK-NEXT: User-def. types            :   0 ,    - ,    -
// CHECK-NEXT: Estimated memory use (KiB) :   4 ,    - ,    -
// CHECK-NEXT: Bytes per node map/stack   :  96 ,    8 ,    -
// CHECK-NEXT: {{(#|-)+}}
// CHECK-NEXT: Allocation type detail (heap, stack, global)
// CHECK-NEXT: 6 :   5 ,    0 ,    0 , double
// CHECK-NEXT: {{(#|-)+}}
// CHECK-NEXT: Free allocation type detail (heap, stack)
// CHECK-NEXT: 6 :   5 ,    0 , double