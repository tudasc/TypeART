// RUN: ASAN_OPTIONS=detect_leaks=0 %run %s -o -O1 2>&1 | %filecheck %s
// REQUIRES: softcounter

#include <stdlib.h>

int main(void) {
  for (int i = 1; i <= 6; ++i) {
    // 6 heap alloc
    // max heap (concurrently) 6
    double* d = (double*)malloc(sizeof(double));
  }

  return 0;
}

// CHECK: Alloc Stats from softcounters
// CHECK-NEXT: Total heap                 :   6 ,    0 ,    -
// CHECK-NEXT: Total stack                :   0 ,    0 ,    -
// CHECK-NEXT: Total global               :   0 ,    0 ,    -
// CHECK-NEXT: Max. heap                  :   6 ,    - ,    -
// CHECK-NEXT: Max. stack                 :   0 ,    - ,    -
// CHECK-NEXT: Addresses checked          :   0 ,    - ,    -
// CHECK-NEXT: Distinct addresses checked :   0 ,    - ,    -
// CHECK-NEXT: Addresses re-used          :   0 ,    - ,    -
// CHECK-NEXT: Addresses missed           :   0 ,    - ,    -
// CHECK-NEXT: Distinct addresses missed  :   0 ,    - ,    -
// CHECK-NEXT: Total free heap            :   0 ,    0 ,    -
// CHECK-NEXT: Total free stack           :   0 ,    0 ,    -
// CHECK-NEXT: OMP stack/heap/free        :   0 ,    0 ,    0
// CHECK-NEXT: Null/Zero/NullZero addr    :   0 ,    0 ,    0
// CHECK-NEXT: User-def. types            :   0 ,    - ,    -
// CHECK-NEXT: Distinct query types       :   0 ,    - ,    -
// CHECK-NEXT: {{(#|-)+}}
// CHECK-NEXT: Allocation type detail (heap, stack, global)
// CHECK-NEXT: 24 :   6 ,    0 ,    0 , double
// CHECK-NEXT: {{(#|-)+}}
// CHECK-NEXT: Free allocation type detail (heap, stack)
// CHECK-NEXT: 24 :   0 ,    0 , double