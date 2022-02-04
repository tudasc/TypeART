// RUN: %run %s -o -O3 2>&1 | %filecheck %s
// REQUIRES: softcounter

void __typeart_leave_scope(int alloca_count);

int main(void) {
  __typeart_leave_scope(0);  // This simply "triggers" the runtime
  return 0;
}

// CHECK: Alloc Stats from softcounters
// CHECK-NEXT: Total heap                 :   0 ,    0 ,    -
// CHECK-NEXT: Total stack                :   0 ,    0 ,    -
// CHECK-NEXT: Total global               :   0 ,    0 ,    -
// CHECK-NEXT: Max. Heap Allocs           :   0 ,    - ,    -
// CHECK-NEXT: Max. Stack Allocs          :   0 ,    - ,    -
// CHECK-NEXT: Addresses checked          :   0 ,    - ,    -
// CHECK-NEXT: Distinct Addresses checked :   0 ,    - ,    -
// CHECK-NEXT: Addresses re-used          :   0 ,    - ,    -
// CHECK-NEXT: Addresses missed           :   0 ,    - ,    -
// CHECK-NEXT: Distinct Addresses missed  :   0 ,    - ,    -
// CHECK-NEXT: Total free heap            :   0 ,    0 ,    -
// CHECK-NEXT: Total free stack           :   0 ,    0 ,    -
// CHECK-NEXT: OMP Stack/Heap/Free        :   0 ,    0 ,    0
// CHECK-NEXT: Null/Zero/NullZero Addr    :   0 ,    0 ,    0
// CHECK-NEXT: User-def. types            :   0 ,    - ,    -
// CHECK-NEXT: Estimated memory use (KiB) :   4 ,    - ,    -
// CHECK-NEXT: Bytes per node map/stack   :  96 ,    8 ,    -
// CHECK-NEXT: {{(#|-)+}}
// CHECK-NEXT: Allocation type detail (heap, stack, global)
// CHECK-NEXT: {{(#|-)+}}
// CHECK-NEXT: Free allocation type detail (heap, stack)