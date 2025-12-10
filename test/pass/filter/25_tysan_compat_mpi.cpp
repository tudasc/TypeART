// clang-format off
// RUN: %clang-cpp -O0 -g -fsanitize=type -emit-llvm -o - -c %s | %apply-typeart --typeart-stack=true --typeart-filter=true -S 2>&1 | %filecheck %s
// clang-format on

// REQUIRES: llvm-20 || llvm-21

#include <cmath>

extern void MPI_mock(void*);

int square(float in) {
  float num  = in * in;
  float calc = fabs(in) * num;
  MPI_mock((void*)&calc);
  return 1;
}

// CHECK: TypeArtPass [Heap & Stack]
// CHECK-NEXT: Malloc :   0
// CHECK-NEXT: Free   :   0
// CHECK-NEXT: Alloca :   1
// CHECK-NEXT: Global :   0

// CHECK: @__typeart_alloc_stack(ptr %calc
// CHECK: call void @__tysan_init