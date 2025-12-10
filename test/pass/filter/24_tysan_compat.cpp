// clang-format off
// RUN: %clang-cpp -O0 -g -fsanitize=type -emit-llvm -o - -c %s | %apply-typeart --typeart-stack=true --typeart-filter=true -S 2>&1 | %filecheck %s
// clang-format on

// REQUIRES: llvm-20 || llvm-21

#include <cmath>

int square(float in) {
  float num = in * in;
  float b   = fabs(num);
  return 1;
}

// CHECK: TypeArtPass [Heap & Stack]
// CHECK-NEXT: Malloc :   0
// CHECK-NEXT: Free   :   0
// CHECK-NEXT: Alloca :   0
// CHECK-NEXT: Global :   0

// CHECK: call void @__tysan_init