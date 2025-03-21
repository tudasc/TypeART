// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -S | %apply-typeart --typeart-stack=true --typeart-heap=false --typeart-filter=true -S 2>&1 | %filecheck %s
// clang-format on

#include <stdlib.h>

extern double* a;

int main(int argc, char** argv) {
  int n = argc * 2;

  a = (double*)malloc(sizeof(double) * n);

  free(a);

  return 0;
}
// CHECK:      TypeArtPass [Stack]
// CHECK-NEXT  Malloc :   0
// CHECK-NEXT  Free   :   0
// CHECK-NEXT  Alloca :   0
// CHECK-NEXT  Global :   0

// This is added with legacy type parser:
// : call void @__typeart_alloc(

// CHECK: call void @__typeart_free(

// CHECK-NOT: call void @__typeart_leave_scope
