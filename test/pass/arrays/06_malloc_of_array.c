// RUN: %c-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s

#include <stdlib.h>

void foo(int n) {
  int(*array)[3] = malloc(2 * sizeof(int[3]));
}

// CHECK: @__typeart_alloc(i8* %{{[0-9]+}}, i32 2, i64 6)