// RUN: %c-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s

#include <stdlib.h>

void foo(int n) {
  int(*array)[3] = malloc(2 * sizeof(int[3]));
}

// CHECK: @__typeart_alloc({{i8\*|ptr}} %{{[0-9a-z]+}}, i32 12, i64 6)