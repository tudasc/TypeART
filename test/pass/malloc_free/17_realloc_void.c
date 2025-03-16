// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s --check-prefix=REALLOC
// clang-format on
#include <stdlib.h>

void foo() {
  void* pd;
  pd = realloc(pd, 20 * sizeof(double));
}

// clang-format off

// REALLOC: __typeart_free({{i8\*|ptr}} [[POINTER:%[0-9a-z]+]])
// REALLOC-NEXT: [[POINTER2:%[0-9a-z]+]] = call{{( align [0-9]+)?}} {{i8\*|ptr}} @realloc({{i8\*|ptr}}{{( noundef)?}} [[POINTER]], i64{{( noundef)?}} 160)
// REALLOC-NEXT: __typeart_alloc({{i8\*|ptr}} [[POINTER2]], i32 {{3|11}}, i64 160)

// clang-format on
