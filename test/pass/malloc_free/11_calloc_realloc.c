// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s --check-prefix=REALLOC
// RUN: %c-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s
// clang-format on
#include <stdlib.h>

int main() {
  double* pd = calloc(10, sizeof(double));

  pd = realloc(pd, 20 * sizeof(double));

  return 0;
}

// clang-format off

// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}2
// CHECK-NEXT: Free{{[ ]*}}:{{[ ]*}}0
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0

// CHECK: [[POINTER:%[0-9a-z]+]] = call noalias{{( align [0-9]+)?}} {{i8\*|ptr}} @calloc(i64{{( noundef)?}} [[SIZE:[0-9]+]], i64{{( noundef)?}} 8)
// CHECK-NEXT: call void @__typeart_alloc({{i8\*|ptr}} [[POINTER]], i32 23, i64 [[SIZE]])

// REALLOC: __typeart_free({{i8\*|ptr}} [[POINTER:%[0-9a-z]+]])
// REALLOC-NEXT: [[POINTER2:%[0-9a-z]+]] = call{{( align [0-9]+)?}} {{i8\*|ptr}} @realloc({{i8\*|ptr}}{{( noundef)?}} [[POINTER]], i64{{( noundef)?}} 160)
// REALLOC-NEXT: __typeart_alloc({{i8\*|ptr}} [[POINTER2]], i32 23, i64 20)

// clang-format on
