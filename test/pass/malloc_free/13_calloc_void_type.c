// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -S | %filecheck %s
// clang-format on
#include <stdlib.h>

void foo(int n) {
  void* pvd = calloc(10, sizeof(double));

  void* pvd2 = calloc(n, sizeof(double));
}

// clang-format off

// CHECK: [[POINTER:%[0-9a-z]+]] = call noalias{{( align [0-9]+)?}} {{i8\*|ptr}} @calloc(i64{{( noundef)?}} [[SIZE:[0-9a-z]+]], i64{{( noundef)?}} 8)
// CHECK-NEXT: call void @__typeart_alloc({{i8\*|ptr}} [[POINTER]], i32 {{(11|3)}}, i64 80)

// CHECK: [[POINTER2:%[0-9a-z]+]] = call noalias{{( align [0-9]+)?}} {{i8\*|ptr}} @calloc(i64{{( noundef)?}} [[SIZE2:%[0-9a-z]+]], i64{{( noundef)?}} 8)
// CHECK-NOT: call void @__typeart_alloc({{i8\*|ptr}} [[POINTER2]], i32 {{(11|3)}}, i64 [[SIZE2]])

// clang-format on
