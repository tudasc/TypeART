// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -S | %filecheck %s
// clang-format on
#include <stdlib.h>

void foo(int n) {
  void* pi = aligned_alloc(64, 20);

  void* pi2 = aligned_alloc(128, n);
}

// clang-format off
// CHECK: [[POINTER:%[0-9a-z]+]] = call noalias{{( align [0-9]+)?}} {{i8\*|ptr}} @aligned_alloc(i64{{( noundef)?}} 64, i64{{( noundef)?}} 20)
// CHECK-NEXT: call void @__typeart_alloc({{i8\*|ptr}} [[POINTER]], i32 {{(11|3)}}, i64 20)

// CHECK: [[POINTER2:%[0-9a-z]+]] = call noalias{{( align [0-9]+)?}} {{i8\*|ptr}} @aligned_alloc(i64{{( noundef)?}} 128, i64{{( noundef)?}} [[SIZE:%[0-9a-z]+]])
// CHECK-NEXT: call void @__typeart_alloc({{i8\*|ptr}} [[POINTER2]], i32 {{(11|3)}}, i64 [[SIZE]])
// clang-format on
