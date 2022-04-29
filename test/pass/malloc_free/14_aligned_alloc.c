// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s
// clang-format on
#include <stdlib.h>

void foo(int n) {
  int* pi = aligned_alloc(64, 20);

  int* pi2 = (int*)aligned_alloc(128, n);
}

// clang-format off
// CHECK: [[POINTER:%[0-9a-z]+]] = call noalias{{( align [0-9]+)?}} i8* @aligned_alloc(i64{{( noundef)?}} 64, i64{{( noundef)?}} 20)
// CHECK-NEXT: call void @__typeart_alloc(i8* [[POINTER]], i32 2, i64 5)
// CHECK-NEXT: bitcast i8* [[POINTER]] to i32*

// CHECK: [[POINTER2:%[0-9a-z]+]] = call noalias{{( align [0-9]+)?}} i8* @aligned_alloc(i64{{( noundef)?}} 128, i64{{( noundef)?}} [[SIZE:%[0-9a-z]+]])
// CHECK-NOT: call void @__typeart_alloc(i8* [[POINTER2]], i32 2, i64 [[SIZE]])
// CHECK: call void @__typeart_alloc(i8* [[POINTER2]], i32 2, i64 %{{[0-9a-z]+}})
// CHECK-NEXT: bitcast i8* [[POINTER2]] to i32*
// clang-format on
