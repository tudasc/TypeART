// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -S 2>&1 | FileCheck %s
// clang-format on
#include <stdlib.h>

void foo(int n) {
  int* pi = aligned_alloc(64, 20);

  int* pi2 = (int*)aligned_alloc(128, n);
}

// CHECK: [[POINTER:%[0-9]+]] = call noalias i8* @aligned_alloc(i64 64, i64 20)
// CHECK-NEXT: call void @__typeart_alloc(i8* [[POINTER]], i32 2, i64 5)
// CHECK-NEXT: bitcast i8* [[POINTER]] to i32*

// CHECK: [[POINTER2:%[0-9]+]] = call noalias i8* @aligned_alloc(i64 128, i64 [[SIZE:%[0-9]+]])
// CHECK-NOT: call void @__typeart_alloc(i8* [[POINTER2]], i32 2, i64 [[SIZE]])
// CHECK: call void @__typeart_alloc(i8* [[POINTER2]], i32 2, i64 %{{[0-9]+}})
// CHECK-NEXT: bitcast i8* [[POINTER2]] to i32*