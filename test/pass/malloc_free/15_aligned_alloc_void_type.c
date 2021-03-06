// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -S | FileCheck %s
// clang-format on
#include <stdlib.h>

void foo(int n) {
  void* pi = aligned_alloc(64, 20);

  void* pi2 = aligned_alloc(128, n);
}

// CHECK: [[POINTER:%[0-9]+]] = call noalias i8* @aligned_alloc(i64 64, i64 20)
// CHECK-NEXT: call void @__typeart_alloc(i8* [[POINTER]], i32 0, i64 20)
// CHECK-NOT: bitcast i8* [[POINTER]] to {{.*}}*

// CHECK: [[POINTER2:%[0-9]+]] = call noalias i8* @aligned_alloc(i64 128, i64 [[SIZE:%[0-9]+]])
// CHECK-NEXT: call void @__typeart_alloc(i8* [[POINTER2]], i32 0, i64 [[SIZE]])
// CHECK-NOT: bitcast i8* [[POINTER]] to {{.*}}*