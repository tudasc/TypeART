// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -S | FileCheck %s
// clang-format on
#include <stdlib.h>

void foo(int n) {
  void* pvd = calloc(10, sizeof(double));

  void* pvd2 = calloc(n, sizeof(double));
}

// CHECK: [[POINTER:%[0-9a-z]+]] = call noalias i8* @calloc(i64 [[SIZE:[0-9a-z]+]], i64 8)
// CHECK-NEXT: call void @__typeart_alloc(i8* [[POINTER]], i32 0, i64 80)
// CHECK-NOT: bitcast i8* [[POINTER]] to double*

// CHECK: [[POINTER2:%[0-9a-z]+]] = call noalias i8* @calloc(i64 [[SIZE2:%[0-9a-z]+]], i64 8)
// CHECK-NOT: call void @__typeart_alloc(i8* [[POINTER2]], i32 0, i64 [[SIZE2]])
// CHECK-NOT: bitcast i8* [[POINTER]] to double*