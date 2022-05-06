// clang-format off
// RUN: %cpp-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s
// clang-format on

#define __cpp_aligned_new 1
#include <new>

struct C {
  int x;
};

C* foo() {
  return new (std::align_val_t(64)) C[10];
}

C* bar() {
  return new (std::align_val_t(128)) C;
}

// CHECK: [[POINTER:%[0-9a-z]+]] = call{{.*}} i8* @_ZnamSt11align_val_t(i64{{( noundef)?}} 40, i64{{( noundef)?}} 64)
// CHECK-NEXT: call void @__typeart_alloc(i8* [[POINTER]], i32 [[ID:2[5-9][0-9]]], i64 10)

// CHECK: [[POINTER2:%[0-9a-z]+]] = call{{.*}} i8* @_ZnwmSt11align_val_t(i64{{( noundef)?}} 4, i64{{( noundef)?}} 128)
// CHECK-NEXT: call void @__typeart_alloc(i8* [[POINTER2]], i32 [[ID]], i64 1)