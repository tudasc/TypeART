// clang-format off
// RUN: %cpp-to-llvm %s | %apply-typeart -S 2>&1 | FileCheck %s
// FIXME array cookie
// clang-format on

#include <new>

struct C {
  int x;
  ~C();
};

C* foo() {
  return new (std::nothrow) C[10];
}

// CHECK: [[POINTER:%[0-9]+]] = call noalias i8* @_ZnamRKSt9nothrow_t
// CHECK-NEXT: call void @__typeart_alloc(i8* [[POINTER]], i32 2{{[5-9][0-9]}}, i64 12)