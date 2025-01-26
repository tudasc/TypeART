// clang-format off
// RUN: %cpp-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s
// clang-format on

// REQUIRES: !llvm-14

#include <new>
// CHECK: invoke{{.*}} ptr @_Znwm(i64{{( noundef)?}} 4)
// CHECK: call void @__typeart_alloc(ptr [[POINTER:%[0-9a-z]+]], i32 12, i64 1)
int main() {
  try {
    auto s = new int;
  } catch (...) {
  }

  return 0;
}