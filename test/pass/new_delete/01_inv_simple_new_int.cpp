// clang-format off
// RUN: %cpp-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s
// clang-format on

// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK-NEXT: Free
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0

#include <new>
// CHECK: invoke{{.*}} {{i8\*|ptr}} @_Znwm(i64{{( noundef)?}} 4)
// CHECK: call void @__typeart_alloc({{i8\*|ptr}} [[POINTER:%[0-9a-z]+]], i32 12, i64 1)
int main() {
  try {
    auto s = new int;
  } catch (...) {
  }

  return 0;
}
