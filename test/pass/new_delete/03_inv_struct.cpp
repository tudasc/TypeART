// clang-format off
// RUN: %cpp-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s
// clang-format on

// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK-NEXT: Free
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0

#include <new>

struct S1 {
  int x;
  virtual ~S1() = default;
};

// CHECK: invoke{{.*}} {{i8\*|ptr}} @_Znwm(i64{{( noundef)?}} 16)
// CHECK: call void @__typeart_alloc({{i8\*|ptr}} [[POINTER:%[0-9a-z]+]], i32 {{2[0-9]+}}, i64 1)
int main() {
  try {
    S1* ss = new S1;
  } catch (...) {
  }

  return 0;
}
