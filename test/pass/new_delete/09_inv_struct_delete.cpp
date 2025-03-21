// clang-format off
// RUN: %cpp-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s
// clang-format on

// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}2
// CHECK-NEXT: Free{{[ ]*}}:{{[ ]*}}1
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0

#include <new>

struct S1 {
  int x;
  virtual ~S1() = default;
};

// CHECK: invoke{{.*}} {{i8\*|ptr}} @_Znwm(i64{{( noundef)?}} 16)
// CHECK: call void @__typeart_alloc({{i8\*|ptr}} [[POINTER:%[0-9a-z]+]], i32 {{2[0-9]+}}, i64 1)
void foo() {
  S1* b{nullptr};
  try {
    b = new S1;
  } catch (...) {
  }
  if (b != nullptr) {
    delete b;
  }
}

// CHECK: invoke{{.*}} {{i8\*|ptr}} @_Znwm(i64{{( noundef)?}} 16)
// CHECK: call void @__typeart_alloc({{i8\*|ptr}} [[POINTER:%[0-9a-z]+]], i32 {{2[0-9]+}}, i64 1)
int main() {
  try {
    S1* ss = new S1;
    delete ss;  // TODO LLVM does not call _ZdaPv here, but in destructor @_ZN2S1D0Ev
  } catch (...) {
  }

  return 0;
}

// CHECK: @_ZN2S1D0Ev
// CHECK: call void @_ZdlPv{{m?}}({{i8\*|ptr}}{{( noundef)?}} [[POINTER2:%[0-9a-z]+]]
// CHECK-NEXT: call void @__typeart_free({{i8\*|ptr}} {{.*}}[[POINTER2]])
