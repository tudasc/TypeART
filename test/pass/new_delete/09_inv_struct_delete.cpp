// clang-format off
// RUN: %cpp-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s
// clang-format on

#include <new>

struct S1 {
  int x;
  virtual ~S1() = default;
};

// CHECK: invoke{{.*}} i8* @_Znwm(i64{{( noundef)?}} 16)
// CHECK: call void @__typeart_alloc(i8* [[POINTER:%[0-9a-z]+]], i32 {{2[0-9]+}}, i64 1)
// CHECK: bitcast i8* [[POINTER]] to %struct.S1*
// CHECK-NOT: call void @_ZdlPv(i8*{{( noundef)?}} [[POINTER2:%[0-9a-z]+]])
// CHECK-NOT: call void @__typeart_free(i8* {{.*}}[[POINTER2]])
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

// CHECK: invoke{{.*}} i8* @_Znwm(i64{{( noundef)?}} 16)
// CHECK: call void @__typeart_alloc(i8* [[POINTER:%[0-9a-z]+]], i32 {{2[0-9]+}}, i64 1)
// CHECK: bitcast i8* [[POINTER]] to %struct.S1*
// CHECK-NOT: call void @_ZdaPv(i8*{{( noundef)?}} [[POINTER2:%[0-9a-z]+]])
// CHECK-NOT: call void @__typeart_free(i8* {{.*}}[[POINTER2]])
int main() {
  try {
    S1* ss = new S1;
    delete ss;  // TODO LLVM does not call _ZdaPv here, but in destructor @_ZN2S1D0Ev
  } catch (...) {
  }

  return 0;
}

// CHECK: @_ZN2S1D0Ev
// CHECK: call void @_ZdlPv(i8*{{( noundef)?}} [[POINTER2:%[0-9a-z]+]])
// CHECK-NEXT: call void @__typeart_free(i8* {{.*}}[[POINTER2]])

// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}2
// CHECK-NEXT: Free{{[ ]*}}:{{[ ]*}}1
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0
