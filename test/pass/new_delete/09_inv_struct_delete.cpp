// clang-format off
// RUN: clang++ -S -emit-llvm %s -o - | opt -load %pluginpath/analysis/meminstfinderpass.so -load %pluginpath/%pluginname %pluginargs -S 2>&1 | FileCheck %s
// clang-format on

#include <new>

struct S1 {
  int x;
  virtual ~S1() = default;
};

// CHECK: invoke i8* @_Znwm(i64 16)
// CHECK: call void @__typeart_alloc(i8* [[POINTER:%[0-9]+]], i32 {{2[0-9]+}}, i64 1)
// CHECK: bitcast i8* [[POINTER]] to %struct.S1*
// CHECK-NOT: call void @_ZdlPv(i8* [[POINTER2:%[0-9]+]])
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

// CHECK: invoke i8* @_Znwm(i64 16)
// CHECK: call void @__typeart_alloc(i8* [[POINTER:%[0-9]+]], i32 {{2[0-9]+}}, i64 1)
// CHECK: bitcast i8* [[POINTER]] to %struct.S1*
// CHECK-NOT: call void @_ZdaPv(i8* [[POINTER2:%[0-9]+]])
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
// CHECK: call void @_ZdlPv(i8* [[POINTER2:%[0-9]+]])
// CHECK-NEXT: call void @__typeart_free(i8* {{.*}}[[POINTER2]])

// CHECK: Malloc{{[ ]*}}:{{[ ]*}}2
// CHECK: Free{{[ ]*}}:{{[ ]*}}1
// CHECK: Alloca{{[ ]*}}:{{[ ]*}}0
