// clang-format off
// RUN: clang++ -S -emit-llvm %s -o - | opt -load %pluginpath/analysis/meminstfinderpass.so -load %pluginpath/%pluginname %pluginargs -S 2>&1 | FileCheck %s
// FIXME revisit for array cookie handling
// clang-format on

#include <new>

struct S1 {
  int x;
  virtual ~S1() = default;
};

// CHECK: invoke i8* @_Znam(i64 56)
// CHECK: call void @__typeart_alloc(i8* [[POINTER:%[0-9]+]], i32 {{2[0-9]+}}, i64 3)
// CHECK: [[MEMORYBLOB:%[0-9]+]] = getelementptr inbounds i8, i8* %{{[0-9]+}}, i64 -8
// CHECK: call void @_ZdaPv(i8* [[MEMORYBLOB]])
// CHECK-NEXT: call void @__typeart_free(i8* [[MEMORYBLOB]])
void foo() {
  S1* b{nullptr};
  try {
    b = new S1[3];
  } catch (...) {
  }
  if (b != nullptr) {
    delete[] b;
  }
}

// CHECK: invoke i8* @_Znam(i64 40)
// CHECK: call void @__typeart_alloc(i8* [[POINTER:%[0-9]+]], i32 {{2[0-9]+}}, i64 2)
// CHECK: [[MEMORYBLOB:%[0-9]+]] = getelementptr inbounds i8, i8* %{{[0-9]+}}, i64 -8
// CHECK: call void @_ZdaPv(i8* [[MEMORYBLOB]])
// CHECK-NEXT: call void @__typeart_free(i8* [[MEMORYBLOB]])
int main() {
  try {
    S1* ss = new S1[2];
    delete[] ss;  // TODO LLVM does not call _ZdaPv here, but in destructor @_ZN2S1D0Ev
  } catch (...) {
  }

  return 0;
}

// CHECK: @_ZN2S1D0Ev
// CHECK: call void @_ZdlPv(i8* [[POINTER2:%[0-9]+]])
// CHECK-NEXT: call void @__typeart_free(i8* [[POINTER2]])

// CHECK: Malloc{{[ ]*}}:{{[ ]*}}2
// CHECK: Free{{[ ]*}}:{{[ ]*}}3
// CHECK: Alloca{{[ ]*}}:{{[ ]*}}0
