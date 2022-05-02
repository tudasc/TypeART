// clang-format off
// RUN: %cpp-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s
// clang-format on

#include <new>

struct S1 {
  int x;
  virtual ~S1() = default;
};

// CHECK: invoke{{.*}} i8* @_Znam(i64{{( noundef)?}} 56)
// CHECK: call void @__typeart_alloc(i8* [[POINTER:%[0-9a-z]+]], i32 {{2[0-9]+}}, i64 3)
// CHECK: [[MEMORYBLOB:%[0-9a-z]+]] = getelementptr inbounds i8, i8* [[ARRPTR:%[0-9a-z]+]], i64 -8
// CHECK: call void @_ZdaPv(i8*{{( noundef)?}} [[MEMORYBLOB]])
// CHECK-NEXT: call void @__typeart_free(i8* [[ARRPTR]])
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

// CHECK: invoke{{.*}} i8* @_Znam(i64{{( noundef)?}} 40)
// CHECK: call void @__typeart_alloc(i8* [[POINTER:%[0-9a-z]+]], i32 {{2[0-9]+}}, i64 2)
// CHECK: [[MEMORYBLOB:%[0-9a-z]+]] = getelementptr inbounds i8, i8* [[ARRPTR:%[0-9a-z]+]], i64 -8
// CHECK: call void @_ZdaPv(i8*{{( noundef)?}} [[MEMORYBLOB]])
// CHECK-NEXT: call void @__typeart_free(i8* [[ARRPTR]])
int main() {
  try {
    S1* ss = new S1[2];
    delete[] ss;  // TODO LLVM does not call _ZdaPv here, but in destructor @_ZN2S1D0Ev
  } catch (...) {
  }

  return 0;
}

// CHECK: @_ZN2S1D0Ev
// CHECK: call void @_ZdlPv(i8*{{( noundef)?}} [[POINTER2:%[0-9a-z]+]])
// CHECK-NEXT: call void @__typeart_free(i8* [[POINTER2]])

// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}2
// CHECK-NEXT: Free{{[ ]*}}:{{[ ]*}}3
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0
