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
int main() {
  try {
    S1* ss = new S1;
  } catch (...) {
  }

  return 0;
}

// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK-NEXT: Free
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0
