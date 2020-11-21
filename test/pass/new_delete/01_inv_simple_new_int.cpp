// clang-format off
// RUN: %cpp-to-llvm %s | opt -load %pluginpath/analysis/meminstfinderpass.so -load %pluginpath/%pluginname %pluginargs -S 2>&1 | FileCheck %s
// clang-format on

#include <new>
// CHECK: invoke i8* @_Znwm(i64 4)
// CHECK: call void @__typeart_alloc(i8* [[POINTER:%[0-9]+]], i32 2, i64 1)
// CHECK-NEXT: bitcast i8* {{.*}}[[POINTER]] to i32*
int main() {
  try {
    auto s = new int;
  } catch (...) {
  }

  return 0;
}

// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK-NEXT: Free
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0
