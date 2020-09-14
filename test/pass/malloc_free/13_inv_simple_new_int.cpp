// clang-format off
// RUN: clang++ -S -emit-llvm %s -o - | opt -load %pluginpath/analysis/meminstfinderpass.so -load %pluginpath/%pluginname %pluginargs -S 2>&1 | FileCheck %s
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

// CHECK: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK: Alloca{{[ ]*}}:{{[ ]*}}0
