// clang-format off
// RUN: clang++ -S -emit-llvm %s -o - | opt -load %pluginpath/analysis/meminstfinderpass.so -load %pluginpath/%pluginname %pluginargs -S 2>&1 | FileCheck %s
// clang-format on

#include <new>

struct S1 {
  int x;
  virtual ~S1() = default;
};

// CHECK: invoke i8* @_Znam(i64 40)
// CHECK: call void @__typeart_alloc(i8* [[POINTER:%[0-9]+]], i32 {{2[5-9][0-9]}}, i64 2)
// CHECK: bitcast i8* [[POINTER]] to %struct.S1*
int main() {
  try {
    S1* ss = new S1[2];
  } catch (...) {
  }

  return 0;
}

// CHECK: Malloc{{[ ]*}}:{{[ ]*}}1
// CHECK: Alloca{{[ ]*}}:{{[ ]*}}0
