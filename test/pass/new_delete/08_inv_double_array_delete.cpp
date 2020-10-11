// clang-format off
// RUN: clang++ -S -emit-llvm %s -o - | opt -load %pluginpath/analysis/meminstfinderpass.so -load %pluginpath/%pluginname %pluginargs -S 2>&1 | FileCheck %s
// clang-format on

#include <new>
// CHECK: invoke i8* @_Znam(i64 16)
// CHECK: call void @__typeart_alloc(i8* [[POINTER:%[0-9]+]], i32 6, i64 2)
// CHECK-NEXT: bitcast i8* {{.*}}[[POINTER]] to double*
// CHECK: call void @_ZdaPv(i8* [[POINTER2:%[0-9]+]])
// CHECK-NEXT: call void @__typeart_free(i8* {{.*}}[[POINTER2]])
int main() {
  try {
    auto s = new double[2];
    delete[] s;
  } catch (...) {
  }

  return 0;
}

// CHECK: invoke i8* @_Znam(i64 16)
// CHECK: call void @__typeart_alloc(i8* [[POINTER:%[0-9]+]], i32 6, i64 2)
// CHECK-NEXT: bitcast i8* {{.*}}[[POINTER]] to double*
// CHECK: call void @_ZdaPv(i8* [[POINTER2:%[0-9]+]])
// CHECK-NEXT: call void @__typeart_free(i8* {{.*}}[[POINTER2]])
void foo() {
  double* b{nullptr};
  try {
    b = new double[2];
  } catch (...) {
  }
  if (b != nullptr) {
    delete[] b;
  }
}

// CHECK: Malloc{{[ ]*}}:{{[ ]*}}2
// CHECK: Free{{[ ]*}}:{{[ ]*}}2
// CHECK: Alloca{{[ ]*}}:{{[ ]*}}0
