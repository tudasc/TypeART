// clang-format off
// RUN: %cpp-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s
// clang-format on

// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}2
// CHECK-NEXT: Free{{[ ]*}}:{{[ ]*}}2
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0

#include <new>
// CHECK: invoke{{.*}} {{i8\*|ptr}} @_Znam(i64{{( noundef)?}} 16)
// CHECK: call void @__typeart_alloc({{i8\*|ptr}} [[POINTER:%[0-9a-z]+]], i32 23, i64 2)
// CHECK: call void @_ZdaPv({{i8\*|ptr}}{{( noundef)?}} [[POINTER2:%[0-9a-z]+]])
// CHECK-NEXT: call void @__typeart_free({{i8\*|ptr}} {{.*}}[[POINTER2]])
int main() {
  try {
    auto s = new double[2];
    delete[] s;
  } catch (...) {
  }

  return 0;
}

// CHECK: invoke{{.*}} {{i8\*|ptr}} @_Znam(i64{{( noundef)?}} 16)
// CHECK: call void @__typeart_alloc({{i8\*|ptr}} [[POINTER:%[0-9a-z]+]], i32 23, i64 2)
// CHECK: call void @_ZdaPv({{i8\*|ptr}}{{( noundef)?}} [[POINTER2:%[0-9a-z]+]])
// CHECK-NEXT: call void @__typeart_free({{i8\*|ptr}} {{.*}}[[POINTER2]])
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
