// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -S 2>&1 | FileCheck %s
// clang-format on
void test() {
  int a[100];
}

// CHECK: Malloc{{[ ]*}}:{{[ ]*}}0
// CHECK: Free{{[ ]*}}:{{[ ]*}}0
// CHECK: Alloca{{[ ]*}}:{{[ ]*}}1
