// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -S 2>&1 | FileCheck %s
// clang-format on
void test(int n) {
  int a[n];
  int b[n][n];
  int c[5][n];
}

// CHECK-NOT: Encountered unhandled type
// CHECK: Malloc{{[ ]*}}:{{[ ]*}}0
// CHECK: Free{{[ ]*}}:{{[ ]*}}0
// CHECK: Alloca{{[ ]*}}:{{[ ]*}}3
