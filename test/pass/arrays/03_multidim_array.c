// clang-format off
// RUN: %c-to-llvm %s | opt -load %pluginpath/analysis/meminstfinderpass.so -load %pluginpath/%pluginname %pluginargs -S 2>&1 | FileCheck %s
// clang-format on
void test() {
  const int n = 64;
  const int m = 128;
  int a[n];
  int b[n][n];
  int c[n][m];
  int d[n][m][n];
}

// CHECK-NOT: Encountered unhandled type
// CHECK: Malloc{{[ ]*}}:{{[ ]*}}0
// CHECK: Free{{[ ]*}}:{{[ ]*}}0
// CHECK: Alloca{{[ ]*}}:{{[ ]*}}4
