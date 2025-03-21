// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart --typeart-stack=true -S 2>&1 | %filecheck %s
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
// CHECK: Alloca{{[ ]*}}:{{[ ]*}}6

// CHECK: call void @__typeart_alloc_stack({{i8\*|ptr}} %{{[0-9a-z]+}}, i32 13, i64 64)
// CHECK: call void @__typeart_alloc_stack({{i8\*|ptr}} %{{[0-9a-z]+}}, i32 13, i64 4096)
// CHECK: call void @__typeart_alloc_stack({{i8\*|ptr}} %{{[0-9a-z]+}}, i32 13, i64 8192)
// CHECK: call void @__typeart_alloc_stack({{i8\*|ptr}} %{{[0-9a-z]+}}, i32 13, i64 524288)
