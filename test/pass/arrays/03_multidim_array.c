// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -typeart-stack -S 2>&1 | %filecheck %s
// clang-format on
void test() {
  const int n = 64;
  const int m = 128;
  int a[n];
  int b[n][n];
  int c[n][m];
  int d[n][m][n];
}

// CHECK: call void @__typeart_alloc_stack(i8* %{{[0-9]+}}, i32 2, i64 64)
// CHECK: call void @__typeart_alloc_stack(i8* %{{[0-9]+}}, i32 2, i64 4096)
// CHECK: call void @__typeart_alloc_stack(i8* %{{[0-9]+}}, i32 2, i64 8192)
// CHECK: call void @__typeart_alloc_stack(i8* %{{[0-9]+}}, i32 2, i64 524288)

// CHECK-NOT: Encountered unhandled type
// CHECK: Malloc{{[ ]*}}:{{[ ]*}}0
// CHECK: Free{{[ ]*}}:{{[ ]*}}0
// CHECK: Alloca{{[ ]*}}:{{[ ]*}}6
