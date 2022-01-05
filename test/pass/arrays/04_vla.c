// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -typeart-stack -typeart-stack-array-only=true -S 2>&1 | %filecheck %s
// clang-format on
void test(int n) {
  int a[n];
  int b[n][n];
  int c[5][n];
}

// CHECK-NOT: Encountered unhandled type
// CHECK: TypeArtPass [Heap & Stack]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}0
// CHECK-NEXT: Free{{[ ]*}}:{{[ ]*}}0
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}3
