// clang-format off
// RUN: %c-to-llvm %s | TYPEART_OPTIONS="no-heap;no-stack;no-stats;no-stack-lifetime" TYPEART_HEAP=true %apply-typeart 2>&1 | %filecheck %s

// CHECK: Emitting TypeART configuration content
// CHECK-NEXT: ---
// CHECK-NEXT: types:           {{.*}}
// CHECK-NEXT: heap:            true
// CHECK-NEXT: stack:           false
// CHECK-NEXT: global:          false
// CHECK-NEXT: stats:          false
// CHECK-NEXT: stack-lifetime:          false

// clang-format on

void test() {
  int a;
}
