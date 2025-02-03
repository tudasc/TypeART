// clang-format off
// RUN: %c-to-llvm %s | TYPEART_OPTIONS_HEAP="stack;no-stats;no-stack-lifetime" %apply-typeart 2>&1 | %filecheck %s

// CHECK: Emitting TypeART configuration content
// CHECK-NEXT: ---
// CHECK-NEXT: types:           {{.*}}
// CHECK-NEXT: heap:            true
// CHECK-NEXT: stack:           true
// CHECK-NEXT: global:          true
// CHECK-NEXT: stats:          false
// CHECK-NEXT: stack-lifetime:          false

// clang-format on

void test() {
  int a;
}
