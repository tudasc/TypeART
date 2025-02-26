// clang-format off
// RUN: %c-to-llvm %s | TYPEART_OPTIONS_STACK="global;stack;no-stats;no-stack-lifetime" %apply-typeart 2>&1 | %filecheck %s

// CHECK: Emitting TypeART configuration content
// CHECK: heap:            true
// CHECK-NOT: stack:           true
// CHECK-NOT: {{^}}global:          true
// CHECK-NOT: stats:          false
// CHECK-NOT: stack-lifetime:          false

// clang-format on

void test() {
  int a;
}
