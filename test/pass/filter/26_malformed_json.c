// clang-format off
// RUN: %c-to-llvm %s | %not-apply-typeart --typeart-stack=true --typeart-filter=true --typeart-filter-implementation=acg --typeart-filter-cg-file=%p/26_malformed_cg.mcg 2>&1 | %filecheck %s
// clang-format on

// REQUIRES: not

// comma is missing between nodes
// CHECK: Fatal

extern void bar(int* ptr);

void foo() {
  int a;
  bar(&a);
}
