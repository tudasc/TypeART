// Template for recursion.ll.in
// clang-format off
// RUN: %c-to-llvm %s | %apply-typeart -typeart-alloca -call-filter -call-filter-impl=deprecated::default -S 2>&1 | %filecheck %s
// RUN: %c-to-llvm %s | %apply-typeart -typeart-alloca -call-filter -S 2>&1 | %filecheck %s
// clang-format on
void bar(int* x) {
}

void foo(int x) {
  bar(&x);
  if (x > 1) {
    foo(x);
  }
}

// CHECK: MemInstFinderPass
// CHECK: Stack call filtered %{{[ :]+}}100.0
