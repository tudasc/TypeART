// clang-format off
// Sanity check for arg names
// RUN: %c-to-llvm %s | %opt -load-pass-plugin %transform_pass -passes='typeart<filter;filter-implementation=none;filter-glob=MPI1;analysis-filter-non-array-alloca;analysis-filter-heap-alloca;analysis-filter-global;analysis-filter-pointer-alloca;stack-lifetime;stats>'
// clang-format on

void foo() {
  int a    = 1;
  double b = 2.0;
}