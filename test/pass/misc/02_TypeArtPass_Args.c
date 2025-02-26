// clang-format off
// Sanity check for arg names
// RUN: %c-to-llvm %s | %opt -load-pass-plugin %transform_pass -passes='typeart<stats;heap;stack;no-global>'
// clang-format on

void foo() {
  int a    = 1;
  double b = 2.0;
}