// clang-format off
// Sanity check for arg names
// RUN: %c-to-llvm %s | %opt -load %transform_pass -typeart \
// RUN: -typeart-stats \
// RUN: -typeart-heap=true \
// RUN: -typeart-stack=true \
// RUN: -typeart-global=false \
// RUN: -typeart-types=typeart_types.yaml
// clang-format on

void foo() {
  int a    = 1;
  double b = 2.0;
}