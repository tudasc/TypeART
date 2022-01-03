// clang-format off
// Sanity check for arg names
// RUN: %c-to-llvm %s | %opt -load %transform_pass -typeart \
// RUN: -typeart-stats \
// RUN: -typeart-no-heap=false \
// RUN: -typeart-alloca=true \
// RUN: -typeart-outfile=typeart_types.yaml
// clang-format on

void foo() {
  int a    = 1;
  double b = 2.0;
}