// clang-format off
// Sanity check for arg names
// RUN: %c-to-llvm %s | %opt -load %transform_pass -typeart \
// RUN: -typeart-call-filter \
// RUN: -typeart-call-filter-impl=none \
// RUN: -typeart-call-filter-str=MPI1 \
// RUN: -typeart-call-filter-deep-str=MPI2 \
// RUN: -typeart-call-filter-cg-file=/foo/file.cg \
// RUN: -typeart-stack-array-only=true \
// RUN: -typeart-malloc-store-filter=true \
// RUN: -typeart-filter-globals=true \
// RUN: -typeart-filter-pointer-alloca=false
// clang-format on

void foo() {
  int a    = 1;
  double b = 2.0;
}