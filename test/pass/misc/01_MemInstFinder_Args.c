// clang-format off
// Sanity check for arg names
// RUN: %c-to-llvm %s | opt -load %analysis_pass -mem-inst-finder \
// RUN: -call-filter \
// RUN: -call-filter-impl=empty \
// RUN: -call-filter-str=MPI1 \
// RUN: -call-filter-deep-str=MPI2 \
// RUN: -call-filter-cg-file=/foo/file.cg \
// RUN: -call-filter-deep=true \
// RUN: -alloca-array-only=true \
// RUN: -malloc-store-filter=true \
// RUN: -filter-globals=true
// clang-format on

void foo() {
  int a    = 1;
  double b = 2.0;
}