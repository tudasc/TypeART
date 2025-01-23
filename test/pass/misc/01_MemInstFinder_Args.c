// clang-format off
// Sanity check for arg names
// RUN: %c-to-llvm %s | %opt -load %transform_pass -typeart \
// RUN: --typeart-filter=true \
// RUN: --typeart-filter-implementation=none \
// RUN: --typeart-filter-glob=MPI1 \
// RUN: --typeart-filter-glob-deep=MPI2 \
// RUN: --typeart-filter-cg-file=/foo/file.cg \
// RUN: --typeart-analysis-filter-non-array-alloca=true \
// RUN: --typeart-analysis-filter-heap-alloca=true \
// RUN: --typeart-analysis-filter-global=true \
// RUN: --typeart-analysis-filter-pointer-alloca=false \
// RUN: --typeart-stack-lifetime=false \
// RUN: --typeart-stats=true \
// RUN: --typeart-types=typeart_types_args.yaml \
// RUN: --typeart-config=07_typeart_config_stack.yml
// clang-format on

void foo() {
  int a    = 1;
  double b = 2.0;
}