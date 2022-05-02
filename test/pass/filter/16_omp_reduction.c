// clang-format off
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %apply-typeart -typeart-stack -typeart-call-filter -S 2>&1 | %filecheck %s
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %opt -O2 -S | %apply-typeart -typeart-stack -typeart-call-filter -S 2>&1 | %filecheck %s

// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %apply-typeart -typeart-stack -typeart-call-filter -S | %filecheck %s --check-prefix=check-inst
// RUN: %c-to-llvm -fno-discard-value-names %omp_c_flags %s | %opt -O2 -S | %apply-typeart -typeart-stack -typeart-call-filter -S | %filecheck %s --check-prefix=check-inst
// REQUIRES: openmp
// clang-format on

extern void MPI_send(void*);

float sum(const float* a, int n) {
  float total = 0.;
#pragma omp parallel for reduction(+ : total)
  for (int i = 0; i < n; i++) {
    total += a[i];
  }
  return total;
}

void foo() {
  const int n    = 10;
  float array[n] = {0};
  // check-inst: define {{.*}} @foo
  // check-inst: %loc = alloca
  // check-inst: [[POINTER:%[0-9a-z]+]] = bitcast float* %loc to i8*
  // check-inst: call void @__typeart_alloc_stack(i8* [[POINTER]], i32 5, i64 1)
  // check-inst-not: __typeart_alloc_stack_omp
  float loc      = sum(array, n);
  MPI_send((void*)&loc);
}

// CHECK: TypeArtPass [Heap & Stack]
// CHECK: Malloc :   0
// CHECK: Free   :   0
// CHECK: Alloca :   1
// CHECK: Global :   0
