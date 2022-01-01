// clang-format off
// RUN: %cpp-to-llvm %omp_cpp_flags %s | %apply-typeart -S 2>&1 | %filecheck %s
// REQUIRES: openmp
// clang-format on

void foo(int** x) {
#pragma omp parallel  // transformed to @__kmpc_fork_call
  {
    double* pd = new double[10];
    delete[] pd;
  }

#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
    x[i] = new int[8];
  }
}

// CHECK: call void @__typeart_alloc_omp
// CHECK: call void @__typeart_free_omp
// CHECK: call void @__typeart_alloc_omp

// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}2
// CHECK-NEXT: Free{{[ ]*}}:{{[ ]*}}1
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0