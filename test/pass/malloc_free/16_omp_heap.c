// clang-format off
// RUN: %c-to-llvm %omp_c_flags %s | %apply-typeart -S 2>&1 | FileCheck %s
// REQUIRES: openmp
// clang-format on

#include <stdlib.h>

void foo(int** x) {
#pragma omp parallel  // transformed to @__kmpc_fork_call
  {
    double* pd = calloc(10, sizeof(double));
    pd         = realloc(pd, 20 * sizeof(double));
  }

#pragma omp parallel for
  for (int i = 0; i < 10; ++i) {
    x[i] = (int*)malloc(8 * sizeof(int));
    free(x[i]);
  }
}

// CHECK: [[POINTER:%[0-9]+]] = call noalias i8* @calloc(i64 [[SIZE:[0-9]+]], i64 8)
// CHECK-NEXT: call void @__typeart_alloc_omp(i8* [[POINTER]], i32 6, i64 [[SIZE]])
// CHECK-NEXT: bitcast i8* [[POINTER]] to double*

// CHECK: __typeart_free_omp(i8* [[POINTER:%[0-9]+]])
// CHECK-NEXT: [[POINTER2:%[0-9]+]] = call i8* @realloc(i8* [[POINTER]], i64 160)
// CHECK-NEXT: __typeart_alloc_omp(i8* [[POINTER2]], i32 6, i64 20)

// CHECK: [[POINTER:%[0-9]+]] = call noalias i8* @malloc
// CHECK-NEXT: call void @__typeart_alloc_omp(i8* [[POINTER]], i32 2, i64 8)
// CHECK-NEXT: bitcast i8* [[POINTER]] to i32*

// CHECK: call void @free
// CHECK-NEXT: call void @__typeart_free_omp

// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}3
// CHECK-NEXT: Free{{[ ]*}}:{{[ ]*}}1
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0