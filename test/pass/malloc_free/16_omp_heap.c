// clang-format off
// RUN: %c-to-llvm %omp_c_flags %s | %apply-typeart -S 2>&1 | %filecheck %s --check-prefixes CHECK,%llvm-version-check
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
// clang-format off

// CHECK: TypeArtPass [Heap]
// CHECK-NEXT: Malloc{{[ ]*}}:{{[ ]*}}3
// CHECK-NEXT: Free{{[ ]*}}:{{[ ]*}}1
// CHECK-NEXT: Alloca{{[ ]*}}:{{[ ]*}}0

// LLVM_LEGACY: [[POINTER:%[0-9a-z]+]] = call noalias{{( align [0-9]+)?}} i8* @calloc(i64{{( noundef)?}} [[SIZE:[0-9a-z]+]], i64{{( noundef)?}} 8)
// LLVM_LEGACY-NEXT: call void @__typeart_alloc_omp(i8* [[POINTER]], i32 24, i64 [[SIZE]])
// LLVM_LEGACY-NEXT: bitcast i8* [[POINTER]] to double*
// LLVM: [[POINTER:%[0-9a-z]+]] = call noalias{{( align [0-9]+)?}} ptr @calloc(i64{{( noundef)?}} [[SIZE:[0-9a-z]+]], i64{{( noundef)?}} 8)
// LLVM-NEXT: call void @__typeart_alloc(ptr [[POINTER]], i32 24, i64 [[SIZE]])

// LLVM_LEGACY: __typeart_free_omp(i8* [[POINTER:%[0-9a-z]+]])
// LLVM_LEGACY-NEXT: [[POINTER2:%[0-9a-z]+]] = call{{( align [0-9]+)?}} i8* @realloc(i8*{{( noundef)?}} [[POINTER]], i64{{( noundef)?}} 160)
// LLVM_LEGACY-NEXT: __typeart_alloc_omp(i8* [[POINTER2]], i32 24, i64 20)
// LLVM: __typeart_free(ptr [[POINTER:%[0-9a-z]+]])
// LLVM-NEXT: [[POINTER2:%[0-9a-z]+]] = call{{( align [0-9]+)?}} ptr @realloc(ptr{{( noundef)?}} [[POINTER]], i64{{( noundef)?}} 160)
// LLVM-NEXT: __typeart_alloc(ptr [[POINTER2]], i32 24, i64 20)

// LLVM_LEGACY: [[POINTER:%[0-9a-z]+]] = call noalias{{( align [0-9]+)?}} i8* @malloc
// LLVM_LEGACY-NEXT: call void @__typeart_alloc_omp(i8* [[POINTER]], i32 13, i64 8)
// LLVM_LEGACY-NEXT: bitcast i8* [[POINTER]] to i32*
// LLVM: [[POINTER:%[0-9a-z]+]] = call noalias{{( align [0-9]+)?}} ptr @malloc
// LLVM-NEXT: call void @__typeart_alloc(ptr [[POINTER]], i32 13, i64 8)

// CHECK: call void @free
// LLVM_LEGACY-NEXT: call void @__typeart_free_omp
// LLVM-NEXT: call void @__typeart_free

// clang-format on
