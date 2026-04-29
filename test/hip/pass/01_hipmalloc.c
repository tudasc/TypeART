// RUN: %hip-cpp-to-llvm %s | TYPEART_GPU=1 %apply-typeart -S 2>&1 | %filecheck %s --check-prefix=%llvm-version-check

// REQUIRES: hip && !llvm-14

// LLVM: call i32 @hipMalloc(ptr {{.*}}[[HIP_POINTER:%[_0-9a-z]+]],
// LLVM: [[HIP_LOAD:%[0-9a-z_]+]] = load ptr, ptr [[HIP_POINTER]]
// LLVM: call void @__typeart_alloc_gpu(ptr [[HIP_LOAD]], i32 23

#include <hip/hip_runtime.h>
int main() {
  const int N = 20;
  float* d_x;

  hipMalloc(&d_x, N * sizeof(float));

  return 0;
}
