// RUN: %hip-cpp-to-llvm %s | TYPEART_GPU=1 %apply-typeart -S 2>&1 | %filecheck %s

// REQUIRES: hip_static && !llvm-14

// CHECK: call i32 @hipFree(ptr {{.*}}[[HIP_POINTER:%[0-9a-z]+]])
// CHECK-NEXT: __typeart_free_gpu(ptr {{.*}}[[HIP_POINTER]])

#include <hip/hip_runtime.h>
int main() {
  float* d_x;

  hipFree(d_x);

  return 0;
}
