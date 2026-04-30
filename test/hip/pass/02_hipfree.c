// RUN: %hip-cpp-to-llvm %s | TYPEART_GPU=1 %apply-typeart -S 2>&1 | %filecheck %s

// REQUIRES: hip && !llvm-14

// CHECK: call i32 @hipFree(ptr {{.*}}[[HIP_POINTER:%[0-9a-z]+]])
// CHECK-NEXT: __typeart_free_gpu(ptr {{.*}}[[HIP_POINTER]])

// CHECK: call i32 @hipFreeHost(ptr {{.*}}[[HIP_POINTER:%[0-9a-z]+]])
// CHECK-NEXT: __typeart_free_gpu(ptr {{.*}}[[HIP_POINTER]])

// CHECK: call i32 @hipFreeAsync(ptr {{.*}}[[HIP_POINTER:%[0-9a-z]+]],
// CHECK-NEXT: __typeart_free_gpu(ptr {{.*}}[[HIP_POINTER]])

#include <hip/hip_runtime.h>
int main() {
  float* d_x;

  hipFree(d_x);

  hipFreeHost(d_x);

  hipStream_t stream;
  hipFreeAsync(d_x, stream);

  return 0;
}
