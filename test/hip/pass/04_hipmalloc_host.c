// RUN: %hip-cpp-to-llvm %s | TYPEART_GPU=1 %apply-typeart -S 2>&1 | %filecheck %s

// REQUIRES: hip && !llvm-14

// CHECK: call i32 @hipMallocHost(ptr {{.*}}[[HIP_POINTER:%[_0-9a-z]+]],
// CHECK-NEXT: [[HIP_PTR:%[0-9a-z_]+]] = load ptr, ptr [[HIP_POINTER]]
// CHECK-NEXT: call void @__typeart_alloc_gpu(ptr [[HIP_PTR]], i32 23, i64 20)

#include <hip/hip_runtime.h>
int main() {
  const int N = 20;
  float* d_x;

  hipMallocHost((void**)&d_x, N * sizeof(float));

  return 0;
}
