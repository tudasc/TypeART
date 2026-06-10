// RUN: %hip-cpp-to-llvm %s | %apply-typeart --typeart-gpu=false -S 2>&1 | %filecheck %s
// RUN: %hip-cpp-to-llvm %s | TYPEART_GPU=0 %apply-typeart -S 2>&1 | %filecheck %s

// REQUIRES: hip && !llvm-14

// CHECK: call i32 @{{.*}}(ptr {{.*}}[[HIP_POINTER:%[_0-9a-z]+]],
// CHECK: [[HIP_PTR:%[_0-9a-z]+]] = load ptr, ptr [[HIP_POINTER]]
// CHECK: call i32 @hipFree(ptr {{.*}}[[HIP_PTR]])
// CHECK-NOT: call void @__typeart_alloc_gpu(
// CHECK-NOT: call void @__typeart_free_gpu(

#include <hip/hip_runtime.h>
int main() {
  const int N = 20;
  float* d_x;

  hipMalloc((void**)&d_x, N * sizeof(float));
  hipFree(d_x);

  return 0;
}
