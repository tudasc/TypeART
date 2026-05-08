// RUN: %hip-cpp-to-llvm %s | TYPEART_GPU=1 %apply-typeart -S 2>&1 | %filecheck %s

// REQUIRES: hip && !llvm-14

// CHECK: [[RET:%[0-9a-z_]+]] = call i32 @hipMalloc(ptr {{.*}}[[HIP_POINTER:%[_0-9a-z]+]],
// CHECK-NEXT: [[SUCCESS:%[0-9a-z_]+]] = icmp eq i32 [[RET]], 0
// CHECK-NEXT: br i1 [[SUCCESS]], label %[[LABEL:[0-9a-z_.]+]], label %[[SKIP:[0-9a-z_.]+]]
// CHECK: [[LABEL]]:
// CHECK-NEXT: [[HIP_PTR:%[0-9a-z_]+]] = load ptr, ptr [[HIP_POINTER]]
// CHECK-NEXT: call void @__typeart_alloc_gpu(ptr [[HIP_PTR]], i32 23, i64 20)

#include <hip/hip_runtime.h>
int main() {
  const int N = 20;
  float* d_x;

  hipMalloc((void**)&d_x, N * sizeof(float));

  return 0;
}
