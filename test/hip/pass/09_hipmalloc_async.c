// RUN: %hip-cpp-to-llvm %s | TYPEART_GPU=1 %apply-typeart -S 2>&1 | %filecheck %s

// REQUIRES: hip && !llvm-14

// CHECK: call i32 @{{.*}}(ptr {{.*}}[[HIP_POINTER_X:%[_0-9a-z]+]], i64 {{.*}}80, ptr {{.*}})
// CHECK-NEXT: [[HIP_PTR_X:%[0-9a-z_]+]] = load ptr, ptr [[HIP_POINTER_X]]
// CHECK-NEXT: call void @__typeart_alloc_gpu(ptr [[HIP_PTR_X]], i32 23, i64 20)

// CHECK: call i32 @{{.*}}(ptr {{.*}}[[HIP_POINTER_Y:%[_0-9a-z]+]], i64 {{.*}}80, ptr {{.*}}, ptr {{.*}})
// CHECK-NEXT: [[HIP_PTR_Y:%[0-9a-z_]+]] = load ptr, ptr [[HIP_POINTER_Y]]
// CHECK-NEXT: call void @__typeart_alloc_gpu(ptr [[HIP_PTR_Y]], i32 23, i64 20)

#include <hip/hip_runtime.h>

int main() {
  const int N = 20;
  float* d_x;
  float* d_y;
  hipStream_t stream = 0;
  hipMemPool_t pool  = 0;

  hipMallocAsync((void**)&d_x, N * sizeof(float), stream);
  hipMallocFromPoolAsync((void**)&d_y, N * sizeof(float), pool, stream);

  return 0;
}
