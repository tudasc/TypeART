// RUN: %hip-cpp-to-llvm %s | TYPEART_GPU=1 %apply-typeart -S 2>&1 | %filecheck %s

// REQUIRES: hip && !llvm-14

// clang-format off
// CHECK: [[RET_X:%[0-9a-z_]+]] = call i32 @hipMallocAsync(ptr {{.*}}[[HIP_POINTER_X:%[_0-9a-z]+]], i64{{.*}} 80, ptr {{.*}})
// CHECK-NEXT: [[SUCCESS_X:%[0-9a-z_]+]] = icmp eq i32 [[RET_X]], 0
// CHECK-NEXT: br i1 [[SUCCESS_X]], label %[[LABEL_X:[0-9a-z_.]+]], label %[[SKIP_X:[0-9a-z_.]+]]
// CHECK: [[LABEL_X]]:
// CHECK-NEXT: [[HIP_PTR_X:%[0-9a-z_]+]] = load ptr, ptr [[HIP_POINTER_X]]
// CHECK-NEXT: call void @__typeart_alloc_gpu(ptr [[HIP_PTR_X]], i32 23, i64 20)

// CHECK: [[RET_Y:%[0-9a-z_]+]] = call i32 @hipMallocFromPoolAsync(ptr {{.*}}[[HIP_POINTER_Y:%[_0-9a-z]+]], i64{{.*}} 80, ptr {{.*}}, ptr {{.*}})
// CHECK-NEXT: [[SUCCESS_Y:%[0-9a-z_]+]] = icmp eq i32 [[RET_Y]], 0
// CHECK-NEXT: br i1 [[SUCCESS_Y]], label %[[LABEL_Y:[0-9a-z_.]+]], label %[[SKIP_Y:[0-9a-z_.]+]]
// CHECK: [[LABEL_Y]]:
// CHECK-NEXT: [[HIP_PTR_Y:%[0-9a-z_]+]] = load ptr, ptr [[HIP_POINTER_Y]]
// CHECK-NEXT: call void @__typeart_alloc_gpu(ptr [[HIP_PTR_Y]], i32 23, i64 20)
// clang-format on

#include <hip/hip_runtime.h>
int main() {
  const int N = 20;
  float* d_x;
  float* d_y;

  hipStream_t stream;
  hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);

  hipMallocAsync((void**)&d_x, N * sizeof(float), stream);

  hipMemPool_t pool;
  hipMallocFromPoolAsync((void**)&d_y, N * sizeof(float), pool, stream);

  return 0;
}
