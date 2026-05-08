// clang-format off
// RUN: %wrapper-cc -x hip --offload-host-only -S -emit-llvm -O1 --typeart-gpu=true %s -o - 2>&1 | %filecheck %s --check-prefix=GPU-ON
// RUN: %wrapper-cc -x hip --offload-host-only -S -emit-llvm -O1 --typeart-gpu=false %s -o - 2>&1 | %filecheck %s --check-prefix=GPU-OFF
// RUN: %wrapper-cc -x hip --offload-host-only -S -emit-llvm -O1 --typeart-gpu --typeart-gpu=true %s -o - 2>&1 | %filecheck %s --check-prefix=GPU-ON
// RUN: %wrapper-cc -x hip --offload-host-only -S -emit-llvm -O1 --typeart-gpu %s -o - 2>&1 | %filecheck %s --check-prefix=GPU-ON
// RUN: TYPEART_WRAPPER=OFF %wrapper-cc -x hip --offload-host-only -S -emit-llvm -O1 --typeart-gpu=true --typeart-gpu %s -o - 2>&1 | %filecheck %s --check-prefix=WRAPPER-OFF
// clang-format on

// REQUIRES: hip && !llvm-14

// GPU-ON: call i32 @hipMalloc(ptr {{.*}}[[HIP_POINTER:%[_0-9a-z]+]],
// GPU-ON: call void @__typeart_alloc_gpu(
// GPU-ON: call i32 @hipFree(ptr {{.*}}[[HIP_PTR:%[_0-9a-z]+]])
// GPU-ON: call void @__typeart_free_gpu(

// GPU-OFF: call i32 @hipMalloc(ptr {{.*}}[[HIP_POINTER:%[_0-9a-z]+]],
// GPU-OFF: call i32 @hipFree(ptr {{.*}}[[HIP_PTR:%[_0-9a-z]+]])
// GPU-OFF-NOT: __typeart_alloc_gpu(
// GPU-OFF-NOT: __typeart_free_gpu(

// WRAPPER-OFF: call i32 @hipMalloc(ptr {{.*}}[[HIP_POINTER:%[_0-9a-z]+]],
// WRAPPER-OFF: call i32 @hipFree(ptr {{.*}}[[HIP_PTR:%[_0-9a-z]+]])
// WRAPPER-OFF-NOT: __typeart_alloc_gpu(
// WRAPPER-OFF-NOT: __typeart_free_gpu(

#include <hip/hip_runtime.h>

int main() {
  const int N = 20;
  float* d_x;

  hipMalloc((void**)&d_x, N * sizeof(float));
  hipFree(d_x);

  return 0;
}
