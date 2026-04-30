// RUN: %cuda-c-to-llvm %s | TYPEART_GPU=true %apply-typeart -S 2>&1 | %filecheck %s --check-prefix=%llvm-version-check

// REQUIRES: cuda

// clang-format off
// LLVM: call i32 @{{.*}}(ptr {{.*}}[[CU_POINTER_X:%[_0-9a-z]+]], i64 80, ptr {{.*}})
// LLVM-NEXT: [[CUDA_PTR_X:%[0-9a-z_]+]] = load ptr, ptr [[CU_POINTER_X]]
// LLVM-NEXT: call void @__typeart_alloc_gpu(ptr [[CUDA_PTR_X]], i32 23, i64 20)

// LLVM: call i32 @{{.*}}(ptr {{.*}}[[CU_POINTER_Y:%[_0-9a-z]+]], i64 80, ptr {{.*}}, ptr {{.*}})
// LLVM-NEXT: [[CUDA_PTR_Y:%[0-9a-z_]+]] = load ptr, ptr [[CU_POINTER_Y]]
// LLVM-NEXT: call void @__typeart_alloc_gpu(ptr [[CUDA_PTR_Y]], i32 23, i64 20)

// LLVM_LEGACY: call i32 @{{.*}}({{(ptr|i8\*\*)}} {{.*}}[[CU_POINTER_X:%[_0-9a-z]+]], i64 80, {{(ptr|i8\*)}} {{.*}})
// LLVM_LEGACY: [[CUDA_PTR_X:%[0-9a-z_]+]] = load i8*, i8** [[CU_POINTER_X]]
// LLVM_LEGACY: call void @__typeart_alloc_gpu(i8* [[CUDA_PTR_X]], i32 23, i64 20)

// LLVM_LEGACY: call i32 @{{.*}}({{(ptr|i8\*\*)}} {{.*}}[[CU_POINTER_Y:%[_0-9a-z]+]], i64 80, {{(ptr|i8\*)}} {{.*}}, {{(ptr|i8\*)}} {{.*}})
// LLVM_LEGACY: [[CUDA_PTR_Y:%[0-9a-z_]+]] = load i8*, i8** [[CU_POINTER_Y]]
// LLVM_LEGACY: call void @__typeart_alloc_gpu(i8* [[CUDA_PTR_Y]], i32 23, i64 20)
// clang-format on

int main() {
  const int N = 20;
  float* d_x;
  float* d_y;
  cudaStream_t stream = 0;
  cudaMemPool_t pool  = 0;

  cudaMallocAsync((void**)&d_x, N * sizeof(float), stream);
  cudaMallocFromPoolAsync((void**)&d_y, N * sizeof(float), pool, stream);

  return 0;
}
