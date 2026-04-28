// RUN: %cuda-c-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s --check-prefix=%llvm-version-check

// REQUIRES: cuda_static

// clang-format off
// LLVM: call i32 @cudaMallocAsync(ptr {{.*}}[[CU_POINTER_X:%[_0-9a-z]+]], i64{{.*}} 80, ptr {{.*}})
// LLVM-NEXT: [[CUDA_PTR_X:%[0-9a-z_]+]] = load ptr, ptr [[CU_POINTER_X]]
// LLVM-NEXT: call void @__typeart_alloc_gpu(ptr [[CUDA_PTR_X]], i32 23, i64 20)

// LLVM: call i32 @cudaMallocFromPoolAsync(ptr {{.*}}[[CU_POINTER_Y:%[_0-9a-z]+]], i64{{.*}} 80, ptr {{.*}}, ptr {{.*}})
// LLVM-NEXT: [[CUDA_PTR_Y:%[0-9a-z_]+]] = load ptr, ptr [[CU_POINTER_Y]]
// LLVM-NEXT: call void @__typeart_alloc_gpu(ptr [[CUDA_PTR_Y]], i32 23, i64 20)

// LLVM_LEGACY: [[CAST1:%[0-9a-z_]+]] = bitcast float** [[SRC_VAR:%[0-9a-zA-Z_]+]] to i8**
// LLVM_LEGACY: call i32 @cudaMallocAsync(i8** {{.*}}[[CAST1]],
// LLVM_LEGACY: [[CAST2:%[0-9a-z_]+]] = bitcast float** [[SRC_VAR]] to i8**
// LLVM_LEGACY: [[LOADED_PTR:%[0-9a-z_]+]] = load i8*, i8** [[CAST2]]
// LLVM_LEGACY: call void @__typeart_alloc_gpu(i8* [[LOADED_PTR]], i32 23, i64 20)

// LLVM_LEGACY: [[CAST1:%[0-9a-z_]+]] = bitcast float** [[SRC_VAR:%[0-9a-zA-Z_]+]] to i8**
// LLVM_LEGACY: call i32 @cudaMallocFromPoolAsync(i8** {{.*}}[[CAST1]],
// LLVM_LEGACY: [[CAST2:%[0-9a-z_]+]] = bitcast float** [[SRC_VAR]] to i8**
// LLVM_LEGACY: [[LOADED_PTR:%[0-9a-z_]+]] = load i8*, i8** [[CAST2]]
// LLVM_LEGACY: call void @__typeart_alloc_gpu(i8* [[LOADED_PTR]], i32 23, i64 20)
// clang-format on

int main() {
  const int N = 20;
  float* d_x;
  float* d_y;

  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  cudaMallocAsync((void**)&d_x, N * sizeof(float), stream);

  cudaMemPool_t pool;
  cudaMallocFromPoolAsync((void**)&d_y, N * sizeof(float), pool, stream);

  return 0;
}