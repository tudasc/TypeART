// RUN: %cuda-c-to-llvm %s | TYPEART_GPU=1 %apply-typeart -S 2>&1 | %filecheck %s --check-prefix=%llvm-version-check

// REQUIRES: cuda

// clang-format off
// LLVM: [[RET_X:%[0-9a-z_]+]] = call i32 @cudaMallocAsync(ptr {{.*}}[[CU_POINTER_X:%[_0-9a-z]+]], i64 {{.*}}80, ptr {{.*}})
// LLVM-NEXT: [[SUCCESS_X:%[0-9a-z_]+]] = icmp eq i32 [[RET_X]], 0
// LLVM-NEXT: br i1 [[SUCCESS_X]], label %[[LABEL_X:[0-9a-z_.]+]], label %[[SKIP_X:[0-9a-z_.]+]]
// LLVM: [[LABEL_X]]:
// LLVM-NEXT: [[CUDA_PTR_X:%[0-9a-z_]+]] = load ptr, ptr [[CU_POINTER_X]]
// LLVM-NEXT: call void @__typeart_alloc_gpu(ptr [[CUDA_PTR_X]], i32 23, i64 20)

// LLVM: [[RET_Y:%[0-9a-z_]+]] = call i32 @cudaMallocFromPoolAsync(ptr {{.*}}[[CU_POINTER_Y:%[_0-9a-z]+]], i64 {{.*}}80, ptr {{.*}}, ptr {{.*}})
// LLVM-NEXT: [[SUCCESS_Y:%[0-9a-z_]+]] = icmp eq i32 [[RET_Y]], 0
// LLVM-NEXT: br i1 [[SUCCESS_Y]], label %[[LABEL_Y:[0-9a-z_.]+]], label %[[SKIP_Y:[0-9a-z_.]+]]
// LLVM: [[LABEL_Y]]:
// LLVM-NEXT: [[CUDA_PTR_Y:%[0-9a-z_]+]] = load ptr, ptr [[CU_POINTER_Y]]
// LLVM-NEXT: call void @__typeart_alloc_gpu(ptr [[CUDA_PTR_Y]], i32 23, i64 20)

// LLVM_LEGACY: [[RET_X:%[0-9a-z_]+]] = call i32 @cudaMallocAsync(i8** {{.*}}[[CU_POINTER_X:%[_0-9a-z]+]], i64 {{.*}}80, 
// LLVM_LEGACY-NEXT: [[SUCCESS_X:%[0-9a-z_]+]] = icmp eq i32 [[RET_X]], 0
// LLVM_LEGACY-NEXT: br i1 [[SUCCESS_X]], label %[[LABEL_X:[0-9a-z_.]+]], label %[[SKIP_X:[0-9a-z_.]+]]
// LLVM_LEGACY: [[LABEL_X]]:
// LLVM_LEGACY-NEXT: [[CAST2_X:%[0-9a-z_]+]] = bitcast float** [[SRC_VAR_X:%[0-9a-zA-Z_]+]] to i8**
// LLVM_LEGACY-NEXT: [[LOADED_PTR_X:%[0-9a-z_]+]] = load i8*, i8** [[CAST2_X]]
// LLVM_LEGACY-NEXT: call void @__typeart_alloc_gpu(i8* [[LOADED_PTR_X]], i32 23, i64 20)

// LLVM_LEGACY: [[RET_Y:%[0-9a-z_]+]] = call i32 @cudaMallocFromPoolAsync(i8** {{.*}}[[CU_POINTER_Y:%[_0-9a-z]+]], i64 {{.*}}80,
// LLVM_LEGACY-NEXT: [[SUCCESS_Y:%[0-9a-z_]+]] = icmp eq i32 [[RET_Y]], 0
// LLVM_LEGACY-NEXT: br i1 [[SUCCESS_Y]], label %[[LABEL_Y:[0-9a-z_.]+]], label %[[SKIP_Y:[0-9a-z_.]+]]
// LLVM_LEGACY: [[LABEL_Y]]:
// LLVM_LEGACY-NEXT: [[CAST2_Y:%[0-9a-z_]+]] = bitcast float** [[SRC_VAR_Y:%[0-9a-zA-Z_]+]] to i8**
// LLVM_LEGACY-NEXT: [[LOADED_PTR_Y:%[0-9a-z_]+]] = load i8*, i8** [[CAST2_Y]]
// LLVM_LEGACY-NEXT: call void @__typeart_alloc_gpu(i8* [[LOADED_PTR_Y]], i32 23, i64 20)
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
