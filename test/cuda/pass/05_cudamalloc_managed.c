// RUN: %cuda-c-to-llvm %s | TYPEART_GPU=1 %apply-typeart -S 2>&1 | %filecheck %s --check-prefix=%llvm-version-check

// REQUIRES: cuda_static

// LLVM: call i32 @cudaMallocManaged(ptr {{.*}}[[CU_POINTER:%[_0-9a-z]+]],
// LLVM-NEXT: [[CUDA_PTR:%[0-9a-z_]+]] = load {{.*}}, {{.*}}[[CU_POINTER]]
// LLVM-NEXT: call void @__typeart_alloc_gpu(ptr {{.*}}[[CUDA_PTR]], i32 23, i64 20)

// LLVM_LEGACY: [[CAST1:%[0-9a-z_]+]] = bitcast float** [[SRC_VAR:%[0-9a-zA-Z_]+]] to i8**
// LLVM_LEGACY: call i32 @cudaMallocManaged(i8** {{.*}}[[CAST1]],
// LLVM_LEGACY: [[CAST2:%[0-9a-z_]+]] = bitcast float** [[SRC_VAR]] to i8**
// LLVM_LEGACY: [[LOADED_PTR:%[0-9a-z_]+]] = load i8*, i8** [[CAST2]]
// LLVM_LEGACY: call void @__typeart_alloc_gpu(i8* [[LOADED_PTR]], i32 23, i64 20)

int main() {
  const int N = 20;
  float* d_x;

  cudaMallocManaged((void**)&d_x, N * sizeof(float));

  return 0;
}
