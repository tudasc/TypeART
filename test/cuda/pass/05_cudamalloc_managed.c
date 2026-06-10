// RUN: %cuda-c-to-llvm %s | TYPEART_GPU=1 %apply-typeart -S 2>&1 | %filecheck %s --check-prefix=%llvm-version-check

// REQUIRES: cuda

// LLVM: [[RET:%[0-9a-z_]+]] = call i32 @cudaMallocManaged(ptr {{.*}}[[CU_POINTER:%[_0-9a-z]+]],
// LLVM-NEXT: [[SUCCESS:%[0-9a-z_]+]] = icmp eq i32 [[RET]], 0
// LLVM-NEXT: br i1 [[SUCCESS]], label %[[LABEL:[0-9a-z_.]+]], label %[[SKIP:[0-9a-z_.]+]]
// LLVM: [[LABEL]]:
// LLVM-NEXT: [[CUDA_PTR:%[0-9a-z_]+]] = load {{.*}}, {{.*}}[[CU_POINTER]]
// LLVM-NEXT: call void @__typeart_alloc_gpu(ptr {{.*}}[[CUDA_PTR]], i32 23, i64 20)

// LLVM_LEGACY: [[RET:%[0-9a-z_]+]] = call i32 @cudaMallocManaged(i8** {{.*}}[[CAST1:%[0-9a-z_]+]],
// LLVM_LEGACY-NEXT: [[SUCCESS:%[0-9a-z_]+]] = icmp eq i32 [[RET]], 0
// LLVM_LEGACY-NEXT: br i1 [[SUCCESS]], label %[[LABEL:[0-9a-z_.]+]], label %[[SKIP:[0-9a-z_.]+]]
// LLVM_LEGACY: [[LABEL]]:
// LLVM_LEGACY-NEXT: [[CAST2:%[0-9a-z_]+]] = bitcast float** [[SRC_VAR:%[0-9a-zA-Z_]+]] to i8**
// LLVM_LEGACY-NEXT: [[LOADED_PTR:%[0-9a-z_]+]] = load i8*, i8** [[CAST2]]
// LLVM_LEGACY-NEXT: call void @__typeart_alloc_gpu(i8* [[LOADED_PTR]], i32 23, i64 20)

int main() {
  const int N = 20;
  float* d_x;

  cudaMallocManaged((void**)&d_x, N * sizeof(float));

  return 0;
}
