// RUN: %cuda-c-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s

// REQUIRES: cuda_static

// CHECK: call i32 @cudaHostAlloc({{(ptr|i8\*)}} {{.*}}[[CU_POINTER:%[_0-9a-z]+]],
// CHECK-NEXT: [[CUDA_PTR:%[0-9a-z_]+]] = load {{.*}}, {{.*}}[[CU_POINTER]]
// CHECK-NEXT: call void @__typeart_alloc_cuda({{(ptr|i8\*)}} {{.*}}[[CUDA_PTR]],

int main() {
  const int N = 20;
  float* d_x;

  cudaHostAlloc((void**)&d_x, N * sizeof(float), cudaHostAllocDefault);

  return 0;
}
