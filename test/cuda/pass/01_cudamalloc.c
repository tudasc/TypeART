// RUN: %cuda-c-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s

// REQUIRES: cuda_static

// CHECK: call i32 @cudaMalloc
// CHECK-NEXT: [[CUDA_PTR:%[0-9a-z]+]] = load {{.*}}, {{.*}}
// CHECK-NEXT: call void @__typeart_alloc_cuda({{(ptr|i8\*)}} {{.*}}[[CUDA_PTR]],

int main() {
  const int N = 20;
  float* d_x;

  cudaMalloc((void**)&d_x, N * sizeof(float));

  return 0;
}
