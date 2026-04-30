// RUN: %cuda-c-to-llvm %s | %apply-typeart --typeart-gpu=false -S 2>&1 | %filecheck %s
// RUN: %cuda-c-to-llvm %s | TYPEART_GPU=false %apply-typeart -S 2>&1 | %filecheck %s

// REQUIRES: cuda

// CHECK: call i32 @{{.*}}({{(ptr|i8\*)}} {{.*}}[[CU_POINTER:%[_0-9a-z]+]],
// CHECK: call i32 @cudaFree({{(ptr|i8\*)}} {{.*}}[[CU_POINTER]])
// CHECK-NOT: call void @__typeart_alloc_gpu(
// CHECK-NOT: call void @__typeart_free_gpu(

int main() {
  const int N = 20;
  float* d_x;

  cudaMalloc((void**)&d_x, N * sizeof(float));
  cudaFree(d_x);

  return 0;
}
