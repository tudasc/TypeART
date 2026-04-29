// RUN: %cuda-c-to-llvm %s | TYPEART_GPU=1 %apply-typeart -S 2>&1 | %filecheck %s

// REQUIRES: cuda

// CHECK: call i32 @cudaFree({{(ptr|i8\*)}} {{.*}}[[CU_POINTER:%[0-9a-z]+]])
// CHECK-NEXT: __typeart_free_gpu({{(ptr|i8\*)}} {{.*}}[[CU_POINTER]])

// CHECK: call i32 @cudaFreeHost({{(ptr|i8\*)}} {{.*}}[[CU_POINTER:%[0-9a-z]+]])
// CHECK-NEXT: __typeart_free_gpu({{(ptr|i8\*)}} {{.*}}[[CU_POINTER]])

// CHECK: call i32 @cudaFreeAsync({{(ptr|i8\*)}} {{.*}}[[CU_POINTER:%[0-9a-z]+]],
// CHECK-NEXT: __typeart_free_gpu({{(ptr|i8\*)}} {{.*}}[[CU_POINTER]])

int main() {
  float* d_x;

  cudaFree(d_x);

  cudaFreeHost(d_x);

  cudaStream_t stream;
  cudaFreeAsync(d_x, stream);

  return 0;
}
