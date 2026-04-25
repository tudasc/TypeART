// RUN: %cuda-c-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s

// REQUIRES: cuda_static

// CHECK: call i32 @cudaFree({{(ptr|i8\*)}} {{.*}}[[CU_POINTER:%[0-9a-z]+]])
// CHECK-NEXT: __typeart_free_cuda({{(ptr|i8\*)}} {{.*}}[[CU_POINTER]])

int main() {
  float* d_x;

  cudaFree(d_x);

  return 0;
}
