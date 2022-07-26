// RUN: %cuda-c-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s

// REQUIRES: cuda

// CHECK: call i32 @cudaFreeHost(i8* [[CU_POINTER:%[0-9a-z]+]])
// CHECK-NEXT: __typeart_free(i8* [[CU_POINTER]])

int main() {
  float* d_x;

  cudaFreeHost(d_x);

  return 0;
}