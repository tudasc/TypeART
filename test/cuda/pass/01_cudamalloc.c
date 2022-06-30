// RUN: %cuda-c-to-llvm %s | %apply-typeart -S 2>&1
// RUN: %cuda-c-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s

// REQUIRES: cuda

// CHECK: [[CU_POINTER:%[0-9a-z]+]] = bitcast float** [[ALLOCA_POINTER:%[0-9a-z]+]] to i8**
// CHECK-NEXT: call i32 @cudaMalloc(i8** [[CU_POINTER]], i64 80)
// CHECK-NEXT: [[TA_POINTER:%[0-9a-z]+]] = bitcast float** [[ALLOCA_POINTER]] to i8*
// CHECK-NEXT: __typeart_alloc(i8* [[TA_POINTER]], i32 5, i64 20)

int main() {
  const int N = 20;
  float* d_x;

  cudaMalloc((void**)&d_x, N * sizeof(float));

  return 0;
}