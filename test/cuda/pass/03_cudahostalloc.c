// RUN: %cuda-c-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s

// REQUIRES: cuda

// CHECK: [[CU_POINTER:%[0-9a-z]+]] = bitcast float** [[ALLOCA_POINTER:%[0-9a-z]+]] to i8**
// CHECK-NEXT: call i32 @cudaHostAlloc(i8** [[CU_POINTER]], i64 80, i32 0)
// CHECK-NEXT: [[LOAD_POINTER:%[0-9a-z]+]] = load float*, float** [[ALLOCA_POINTER]]
// CHECK-NEXT: [[TA_POINTER:%[0-9a-z]+]] = bitcast float* [[LOAD_POINTER]] to i8*
// CHECK-NEXT: __typeart_alloc(i8* [[TA_POINTER]], i32 5, i64 20)

int main() {
  const int N = 20;
  float* d_x;

  cudaHostAlloc((void**)&d_x, N * sizeof(float), cudaHostAllocDefault);

  return 0;
}