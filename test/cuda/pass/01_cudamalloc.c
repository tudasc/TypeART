// RUN: %cuda-c-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s

// REQUIRES: cuda

// CHECK-NOT: [Error]

int main() {
  const int N = 20;
  float* x;
  float* y;
  float* d_x;
  float* d_y;

  x = (float*)malloc(N * sizeof(float));
  y = (float*)malloc(N * sizeof(float));

  cudaMalloc(&d_x, N * sizeof(float));
  cudaMalloc(&d_y, N * sizeof(float));

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);

  return 0;
}