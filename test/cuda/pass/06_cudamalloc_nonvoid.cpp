// RUN: %cuda-c-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s

// REQUIRES: cuda

// CHECK: __typeart_alloc(i8* %{{[0-9a-z]+}}, i32 5
// CHECK: __typeart_alloc(i8* %{{[0-9a-z]+}}, i32 6
// CHECK: __typeart_alloc(i8* %{{[0-9a-z]+}}, i32 {{[2][0-9]+}}

struct X {
  int a;
  int b;
};

int main() {
  const int N = 20;
  float* d_x;
  double* dd_y;
  X* sd_z;

  cudaMalloc(&d_x, N * sizeof(float));
  cudaMalloc(&dd_y, N * sizeof(double));
  cudaMalloc(&sd_z, N * sizeof(X));

  return 0;
}