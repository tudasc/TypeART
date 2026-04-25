// RUN: %cuda-c-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s

// REQUIRES: cuda_static

// CHECK: __typeart_alloc_cuda({{(ptr|i8\*)}} %{{[0-9a-z]+}}, i32 {{[0-9]+}}, i64 {{.*}})
// CHECK: __typeart_alloc_cuda({{(ptr|i8\*)}} %{{[0-9a-z]+}}, i32 {{[0-9]+}}, i64 {{.*}})
// CHECK: __typeart_alloc_cuda({{(ptr|i8\*)}} %{{[0-9a-z]+}}, i32 {{[0-9]+}}, i64 {{.*}})

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
