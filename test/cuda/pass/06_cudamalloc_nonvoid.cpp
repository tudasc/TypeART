// RUN: %cuda-c-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s --check-prefix=%llvm-version-check

// REQUIRES: cuda_static

// LLVM: __typeart_alloc_gpu(ptr %{{[0-9a-z_]+}}, i32 23, i64 {{.*}})
// LLVM: __typeart_alloc_gpu(ptr %{{[0-9a-z_]+}}, i32 24, i64 {{.*}})
// LLVM: __typeart_alloc_mty_gpu(ptr %{{[0-9a-z_]+}}, ptr {{[@_a-z0-9A-Z]+}}, i64 {{.*}})

// LLVM_LEGACY: __typeart_alloc_gpu(i8* %{{[0-9a-z_]+}}, i32 23, i64 {{.*}})
// LLVM_LEGACY: __typeart_alloc_gpu(i8* %{{[0-9a-z_]+}}, i32 24, i64 {{.*}})
// LLVM_LEGACY: __typeart_alloc_gpu(i8* %{{[0-9a-z_]+}}, i32 2{{[0-9][0-9]}}, i64 {{.*}})

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
