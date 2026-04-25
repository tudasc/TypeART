// RUN: %cuda-c-to-llvm %s | %apply-typeart -S 2>&1 | %filecheck %s

// REQUIRES: cuda_static

// CHECK: call void @__typeart_alloc_cuda({{(ptr|i8\*)}} %{{[0-9a-z]+}}, i32 {{[0-9]+}}, i64 20)

int main() {
  const int N = 20;
  float* d_x;

  cudaHostAlloc((void**)&d_x, N * sizeof(float), cudaHostAllocDefault);

  return 0;
}
