// RUN: %wrapper-cc -x cuda -gencode arch=compute_50,code=sm_50 %s -o %s.exe
// RUN: %s.exe 2>&1 | %filecheck %s

// REQUIRES: cuda_runnable

// CHECK: Alloc [[CU_POINTER:0x[0-9a-z]+]] 5 float 4 20 ({{.*}}) H
// CHECK: Free [[CU_POINTER]] 5 float 4 20

#include <stdio.h>
int main() {
  const int N = 20;
  float* d_x;

  cudaMalloc((void**)&d_x, N * sizeof(float));
  cudaFree(d_x);

  return 0;
}
