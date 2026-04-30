// RUN: TYPEART_GPU=true %wrapper-cc -x cuda --cuda-gpu-arch=sm_50 %cuda_link %s -o %s.exe
// RUN: %s.exe 2>&1 | %filecheck %s

// REQUIRES: cuda_runtime && softcounter
// UNSUPPORTED: sanitizer

// CHECK: [0]=2 [1]=4 [2]=6 [3]=8
// CHECK: Total heap{{[ ]*}}:   2 ,    2 ,    -

#include <stdio.h>
__global__ void axpy(float a, float* x, float* y) {
  y[threadIdx.x] = a * x[threadIdx.x];
}

int main(int argc, char* argv[]) {
  const int kDataLen = 4;

  float a                = 2.0f;
  float host_x[kDataLen] = {1.0f, 2.0f, 3.0f, 4.0f};
  float host_y[kDataLen];

  float* device_x;
  float* device_y;
  cudaMalloc((void**)&device_x, kDataLen * sizeof(float));
  cudaMalloc((void**)&device_y, kDataLen * sizeof(float));

  cudaMemcpy(device_x, host_x, kDataLen * sizeof(float), cudaMemcpyHostToDevice);

  axpy<<<1, kDataLen>>>(a, device_x, device_y);

  cudaDeviceSynchronize();
  cudaMemcpy(host_y, device_y, kDataLen * sizeof(float), cudaMemcpyDeviceToHost);

  cudaDeviceReset();

  for (int i = 0; i < kDataLen; ++i) {
    printf("[%i]=%.0f ", i, host_y[i]);
  }
  printf("\n");

  return 0;
}
