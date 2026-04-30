// RUN: TYPEART_GPU=true %wrapper-cc -x hip --offload-arch=%hip_arch %hip_link %s -o %s.exe
// RUN: %s.exe 2>&1 | %filecheck %s

// REQUIRES: hip_runtime && !llvm-14 && softcounter
// UNSUPPORTED: sanitizer

// CHECK: [0]=2 [1]=4 [2]=6 [3]=8
// CHECK: Total heap{{[ ]*}}:   2 ,    2 ,    -

#include <hip/hip_runtime.h>
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
  hipMalloc((void**)&device_x, kDataLen * sizeof(float));
  hipMalloc((void**)&device_y, kDataLen * sizeof(float));

  hipMemcpy(device_x, host_x, kDataLen * sizeof(float), hipMemcpyHostToDevice);

  axpy<<<1, kDataLen>>>(a, device_x, device_y);

  hipDeviceSynchronize();
  hipMemcpy(host_y, device_y, kDataLen * sizeof(float), hipMemcpyDeviceToHost);

  hipDeviceReset();

  for (int i = 0; i < kDataLen; ++i) {
    printf("[%i]=%.0f ", i, host_y[i]);
  }
  printf("\n");

  return 0;
}
