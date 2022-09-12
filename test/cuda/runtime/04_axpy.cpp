// RUN: %wrapper-cxx -x cuda --cuda-gpu-arch=sm_72 %s -o %s.exe
// RUN: cuda-memcheck --leak-check full %s.exe 2>&1 | %filecheck %s

// REQUIRES: cuda_runnable && softcounter

// CHECK: Status OK
// CHECK: Status OK
// CHECK: Status OK
// CHECK: Status OK
// CHECK: 5   :   2 ,    3 ,    1 , float

#include "../../lib/runtime/RuntimeInterface.h"

#include <iostream>

void type_check(const void* addr) {
  int id_result      = 0;
  size_t count_check = 0;
  typeart_status status;
  status = typeart_get_type(addr, &id_result, &count_check);

  if (status != TYPEART_OK) {
    std::cerr << "[Error]: Status not OK: " << status << " " << addr << "\n";
  } else {
    std::cerr << "Status OK: " << id_result << " " << count_check << "\n";
  }
}

__global__ void axpy(float a, float* x, float* y) {
  y[threadIdx.x] = a * x[threadIdx.x];
}

int main(int argc, char* argv[]) {
  const int kDataLen = 4;

  float a                = 2.0f;
  float host_x[kDataLen] = {1.0f, 2.0f, 3.0f, 4.0f};
  float host_y[kDataLen];

  type_check((void*)&host_x[0]);
  type_check((void*)&host_y[0]);

  float* device_x;
  float* device_y;
  cudaMalloc((void**)&device_x, kDataLen * sizeof(float));
  cudaMalloc((void**)&device_y, kDataLen * sizeof(float));

  type_check((void*)device_x);
  type_check((void*)device_y);

  cudaMemcpy(device_x, host_x, kDataLen * sizeof(float), cudaMemcpyHostToDevice);

  axpy<<<1, kDataLen>>>(a, device_x, device_y);

  cudaDeviceSynchronize();
  cudaMemcpy(host_y, device_y, kDataLen * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < kDataLen; ++i) {
    std::cout << "y[" << i << "] = " << host_y[i] << "\n";
  }

  cudaFree(device_x);
  cudaFree(device_y);

  cudaDeviceReset();
  return 0;
}