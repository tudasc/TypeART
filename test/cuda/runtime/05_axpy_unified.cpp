// RUN: %wrapper-cxx -x cuda --cuda-gpu-arch=sm_72 %s -o %s.exe
// RUN: cuda-memcheck --leak-check full %s.exe 2>&1 | %filecheck %s

// REQUIRES: cuda_runnable && softcounter

// CHECK: Status OK
// CHECK: Status OK
// CHECK: 5   :   2 ,    {{[0-9]+}} ,    0 , float

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

  float a = 2.0f;

  float* x;
  float* y;
  cudaMallocManaged((void**)&x, kDataLen * sizeof(float));
  cudaMallocManaged((void**)&y, kDataLen * sizeof(float));

  for (int i = 0; i < kDataLen; ++i) {
    x[i] = (float)i;
  }

  type_check((void*)x);
  type_check((void*)y);

  axpy<<<1, kDataLen>>>(a, x, y);

  cudaDeviceSynchronize();

  for (int i = 0; i < kDataLen; ++i) {
    std::cout << "y[" << i << "] = " << y[i] << "\n";
  }

  cudaFree(x);

  cudaDeviceReset();
  return 0;
}