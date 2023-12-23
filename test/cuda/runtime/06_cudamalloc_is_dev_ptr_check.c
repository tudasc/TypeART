// RUN: %wrapper-cc -x cuda -gencode arch=compute_50,code=sm_50 %s -o %s.exe
// RUN: %s.exe 2>&1 | %filecheck %s

// REQUIRES: cuda_runnable
// UNSUPPORTED: sanitizer

// CHECK: Status OK
// CHECK: Not device ptr
// CHECK: Not device ptr
// CHECK: Not device ptr

#include "../../lib/runtime/CudaRuntimeInterface.h"

#include <stdio.h>

void type_check(const void* addr) {
  bool id_result = false;
  typeart_status status;
  status = typeart_cuda_is_device_ptr(addr, &id_result);

  if (status != TYPEART_OK) {
    fprintf(stderr, "[Error]: Status not OK: %i for %p\n", status, addr);
  } else {
    if (id_result == true) {
      fprintf(stderr, "Status OK\n");
    } else {
      fprintf(stderr, "Not device ptr\n");
    }
  }
}

int main() {
  const int N = 20;

  float* d_x;
  cudaMalloc((void**)&d_x, N * sizeof(float));
  type_check((void*)d_x);
  cudaFree(d_x);
  type_check((void*)d_x);

  int* h_x = (int*)malloc(sizeof(int));
  type_check((void*)h_x);
  free(h_x);

  cudaHostAlloc((void**)&h_x, sizeof(int), cudaHostAllocDefault);
  type_check((void*)h_x);
  cudaFreeHost(h_x);

  return 0;
}
