// RUN: %wrapper-cc -x cuda -gencode arch=compute_50,code=sm_50 %s -o %s.exe
// RUN: %s.exe 2>&1 | %filecheck %s

// REQUIRES: cuda_runnable

// CHECK: Alloc [[CU_POINTER:0x[0-9a-z]+]] 5 float 4 20 ({{.*}}) H
// CHECK: Status OK: 5 20
// CHECK: Free [[CU_POINTER]] 5 float 4 20
// CHECK: [Error]: Status not OK

#include "../../lib/runtime/RuntimeInterface.h"

#include <stdio.h>

void type_check(const void* addr) {
  int id_result      = 0;
  size_t count_check = 0;
  typeart_status status;
  status = typeart_get_type(addr, &id_result, &count_check);

  if (status != TYPEART_OK) {
    fprintf(stderr, "[Error]: Status not OK: %i for %p\n", status, addr);
  } else {
    fprintf(stderr, "Status OK: %i %zu\n", id_result, count_check);
  }
}

int main() {
  const int N = 20;
  float* d_x;

  cudaMallocHost((void**)&d_x, N * sizeof(float));
  type_check((void*)d_x);
  cudaFreeHost(d_x);
  type_check((void*)d_x);

  return 0;
}
