// RUN: %apply %s -x cuda --cuda-gpu-arch=sm_72 2>&1 | %filecheck %s

// REQUIRES: cuda

// CHECK: Malloc :   2

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
  return 0;
}