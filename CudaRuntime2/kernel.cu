
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void kernel1() { printf("Hello, world!\n"); }

__global__ void kernel2() {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  printf("I am from %d block, %d thread (global index: %d)\n", blockIdx.x,
         threadIdx.x, k);
}

__global__ void kernel3(int* a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    a[i] += i;
  }
}

int main() {
  int n = 10;
  const int block_size = 256;
  const int num_block = (n + block_size - 1) / block_size;

  kernel1<<<2, 2>>>();

  kernel2<<<2, block_size>>>();

  int* a = new int[n];
  int* a_gpu;
  for (int i = 0; i < n; i++) {
    a[i] = 0;
  }
  cudaMalloc((void**)&a_gpu, n * sizeof(int));

  cudaMemcpy(a_gpu, a, n * sizeof(int), cudaMemcpyHostToDevice);

  kernel3<<<num_block, block_size>>>(a_gpu, n);

  cudaMemcpy(a, a_gpu, n * sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i < n; i++) {
    printf("%d\n", a[i]);
  }
  delete[] a;
  cudaFree(a_gpu);

  cudaDeviceSynchronize();
  return 0;
}
