#include <stdio.h>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "GPU.cuh"
#include <omp.h>


__global__ void saxpy_GPU_kernel(int n, float a, float* x, int incx, float* y, int incy) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {

    y[i * incy] = y[i * incy] + a * x[i * incx];

  }

}

__global__ void daxpy_GPU_kernel(int n, double a, double* x, int incx, double* y, int incy) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {

    y[i * incy] = y[i * incy] + a * x[i * incx];

  }

}

int call_saxpy_GPU(int n, float a, float* x, int incx, float* y, int incy, int block_size) {
  float *x_float, *y_float;

  int x_gpu_size = 1 + (n - 1) * incx;
  int y_gpu_size = 1 + (n - 1) * incy;

  cudaMalloc((void**)&x_float, x_gpu_size * sizeof(float));
  cudaMalloc((void**)&y_float, y_gpu_size * sizeof(float));

  cudaMemcpy(x_float, x, x_gpu_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(y_float, y, y_gpu_size * sizeof(float), cudaMemcpyHostToDevice);

  int num_blocks = (n + block_size - 1) / block_size;

  cudaEvent_t start, stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  saxpy_GPU_kernel<<<num_blocks, block_size>>>(n, a, x_float, incx, y_float, incy);

  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("Float:  time = %f s, block_size = %d\n", elapsedTime / 1000, block_size);

  cudaMemcpy(y, y_float, y_gpu_size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(x_float);
  cudaFree(y_float);

  return 0;
}

int call_daxpy_GPU(int n, double a, double* x, int incx, double* y, int incy, int block_size) {
  double *x_double, *y_double;

  int x_gpu_size = 1 + (n - 1) * incx;
  int y_gpu_size = 1 + (n - 1) * incy;

  cudaMalloc((void**)&x_double, x_gpu_size * sizeof(double));
  cudaMalloc((void**)&y_double, y_gpu_size * sizeof(double));

  cudaMemcpy(x_double, x, x_gpu_size * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(y_double, y, y_gpu_size * sizeof(double), cudaMemcpyHostToDevice);

  int num_blocks = (n + block_size - 1) / block_size;

  cudaEvent_t start, stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  daxpy_GPU_kernel<<<num_blocks, block_size>>>(n, a, x_double, incx, y_double, incy);

  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("Double: time = %f s, block_size = %d\n", elapsedTime / 1000, block_size);

  cudaMemcpy(y, y_double, y_gpu_size * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(x, x_double, x_gpu_size * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(x_double);
  cudaFree(y_double);

  return 0;
}
