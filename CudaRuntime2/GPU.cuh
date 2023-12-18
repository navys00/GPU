#pragma once
int call_saxpy_GPU(int n, float a, float* x, int incx, float* y, int incy, int block_size);
int call_daxpy_GPU(int n, double a, double* x, int incx, double* y, int incy, int block_size);

__global__ void saxpy_GPU_kernel(int n, float a, float* x, int incx, float* y,
                                int incy);

__global__ void daxpy_GPU_kernel(int n, double a, double* x, int incx, double* y,
                                int incy);
