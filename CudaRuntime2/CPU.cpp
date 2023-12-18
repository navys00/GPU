#include "CPU.h"
#include "omp.h"
#include "stdio.h"

void saxpy_CPU(int n, float a, float* x, int incx, float* y, int incy) {
  double begin, end;
  begin = omp_get_wtime();
  for (int i = 0; i < n; i++) {
    y[i * incy] = y[i * incy] + a * x[i * incx];
  }
  end = omp_get_wtime();
  printf("CPU FLOAT TIME: %f", end - begin);
}

void daxpy_CPU(int n, double a, double* x, int incx, double* y, int incy) {
  double begin, end;
  begin = omp_get_wtime();
  for (int i = 0; i < n; i++) {
    y[i * incy] = y[i * incy] + a * x[i * incx];
  }
  end = omp_get_wtime();
  printf("CPU DOUBLE TIME: %f", end - begin);
}

void saxpy_CPU_OMP(int n, float a, float* x, int incx, float* y, int incy) {
  double begin, end;
  begin = omp_get_wtime();
#pragma omp parallel for num_threads(8)
  
  for (int i = 0; i < n; i++) {
    y[i * incy] = y[i * incy] + a * x[i * incx];
  }
  end = omp_get_wtime();
  printf("CPU_OMP FLOAT TIME: %f", end - begin);
}

void daxpy_CPU_OMP(int n, double a, double* x, int incx, double* y, int incy) {
  double begin, end;
  begin = omp_get_wtime();
#pragma omp parallel for num_threads(8)
  
  for (int i = 0; i < n; i++) {
    y[i * incy] = y[i * incy] + a * x[i * incx];
  }
  end = omp_get_wtime();
  printf("CPU_OMP DOUBLE TIME: %f", end - begin);
}

int call_CPU_saxpy(int n, float a, float* x, int incx, float* y, int incy) {
  saxpy_CPU(n, a, x, incx,y,incy);
  return 0;
}

int call_CPU_daxpy(int n, double a, double* x, int incx, double* y, int incy) {
  daxpy_CPU(n, a, x, incx,y,incy);
  return 0;
}

int call_CPU_OMP_saxpy(int n, float a, float* x, int incx, float* y, int incy) {
  saxpy_CPU_OMP(n, a, x, incx, y, incy);
  return 0;
}

int call_CPU_OMP_daxpy(int n, double a, double* x, int incx, double* y, int incy) {
  daxpy_CPU_OMP(n, a, x, incx, y, incy);
  return 0;
}