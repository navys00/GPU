#include <omp.h>
#include <stdio.h>
#include <iostream>
#include "CPU.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "GPU.cuh"

const int n = 1000000;
int incx = 2;
int incy = 3;

int size_x = 1 + (n - 1) * incx;
int size_y = 1 + (n - 1) * incy;

float* x_float = new float[size_x];
float* y_float = new float[size_y];

double* x_double = new double[size_x];
double* y_double = new double[size_y];

const float a_float = 2.0f;
const double a_double = 2.0;



int main() {
  std::fill_n(x_float, n, rand() % 10);
  std::fill_n(y_float, n, rand() % 10);

  std::fill_n(x_double, n, rand() % 10);
  std::fill_n(y_double, n, rand() % 10);

  std::cout << "CPU: \n";
  
  call_CPU_saxpy(n, a_float, x_float, incx, y_float, incy);
  
  printf("\n");
 
  call_CPU_daxpy(n, a_double, x_double, incx, y_double, incy);

  printf("\n");

  std::cout << "CPU OMP: \n";
  call_CPU_OMP_saxpy(n, a_float, x_float, incx, y_float, incy);
  printf("\n");

  call_CPU_OMP_daxpy(n, a_double, x_double, incx, y_double, incy);
  printf("\n");

  std::cout << "GPU: \n";
  for (int i = 3; i < 9; i++) {
    call_saxpy_GPU(n, a_float, x_float, incx, y_float, incy, pow(2, i));
    call_daxpy_GPU(n, a_double, x_double, incx, y_double,incy, pow(2, i));
  }
}