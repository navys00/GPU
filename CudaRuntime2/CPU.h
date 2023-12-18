#pragma once

int call_CPU_saxpy(int n, float a, float* x, int incx, float* y, int incy);

int call_CPU_daxpy(int n, double a, double* x, int incx, double* y, int incy);

int call_CPU_OMP_saxpy(int n, float a, float* x, int incx, float* y, int incy);

int call_CPU_OMP_daxpy(int n, double a, double* x, int incx, double* y, int incy);