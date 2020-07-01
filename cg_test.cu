#define MAIN_PROGRAM

#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include "common.h"
#include "global.h"
#include "linalg_cpu.h"
#include "linalg_gpu.h"



void cpu_test(int n) {
    printf("conjugate gradient test starting on GPU with n=%d...\n", n);
    npts = n;
    int max_iter = 1100;
    float prec = 1e-20;

    float * A = (float *) malloc(n*n*sizeof(float));
    float * x = (float *) malloc(n*sizeof(float));
    float * b = (float *) malloc(n*sizeof(float));
    float * ax = (float *) malloc(n*sizeof(float));
    float * r = (float *) malloc(n*sizeof(float));

    gen_mat(A, n);
    set_zero(x, n);
    random_arr(b, n);

    cg(x, A, b, n, max_iter, prec);

    mat_vec_mul(ax, A, x, n);
    scalar_vec_add(r, ax, b, -1.0f, n);

    printf("Error: %e\n", abs_vec(r, n)/n);
}

void gpu_test(int n) {
    printf("conjugate gradient test starting on GPU with n=%d...\n", n);
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("device %d: %s \n\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    npts = n;

    blocksize=512;
    gridsize= (n+blocksize-1)/blocksize;
    int max_iter = 1100;
    float prec = 1e-20;

    float * A = (float *) malloc(n*n*sizeof(float));
    float * x = (float *) malloc(n*sizeof(float));
    float * b = (float *) malloc(n*sizeof(float));
    float * ax = (float *) malloc(n*sizeof(float));
    float * r = (float *) malloc(n*sizeof(float));

    float *A_d, *x_d, *b_d; // device pointer

    CHECK(cudaMalloc((void**)&A_d, n*n*sizeof(float)));
    CHECK(cudaMalloc((void**)&x_d, n*sizeof(float)));
    CHECK(cudaMalloc((void**)&b_d, n*sizeof(float)));

    gen_mat(A, n);
    set_zero(x, n);
    random_arr(b, n);

    CHECK(cudaMemcpy(A_d, A, n*n*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(x_d, x, n*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_d, b, n*sizeof(float), cudaMemcpyHostToDevice));

    cg_gpu(x_d, A_d, b_d, n, max_iter, prec);

    //CHECK(cudaMemcpy(A, A_d, n*n*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(x, x_d, n*sizeof(float), cudaMemcpyDeviceToHost));
    //CHECK(cudaMemcpy(b, b_d, n*sizeof(float), cudaMemcpyDeviceToHost));

    mat_vec_mul(ax, A, x, n);
    scalar_vec_add(r, ax, b, -1.0f, n);

    printf("Error: %e\n", abs_vec(r, n)/n);
}

int main() {
    int n = 1 << 10;
    
    cpu_test(n);
    gpu_test(n);

    return 0;
}