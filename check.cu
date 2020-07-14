#define MAIN_PROGRAM

#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include "common.h"
#include "global.h"
#include "linalg_cpu.h"
#include "linalg_gpu.h"

void dot_test() {
    
    int n = npts;

    blocksize=512;
    gridsize= (n+blocksize-1)/blocksize;

    printf("Testing dotproduct with n=%d\n", n);

    printf("block=%d\n", blocksize);
    printf("grid=%d\n\n", gridsize);

    float *a = (float *) malloc(n*sizeof(float));
    float *b = (float *) malloc(n*sizeof(float));
    float res;
    float res_gpu;

    float *d_a, *d_b, *d_res;
    CHECK(cudaMalloc((void**)&d_a, n*sizeof(float)));
    CHECK(cudaMalloc((void**)&d_b, n*sizeof(float)));
    CHECK(cudaMalloc((void**)&d_res, sizeof(float)));

    random_arr(a, n);
    random_arr(b, n);

    CHECK(cudaMemcpy(d_a, a, n*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, n*sizeof(float), cudaMemcpyHostToDevice));
    
    res = scalar_prod(a, b, n);
    dot_wrapper(d_res, d_a, d_b, n);

    CHECK(cudaMemcpy(&res_gpu, d_res, sizeof(float), cudaMemcpyDeviceToHost));
    //res_gpu = vector_prod_gpu2(d_a, d_b);

    printf("res=%f\n", res);
    printf("res_gpu=%f\n", res_gpu);
    printf("abs error=%f\n", (res-res_gpu));
    printf("rel error=%f\n", (res-res_gpu)/n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);
    free(a);
    free(b);
}

void mat_test() {
    
    int n = npts;
    blocksize=512;
    gridsize= (n+blocksize-1)/blocksize;

    printf("\nTesting mat-vec product with n=%d\n", n);

    printf("block=%d\n", blocksize);
    printf("grid=%d\n\n", gridsize);

    float *mat = (float *) malloc(n*n*sizeof(float));
    float *x = (float *) malloc(n*sizeof(float));
    float *res = (float *) malloc(n*sizeof(float));
    float *res_gpu = (float *) malloc(n*sizeof(float));
    float *diff = (float *) malloc(n*sizeof(float));

    float *d_mat, *d_x, *d_res;
    CHECK(cudaMalloc((void**)&d_mat, n*n*sizeof(float)));
    CHECK(cudaMalloc((void**)&d_x, n*sizeof(float)));
    CHECK(cudaMalloc((void**)&d_res, n*sizeof(float)));

    random_arr(mat, n*n);
    random_arr(x, n);

    CHECK(cudaMemcpy(d_mat, mat, n*n*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice));

    mat_vec_mul(res, mat, x, n);
    mat_vec_mul_gpu(d_res, d_mat, d_x, n);

    CHECK(cudaMemcpy(res_gpu, d_res, n*sizeof(float), cudaMemcpyDeviceToHost));

    scalar_vec_add(diff, res, res_gpu, -1.0f, n);
    float err = abs_vec(diff, n);

    printf("abs error=%f\n", err);
    printf("rel error=%f\n", err/n);
}

int main() {

    npts = 1 << 10;
    dot_test();
    mat_test();

    return 0;
}