#define MAIN_PROGRAM

#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include "common.h"
#include "global.h"
#include "linalg_cpu.h"
#include "linalg_gpu.h"

void dot_test() {
    npts = 1 << 13;
    int n = npts;

    blocksize=512;
    gridsize= (n+blocksize-1)/blocksize;
    printf("block=%d\n", blocksize);
    printf("grid=%d\n\n", gridsize);

    printf("Testing dotproduct with n=%d\n", n);
    float *a = (float *) malloc(n*sizeof(float));
    float *b = (float *) malloc(n*sizeof(float));
    float res;
    float res_gpu;

    float *d_a, *d_b, *d_res;
    CHECK(cudaMalloc((void**)&d_a, n*sizeof(float)));
    CHECK(cudaMalloc((void**)&d_b, n*sizeof(float)));
    CHECK(cudaMalloc((void**)&d_res, sizeof(float)));

    for (int i=0; i<n; i++) {
        a[i] = 2.0f;
        b[i] = 1.0f;
    }

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

}

int main() {
    
    dot_test();

    return 0;
}