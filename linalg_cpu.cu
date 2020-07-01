#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

//#include "cg_cpu.h"

/*
The following functions implement the conjugate gradient algorithm 
on the CPU as a pretest for the implementation on the GPU.
Uses float as this will also be fastest on the GPU.
*/
void set_zero(float * v, int n) {
    for (int i=0; i<n; i++) {
        v[i] = 0.0f;
    }
}

void scalar_vec_add(float * res, float * v, float * w, float s, int n) {
    /* add two arrays length n, with scalar multiplication as res = v + s*w */
    for (int i=0; i<n; i++) {
        res[i] = v[i] + s*w[i];
    }
}

float scalar_prod(float * v, float * w, int n) {
    /* returns scalar product of two arrays v, w*/
    float res = 0;
    for (int i=0; i<n; i++) {
        res += v[i] * w[i];
    }
    return res;
}

float abs_vec(float * v, int n) {
    /* returns the abs of a vector*/
    return sqrt(scalar_prod(v, v, n));
}

void mat_vec_mul(float * res, float * mat, float * v, int n) {
    /* matrix-vector multiplication for square matrices size n*n */
    for (int i=0; i<n; i++) { // row index
        res[i] = 0.0f;
        for (int j=0; j<n; j++) { // column index
            res[i] += mat[i*n + j]*v[j];
        }
    }
}

void cg(float * x, float * A, float * b, int n, int max_iter, float prec) {
    /* conjugate gradient A*x = b (A=mat) */
    float * r = (float*)malloc(n * sizeof(float));  // residue
    float * p = (float*)malloc(n * sizeof(float));  // orthogonal search directions
    float * ap = (float*)malloc(n * sizeof(float)); // A*p storage vector

    float alpha = 0.0f;
    float beta = 0.0f;  
    float rr = 0.0f;    // <r,r> dot product
    float rr_n = 0.0f;  // <r,r> dot product for next iteration
    float err;
    float err_0;
    
    int k = 0;  // current iteration

    // initialize starting values
    mat_vec_mul(p, A, x, n);  // p=A*x temp storage of A*x
    scalar_vec_add(r, b, p, -1.0f, n);  // r = b + -1*p
    memcpy(p, r, n*sizeof(float));
    rr = scalar_prod(r, r, n);
    err = rr;
    err_0 = err;

    while((k < max_iter) && (err > prec*prec*err_0)) {
        mat_vec_mul(ap, A, p, n);
        alpha = rr/scalar_prod(r, ap, n);
        scalar_vec_add(x, x, p, alpha, n);
        if (k%50 == 0) {
            mat_vec_mul(ap, A, x, n);
            scalar_vec_add(r, b, ap, -1.0f, n);
            printf("Residual adjusted!\n");
        } else {
            scalar_vec_add(r, r, ap, -1.0f*alpha, n);
        }
        rr_n = scalar_prod(r, r, n);
        beta = rr_n/rr;
        scalar_vec_add(p, r, p, beta, n);
        rr = rr_n;
        err = rr;
        printf("Iteration: %d\n", k);
        printf("Precision: %e\n", err);

        k++;
    }
    if (k >= max_iter) {
        printf("Maximum number of iterations (%d) reached, aborting calculation at precision (relative error) %e\n", k, abs_vec(r,n));
    }
    
    free(r);
    free(p);
    free(ap);
}

/* following are functions to produce positive definite, symmetric matrices */

void random_arr(float * arr, int n) {
    // fill an array with (semi) random floats in range (-0.5, 0,5)
    for (int i=0; i<n; i++) {
        arr[i] = (float)rand()/RAND_MAX - 0.5f;
    }
}

void gen_mat(float * mat, int n) {
    float * transp = (float*)malloc(n*n*sizeof(float));
    random_arr(mat, n*n);

    for (int i=0; i<n; i++) {
        mat[i*n + i] = 0.0f;
    }

    for (int i=0; i<n; i++){
        for (int j=0; j<n; j++){
            transp[i*n+j] = mat[j*n+i];
        }
    }
    scalar_vec_add(mat, transp, mat, 1.0f, n*n);

    for (int i=0; i<n; i++){
        mat[i*n+i] += n;
    }
    free(transp);
}






