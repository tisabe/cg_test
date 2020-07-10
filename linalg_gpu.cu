#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "global.h"
#include "common.h"

#define TB_SIZE 256

__global__ void assign_v2v_gpu_kernel(float *v, float *w, int n)
{
   int idx=blockIdx.x*blockDim.x + threadIdx.x;

   //for (; (idx<n); idx+=blockDim.x*gridDim.x)
   if (idx<n) {
      //printf("assigned %d\n", idx);
      v[idx] = w[idx];
   }
}

void assign_v2v_gpu(float *v, float *w, int n)
{
   assign_v2v_gpu_kernel<<<gridsize,blocksize>>>(v,w,n);
   CHECK(cudaDeviceSynchronize());
}

__global__ void sub_gpu_kernel(float *res, float *v, float *w, int n)
{
   int idx=blockIdx.x*blockDim.x + threadIdx.x;

   for (; (idx<n); idx+=blockDim.x*gridDim.x)
      res[idx] += v[idx] - w[idx];
}

__global__ void mul_add_gpu_kernel(float *res, float *v, float *alpha, float *w, int n)
{
   int idx=blockIdx.x*blockDim.x + threadIdx.x;

   for (; (idx<n); idx+=blockDim.x*gridDim.x)
      res[idx] += v[idx] + alpha[0]*w[idx];
}

__global__ void mul_sub_gpu_kernel(float *res, float *v, float *alpha, float *w, int n)
{
   int idx=blockIdx.x*blockDim.x + threadIdx.x;

   for (; (idx<n); idx+=blockDim.x*gridDim.x)
      res[idx] += v[idx] - alpha[0]*w[idx];
}

void sub_gpu(float *res, float *v, float *w, int n)
{
   sub_gpu_kernel<<<gridsize,blocksize>>>(res,v,w,n);
   //CHECK(cudaDeviceSynchronize());
}

void mul_add_gpu(float *res, float *v, float *alpha, float *w, int n)
{
   mul_add_gpu_kernel<<<gridsize,blocksize>>>(res,v,alpha,w,n);
   //CHECK(cudaDeviceSynchronize());
}

void mul_sub_gpu(float *res, float *v, float *alpha, float *w, int n)
{
   mul_add_gpu_kernel<<<gridsize,blocksize>>>(res,v,alpha,w,n);
   //CHECK(cudaDeviceSynchronize());
}

__global__ void vector_mul_gpu(float *u, float *v, float *w, int n)
{
   int idx=blockIdx.x*blockDim.x + threadIdx.x;

   for (; (idx<n); idx+=blockDim.x*gridDim.x)
      u[idx] = v[idx]*w[idx];
}

__device__ void warpReduce(volatile float* sdata, int tid) {
   /*this function unrolls the last warp*/
   sdata[tid] += sdata[tid + 32];
   sdata[tid] += sdata[tid + 16];
   sdata[tid] += sdata[tid + 8];
   sdata[tid] += sdata[tid + 4];
   sdata[tid] += sdata[tid + 2];
   sdata[tid] += sdata[tid + 1];
}

__global__ void dot_reduce(float *g_a, float *g_b, float *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int gridSize = blockDim.x*2*gridDim.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    __shared__ float sdata[TB_SIZE];
    //sdata[tid] = g_idata[idx];
	__syncthreads();

    // add as many as possible (= 2*(n/gridSize))
    //int sum=0;
    sdata[tid] = 0.0f;
    int i=idx;
    while (i<n)
    {
        //sum += g_idata[i] + g_idata[i+blockDim.x];
        sdata[tid] += g_a[i]*g_b[i] + g_a[i+blockDim.x]*g_b[i+blockDim.x];
        i += gridSize;
    }
    //g_idata[idx] = sum;

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 32; stride /= 2)
    //for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            //g_idata[idx] += g_idata[idx + stride];
            sdata[tid] += sdata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }
        
    if (tid < 32) warpReduce(sdata, tid);

    // write result for this block to global mem
    //if (tid == 0) g_odata[blockIdx.x] = g_idata[idx];
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduceAdd (float * res, float *g_idata, float *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    
    __shared__ float sdata[TB_SIZE];
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    sdata[tid] = g_idata[i];

    
    __syncthreads();

    // in-place reduction in global memory
    //for (int stride = blockDim.x/2; stride > 32; stride /= 2)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            //g_idata[idx] += g_idata[idx + stride];
            sdata[tid] += sdata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }
        
    //if (tid < 32) warpReduce(sdata, tid);

    // write result for this block to global mem
    //if (tid == 0) g_odata[blockIdx.x] = g_idata[idx];
    //if (tid == 0) g_odata[blockIdx.x] = sdata[0];
    if (tid == 0) {
         *res = sdata[0];
         //printf("res in function = %f\n", *res);
    }
}


void dot_wrapper(float *res, float *g_a, float *g_b, unsigned int n) {
    /* this function calculates the sum of elements in g_odata on the gpu,
    and saves it in the 0th element in g_idata. g_idata needs to have as many elements as there are threadblocks.
    It is assumed, that all arrays are already allocated on the gpu. 
    */

    float *tmp;
    CHECK(cudaMalloc((void**) &tmp, gridsize*sizeof(float)));
    dot_reduce<<<gridsize, TB_SIZE>>>(g_a, g_b, tmp, n);
    //CHECK(cudaDeviceSynchronize());
    reduceAdd<<<1, gridsize>>>(res, tmp, tmp, n);
    CHECK(cudaFree(tmp));

}

__global__ void mat_vec_mul_kernel(float * res, float * mat, float * v, int n) {
   /* matrix-vector multiplication for square n*n matrix mat:
      res = mat*v
   */
   int idx=blockIdx.x*blockDim.x + threadIdx.x;

   for (; (idx<n); idx+=blockDim.x*gridDim.x) {
      res[idx] = 0.0f;
      for (int i=0; i<n; i++) {
         res[idx] += mat[idx*n + i]*v[i];
      }
   }
}

void mat_vec_mul_gpu(float *res, float *mat, float *v, int n) {
   mat_vec_mul_kernel<<<gridsize, blocksize>>>(res, mat, v, n);
}

__global__ void div_kernel(float *res, float *a, float *b) {
   // res = s*a/b

   int idx=blockIdx.x*blockDim.x + threadIdx.x;

   if (idx == 0) {
      *res = (*a)/(*b);
   }
}

void div_gpu(float *res, float *a, float *b) {
   // res = a/b
   div_kernel<<<1,1>>>(res, a, b);
}

void scalar_cpy(float * dest, float * src) {
   cudaMemcpy(dest, src, sizeof(float), cudaMemcpyDeviceToDevice); 
}

void cg_gpu(float * d_x, float * d_A, float * d_b, int n, int max_iter, float prec) {
   float *d_r, *d_p, *d_ap; // vectors on device
   
   CHECK(cudaMalloc((void**) &d_r, n*sizeof(float)));
   CHECK(cudaMalloc((void**) &d_p, n*sizeof(float)));
   CHECK(cudaMalloc((void**) &d_ap, n*sizeof(float)));

   float *d_alpha, *d_beta, *d_rr, *d_rr_n, *d_err, *d_pap; // scalars on device

   CHECK(cudaMalloc((void**) &d_alpha, sizeof(float)));
   CHECK(cudaMalloc((void**) &d_beta, sizeof(float)));
   CHECK(cudaMalloc((void**) &d_rr, sizeof(float)));
   CHECK(cudaMalloc((void**) &d_rr_n, sizeof(float)));
   CHECK(cudaMalloc((void**) &d_err, sizeof(float)));
   CHECK(cudaMalloc((void**) &d_pap, sizeof(float)));

   float h_err, h_err0; // scalars on host
   
   int k = 0;  // current iteration

   // value initialization
   mat_vec_mul_gpu(d_p, d_A, d_x, n);  // p=A*x temp storage of A*x
   sub_gpu(d_r, d_b, d_p, n); // r = b + -1*p
   assign_v2v_gpu(d_p, d_r, n); // p = r
   dot_wrapper(d_rr, d_r, d_r, n); // rr = dot(r, r)
   //assign_v2v_gpu(err, rr, gridsize);
   scalar_cpy(d_err, d_rr);

   CHECK(cudaMemcpy(&h_err0, d_err, sizeof(float), cudaMemcpyDeviceToHost));
   h_err = h_err0;

   printf("h_err0=%f\n", h_err0);
   printf("h_err=%f\n", h_err);

   CHECK(cudaDeviceSynchronize());
   
   

   while((k < max_iter) && (h_err > prec*prec*h_err0)) {
      mat_vec_mul_gpu(d_ap, d_A, d_p, n);
      dot_wrapper(d_pap, d_r, d_ap, n); // pap = dot(r, ap)
      div_gpu(d_alpha, d_rr, d_pap); // alpha = rr/pap
      //scalar_vec_add(x, x, p, alpha, n);
      mul_add_gpu(d_x, d_x, d_alpha, d_p, n);
      if (k != 0 && k%50 == 0) {
         //mat_vec_mul(ap, A, x, n);
         mat_vec_mul_gpu(d_ap, d_A, d_x, n);
         //scalar_vec_add(r, b, ap, -1.0f, n);
         sub_gpu(d_r, d_b, d_ap, n);
         printf("Residual adjusted!\n");
      } else {
         //scalar_vec_add(r, r, ap, -1.0f*alpha, n);
         mul_sub_gpu(d_r, d_r, d_alpha, d_ap, n);
      }
      CHECK(cudaDeviceSynchronize());
      //rr_n = scalar_prod(r, r, n);
      dot_wrapper(d_rr_n, d_r, d_r, n);
      //beta = rr_n/rr;
      div_gpu(d_beta, d_rr_n, d_rr);
      //scalar_vec_add(p, r, p, beta, n);
      mul_add_gpu(d_p, d_r, d_beta, d_p, n);
      //rr = rr_n;
      scalar_cpy(d_rr, d_rr_n);
      //err = rr;
      scalar_cpy(d_err, d_rr);
      
      //CHECK(cudaMemcpy(h_err, err, gridsize*sizeof(float), cudaMemcpyDeviceToHost));
      //CHECK(cudaMemcpyToSymbol(&h_err, d_err, sizeof(float), cudaMemcpyDeviceToHost));
      CHECK(cudaMemcpy(&h_err, d_err, sizeof(float), cudaMemcpyDeviceToHost));

      printf("Iteration: %d\n", k);
      printf("error: %e\n\n", h_err);

      CHECK(cudaDeviceSynchronize());

      k++;
   }
   CHECK(cudaDeviceSynchronize());
   CHECK(cudaFree(d_r));
   CHECK(cudaFree(d_p));
   CHECK(cudaFree(d_ap));
   
}
