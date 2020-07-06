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

__global__ void reduceAdd (float *g_idata, float *g_odata, unsigned int n)
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
        sdata[tid] += g_idata[i] + g_idata[i+blockDim.x];
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

void dot_wrapper(float *g_a, float *g_b, float *g_odata, unsigned int n) {
    /* this function calculates the sum of elements in g_odata on the gpu,
    and saves it in the 0th element in g_idata. g_idata needs to have as many elements as there are threadblocks.
    It is assumed, that all arrays are already allocated on the gpu. 
    */
    dot_reduce<<<TB_SIZE, TB_SIZE>>>(g_a, g_b, g_odata, n);
    reduceAdd<<<1, TB_SIZE>>>(g_odata, g_odata, TB_SIZE);

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

__global__ void div_kernel(float *res, float *a, float *b, float s) {
   // res = s*a/b

   int idx=blockIdx.x*blockDim.x + threadIdx.x;

   if (idx == 0) {
      res[0] = s*a[0]/b[0];
   }
}

void div_gpu(float *res, float *a, float *b, float s) {
   // res = s*a/b
   div_kernel<<<1,1>>>(res, a, b, s);
}

void cg_gpu(float * x, float * A, float * b, int n, int max_iter, float prec) {
   float *r, *p, *ap; // vectors
   
   CHECK(cudaMalloc((void**) &r, n*sizeof(float)));
   CHECK(cudaMalloc((void**) &p, n*sizeof(float)));
   CHECK(cudaMalloc((void**) &ap, n*sizeof(float)));

   float *alpha, *beta, *rr, *rr_n, *err, *pap; // scalars (vectors with size gridsize)

   CHECK(cudaMalloc((void**) &alpha, gridsize*sizeof(float)));
   CHECK(cudaMalloc((void**) &beta, gridsize*sizeof(float)));
   CHECK(cudaMalloc((void**) &rr, gridsize*sizeof(float)));
   CHECK(cudaMalloc((void**) &rr_n, gridsize*sizeof(float)));
   CHECK(cudaMalloc((void**) &err, gridsize*sizeof(float)));
   CHECK(cudaMalloc((void**) &pap, gridsize*sizeof(float)));
   
   int k = 0;  // current iteration

   // value initialization
   mat_vec_mul_gpu(p, A, x, n);  // p=A*x temp storage of A*x
   sub_gpu(r, b, p, n); // r = b + -1*p
   assign_v2v_gpu(p, r, n); // p = r
   dot_wrapper(r, r, rr, n); 
   assign_v2v_gpu(err, rr, gridsize);
   CHECK(cudaDeviceSynchronize());
   
   float *h_err=(float*)malloc(gridsize*sizeof(float)); // error needs to be copied from device to host
   CHECK(cudaMemcpy(h_err, err, gridsize*sizeof(float), cudaMemcpyDeviceToHost));
   printf("h_err[0]=%f\n", h_err);

   float *err_0=(float*)malloc(gridsize*sizeof(float));
   CHECK(cudaMemcpy(err_0, err, gridsize*sizeof(float), cudaMemcpyDeviceToHost));
   printf("err_0[0]=%f\n", err_0);

   while((k < max_iter) && (h_err[0] > prec*prec*err_0[0])) {
      mat_vec_mul_gpu(ap, A, p, n);
      dot_wrapper(r, ap, pap, n); // pap = dot(r, ap)
      div_gpu(alpha, rr, pap, 1.0f);
      //scalar_vec_add(x, x, p, alpha, n);
      mul_add_gpu(x, x, alpha, p, n);
      if (k != 0 && k%50 == 0) {
         //mat_vec_mul(ap, A, x, n);
         mat_vec_mul_gpu(ap, A, x, n);
         //scalar_vec_add(r, b, ap, -1.0f, n);
         sub_gpu(r, b, ap, n);
         printf("Residual adjusted!\n");
      } else {
         //scalar_vec_add(r, r, ap, -1.0f*alpha, n);
         mul_sub_gpu(r, r, alpha, ap, n);
      }
      CHECK(cudaDeviceSynchronize());
      //rr_n = scalar_prod(r, r, n);
      dot_wrapper(r, r, rr_n, n);
      //beta = rr_n/rr;
      div_gpu(beta, rr_n, rr, 1.0f);
      //scalar_vec_add(p, r, p, beta, n);
      mul_add_gpu(p, r, beta, p, n);
      //rr = rr_n;
      assign_v2v_gpu(rr, rr_n, gridsize);
      //err = rr;
      assign_v2v_gpu(err, rr, gridsize);
      
      CHECK(cudaMemcpy(h_err, err, gridsize*sizeof(float), cudaMemcpyDeviceToHost));

      printf("Iteration: %d\n", k);
      printf("Precision: %e\n", h_err[0]);

      CHECK(cudaDeviceSynchronize());

      k++;
   }
   CHECK(cudaDeviceSynchronize());
   CHECK(cudaFree(r));
   CHECK(cudaFree(p));
   CHECK(cudaFree(ap));
   CHECK(cudaFree(alpha));
   CHECK(cudaFree(beta));
   CHECK(cudaFree(rr));
   CHECK(cudaFree(rr_n));
   CHECK(cudaFree(err));
   CHECK(cudaFree(pap));
   free(h_err);
   free(err_0);

}
