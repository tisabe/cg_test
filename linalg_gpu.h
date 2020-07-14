#ifndef LINALG_GPU_H
#define LINALG_GPU_H

extern void assign_v2v_gpu(float *v, float *w, int n);
extern void cg_gpu(float * x, float * A, float * b, int n, int max_iter, float prec);
extern void dot_wrapper(float *res, float *g_a, float *g_b, unsigned int n);
extern void mat_vec_mul_gpu(float *res, float *mat, float *v, int n);
#endif