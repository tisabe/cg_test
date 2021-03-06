#ifndef LINALG_GPU_H
#define LINALG_GPU_H

extern void assign_v2v_gpu(float *v, float *w, int n);
extern void cg_gpu(float * x, float * A, float * b, int n, int max_iter, float prec);

#endif