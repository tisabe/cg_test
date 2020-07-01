#ifndef LINALG_CPU_H
#define LINALG_CPU_H

#include "global.h"
#include <cuda_runtime.h>

extern void set_zero(float * v, int n);
extern void scalar_vec_add(float * res, float * v, float * w, float s, int n);
extern float scalar_prod(float * v, float * w, int n);
extern float abs_vec(float * v, int n);
extern void mat_vec_mul(float * res, float * mat, float * v, int n);
extern void cg(float * x, float * A, float * b, int n, int max_iter, float prec);
extern void random_arr(float * arr, int n);
extern void gen_mat(float * mat, int n);

#endif