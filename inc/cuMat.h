
#ifndef _CU_MAT_H_
#define _CU_MAT_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void cuMat2DAdd_kernel(float* _C, const float* _A, const float* _B, int n);


__global__ void cuMat2DRolate180_kernel(float* _B, const float* _A, int n);


void cuMat2dCorrelation_Valid(float *srcMat, nSize srcSize, float *mapMat, nSize mapSize, float *dstMat, nSize dstSize);

#endif //_CU_MAT_H_