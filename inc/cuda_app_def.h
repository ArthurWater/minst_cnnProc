
#ifndef _CUDA_APP_DEF_H_
#define _CUDA_APP_DEF_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define CUDA_STS_CHECK(st) do{st = cudaGetLastError();if (st != cudaSuccess){printf("addKernel launch failed: %s\n", cudaGetErrorString(st));}}while (0);


#endif //_CUDA_APP_DEF_H_
