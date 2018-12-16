

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "../inc/common.h"
#include "cuda_app_def.h"
#include "mat.h"
#include "cuMat.h"

/*矩阵旋转180度*/
__global__ void cuMat2dRotate_180_kernel(float *matIn, float *matOut, unsigned int outW, unsigned int outH)
{
	int i = threadIdx.x;
	int j = threadIdx.y;
	if (!matIn || !matOut || !outW || !outH)
	{
		return;
	}
	int row = blockIdx.y*blockDim.y + threadIdx.y;  // X 对应矩阵row, Y对应举证col
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	matOut[j * outW + i] = matIn[i * outH+ j];

}

__global__ void addKernel(float *c, const float *a, const float *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

//矩阵加法的kernel
__global__ void cuMat2DAdd_kernel(float* _C, const float* _A, const float *_B, int n)
{

	//找出该线程所在的行列
	int row = (blockIdx.x*blockDim.x + threadIdx.x)/n;  // X 对应矩阵row, Y对应举证col
	int col = (blockIdx.x*blockDim.x + threadIdx.x)%n;
	if (row < n)
	{
		//线程Thread(row,col)负责计算C(row,col)
		_C[col + row*n] = _A[col + row*n] + _B[col + row*n];
	}
}

__global__ void cuMat2DRolate180_kernel(float* _B, const float* _A, int n)
{

	//找出该线程所在的行列
	int row = (blockIdx.x*blockDim.x + threadIdx.x) / n;  // X 对应矩阵row, Y对应举证col
	int col = (blockIdx.x*blockDim.x + threadIdx.x) % n;
	if (row < n)
	{
		//线程Thread(row,col)负责计算C(row,col)
		_B[col + row*n] = _A[(n-1-col) + (n-1-row)*n];
	}
}

__global__ void cuMat2DSubSum_kernel(float* inData, nSize inSize, float* mapMat, nSize mapSize, float* outData, int outSizeW)
{
	int i = threadIdx.x;
	int j = threadIdx.y;
	int r, c;
//	printf("x:%d, y:%d\n", blockDim.x, blockDim.y);
	outData[j * outSizeW + i] = (float)0.0;
	for (r = 0; r < mapSize.r; r++)
	{
		for (c = 0; c < mapSize.c; c++)
		{
			outData[j * outSizeW + i] += mapMat[r * mapSize.c + c] * inData[(j + r) * inSize.c + i + c];
		}
	}
}

__global__ void cuMat2dEdgeExpand_kernel(float *matIn, nSize matSize, float *matOut, int addc, int addr)
{
	int col, row;
	int out_c = matSize.c + 2 * addc;
	col = blockIdx.x*blockDim.x + threadIdx.x;
	row = blockIdx.y*blockDim.y + threadIdx.y;
//	printf("x:%d, y:%d\n", blockDim.x, blockDim.y);

	if (row < addr || row >= (matSize.r + addr) \
		|| col < addc || col >= (matSize.c + addc))
	{
		matOut[row * out_c + col] = (float)0.0;
	}		
	else
	{
		matOut[row * out_c + col] = matIn[(row - addr) * matSize.c + col - addc]; /* 复制原向量的数据 */
	}
}

__global__ void cuMat2dEdgeShrink_kernel(float *matIn, nSize matSize, float *matOut, int shrinkc, int shrinkr)
{
	int i, j;
	int w = matSize.c;
	int h = matSize.r;
	i = threadIdx.x;
	j = threadIdx.y;
	
	if ((j >= shrinkr) && (i >= shrinkc) && (j < (h - shrinkr)) && (i < (w - shrinkc)))
	{
		matOut[(j - shrinkr) * (w - 2 * shrinkc) + i - shrinkc] = matIn[j * w + i]; /* 复制原向量的数据 */
	}

}

void cuMat2dCorrelation_Valid(float *srcMat, nSize srcSize, float *mapMat, nSize mapSize, float *dstMat, nSize dstSize)
{
	int i, j, c, r;
	float *pTmpData = NULL;
	nSize exSize = { 0, 0 };
	int halfmapsizew;
	int halfmapsizeh;
	cudaError cuRet = cudaSuccess;
	cudaError_t cudaStatus = cudaSuccess;

	if (!srcMat || !mapMat || !dstMat)
	{
		PRT_ERR("param error !\n");
		return;
	}

	if (mapSize.r % 2 == 0 && mapSize.c % 2 == 0)/* 模板大小为偶数 */
	{
		halfmapsizew = (mapSize.c) / 2; /* 卷积模块的半瓣大小 */
		halfmapsizeh = (mapSize.r) / 2;
	}
	else
	{
		halfmapsizew = (mapSize.c - 1) / 2; /* 卷积模块的半瓣大小 */
		halfmapsizeh = (mapSize.r - 1) / 2;
	}

	/* 这里先默认进行full模式的操作，full模式的输出大小为inSize+(mapSize-1) */
	int outSizeW = srcSize.c + (mapSize.c - 1); /* 这里的输出扩大一部分 */
	int outSizeH = srcSize.r + (mapSize.r - 1);
	nSize outSize = { outSizeW, outSizeH };

	float *pOutDataDev = NULL;
	cuRet = cudaMalloc((void**)&pOutDataDev, outSizeW*outSizeH*sizeof(float));
	RET_CHEAK_ZERO(cuRet);

	/* 这里先默认进行full模式的操作，full模式的输出大小为inSize+(mapSize-1) */
	/* 为了方便计算，将inputData扩大一圈 */
	exSize.c = srcSize.c + 2 * (mapSize.c - 1);
	exSize.r = srcSize.r + 2 * (mapSize.r - 1);


	float *pTmpDev = NULL;
	cuRet = cudaMalloc((void**)&pTmpDev, exSize.c*exSize.r*sizeof(float));
	RET_CHEAK_ZERO(cuRet);

	dim3 blk;
	dim3 gid;
	gid.x = 2;
	gid.y = 2;
	gid.z = 1;
		 
	blk.x = exSize.c/2;
	blk.y = exSize.r/2;
	blk.z = 1;
//	printf("x,y,z: %d-%d-%d\n", blk.x, blk.y, blk.z);
	cuMat2dEdgeExpand_kernel << <gid, blk >> >(srcMat, srcSize, pTmpDev, mapSize.c - 1, mapSize.r - 1);
	CUDA_STS_CHECK(cudaStatus);
	cudaThreadSynchronize();

	blk.x = outSizeW;
	blk.y = outSizeH;
//	printf("x,y,z: %d-%d-%d\n", blk.x, blk.y, blk.z);
	cuMat2DSubSum_kernel << <1, blk >> >(pTmpDev, exSize, mapMat, mapSize, pOutDataDev, outSizeW);
	CUDA_STS_CHECK(cudaStatus);
	cudaThreadSynchronize();

	blk.x = outSizeW;
	blk.y = outSizeH;
	if (mapSize.r % 2 == 0 && mapSize.c % 2 == 0)/* 模板大小为偶数 */
	{
		cuMat2dEdgeShrink_kernel << <1, blk >> >(pOutDataDev, outSize, dstMat, halfmapsizew * 2-1, halfmapsizeh * 2-1);
	}
	else
	{
		cuMat2dEdgeShrink_kernel << <1, blk >> >(pOutDataDev, outSize, dstMat, halfmapsizew * 2, halfmapsizeh * 2);
	}
	CUDA_STS_CHECK(cudaStatus);
	cudaThreadSynchronize();
	cudaFree(pTmpDev);
	cudaFree(pOutDataDev);
}

