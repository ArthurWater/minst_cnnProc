#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
/* #include <random.h> */
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "cnn.h"
#include "cuMat.h"
#include "cuCnn.h"


void cuCnnDestroy(CNN_NET_STR *stCnnNet)
{
	int i = 0;
	cudaError_t cuRet = cudaSuccess;
	CNN_OUT_LAYER_L5 *pOutLayer = NULL;
	CHEAK_POINT_NULL(stCnnNet);


	pOutLayer = &stCnnNet->stOutL5;
	cudaFree((void*)pOutLayer->vDev);
	cudaFree((void*)pOutLayer->basicDev);
	cudaFree((void*)pOutLayer->yDev);
	PRT("cnn memory released ok !\n");
}

/* CNN初始化函数 */
void cuCnnSetUp(CNN_NET_STR *stCnnNet)
{
	int i = 0;
	cudaError_t cuRet = cudaSuccess;
	CNN_POOLING_LAYER_L2 *pPoolingLayer2 = NULL;
	CNN_POOLING_LAYER_L4 *pPoolingLayer4 = NULL;
	CNN_OUT_LAYER_L5 *pOutLayer = NULL;
	if (!stCnnNet)
	{
		PRT_ERR("param error\n");
	}

	cuRet = cudaMalloc((void **)&stCnnNet->pInDataDev, CNN_LAYER1_IN_SIZE*CNN_LAYER1_IN_SIZE*sizeof(FLOAT));
	RET_CHEAK_ZERO(cuRet);

	stCnnNet->layerNum = CNN_LAYER_NUM;
	//########################################################################//
	stCnnNet->stCovL1.inChannels = CNN_LAYER1_IN_CHANNEL_NUM;
	stCnnNet->stCovL1.outChannels = CNN_LAYER1_OUT_CHANNEL_NUM;
	stCnnNet->stCovL1.isFullConnect = TRUE;
	stCnnNet->stCovL1.inputWidth = CNN_LAYER1_IN_SIZE;
	stCnnNet->stCovL1.inputHeight = CNN_LAYER1_IN_SIZE;
	stCnnNet->stCovL1.mapSize = CNN_LAYER1_MAP_SIZE;
	for (i = 0; i < CNN_LAYER1_OUT_CHANNEL_NUM; i++)
	{
		cuRet = cudaMalloc((void **)&stCnnNet->stCovL1.vDev[i], CNN_LAYER1_OUT_SIZE*CNN_LAYER1_OUT_SIZE*sizeof(FLOAT));
		RET_CHEAK_ZERO(cuRet);
		cuRet = cudaMalloc((void **)&stCnnNet->stCovL1.yDev[i], CNN_LAYER1_OUT_SIZE*CNN_LAYER1_OUT_SIZE*sizeof(FLOAT));
		RET_CHEAK_ZERO(cuRet);
		cuRet = cudaMalloc((void **)&stCnnNet->stCovL1.dDev[i], CNN_LAYER1_OUT_SIZE*CNN_LAYER1_OUT_SIZE*sizeof(FLOAT));
		RET_CHEAK_ZERO(cuRet);
	}
	cuRet = cudaMalloc((void **)&stCnnNet->stCovL1.mapOutDev, CNN_LAYER1_OUT_SIZE*CNN_LAYER1_OUT_SIZE*sizeof(FLOAT));
	RET_CHEAK_ZERO(cuRet);
	cuRet = cudaMalloc((void**)&stCnnNet->stCovL1.basicDev, CNN_LAYER1_OUT_CHANNEL_NUM*sizeof(FLOAT));
	RET_CHEAK_ZERO(cuRet);
	//########################################################################//
	pPoolingLayer2 = &stCnnNet->stPoolL2;
	stCnnNet->stPoolL2.inChannels = CNN_LAYER2_IN_CHANNEL_NUM;
	stCnnNet->stPoolL2.outChannels = CNN_LAYER2_IN_CHANNEL_NUM;
	stCnnNet->stPoolL2.inputWidth = CNN_LAYER1_OUT_SIZE;
	stCnnNet->stPoolL2.inputHeight = CNN_LAYER1_OUT_SIZE;
	stCnnNet->stPoolL2.mapSize = 2;
	stCnnNet->stPoolL2.poolType = CNN_AVE_POOL;
	cuRet = cudaMalloc((void **)&pPoolingLayer2->basicDev, CNN_LAYER2_OUT_CHANNEL_NUM*sizeof(FLOAT));
	RET_CHEAK_ZERO(cuRet);
	for (i = 0; i < CNN_LAYER2_OUT_CHANNEL_NUM; i++)
	{
		cuRet = cudaMalloc((void **)&pPoolingLayer2->yDev[i], CNN_LAYER2_OUT_SIZE*CNN_LAYER2_OUT_SIZE*sizeof(FLOAT));
		RET_CHEAK_ZERO(cuRet);
	}
	cuRet = cudaMalloc((void **)&pPoolingLayer2->dDev, CNN_LAYER2_OUT_SIZE*CNN_LAYER2_OUT_SIZE*CNN_LAYER2_OUT_CHANNEL_NUM*sizeof(FLOAT));
	RET_CHEAK_ZERO(cuRet);
	//########################################################################//
	stCnnNet->stCovL3.inChannels = CNN_LAYER3_IN_CHANNEL_NUM;
	stCnnNet->stCovL3.outChannels = CNN_LAYER3_OUT_CHANNEL_NUM;
	stCnnNet->stCovL3.isFullConnect = TRUE;
	stCnnNet->stCovL3.inputWidth = CNN_LAYER3_IN_SIZE;
	stCnnNet->stCovL3.inputHeight = CNN_LAYER3_IN_SIZE;
	stCnnNet->stCovL3.mapSize = CNN_LAYER3_MAP_SIZE;
	for (i = 0; i < CNN_LAYER3_OUT_CHANNEL_NUM; i++)
	{
		cuRet = cudaMalloc((void **)&stCnnNet->stCovL3.vDev[i], CNN_LAYER3_OUT_SIZE*CNN_LAYER3_OUT_SIZE*sizeof(FLOAT));
		RET_CHEAK_ZERO(cuRet);
		cuRet = cudaMalloc((void **)&stCnnNet->stCovL3.yDev[i], CNN_LAYER3_OUT_SIZE*CNN_LAYER3_OUT_SIZE*sizeof(FLOAT));
		RET_CHEAK_ZERO(cuRet);
		cuRet = cudaMalloc((void **)&stCnnNet->stCovL3.dDev[i], CNN_LAYER3_OUT_SIZE*CNN_LAYER3_OUT_SIZE*sizeof(FLOAT));
		RET_CHEAK_ZERO(cuRet);
	}
	cuRet = cudaMalloc((void **)&stCnnNet->stCovL3.mapOutDev, CNN_LAYER3_OUT_SIZE*CNN_LAYER3_OUT_SIZE*sizeof(FLOAT));
	RET_CHEAK_ZERO(cuRet);
	cuRet = cudaMalloc((void**)&stCnnNet->stCovL3.basicDev, CNN_LAYER3_OUT_CHANNEL_NUM*sizeof(FLOAT));
	RET_CHEAK_ZERO(cuRet);
	//########################################################################//
	pPoolingLayer4 = &stCnnNet->stPoolL4;
	stCnnNet->stPoolL4.inChannels = CNN_LAYER4_IN_CHANNEL_NUM;
	stCnnNet->stPoolL4.outChannels = CNN_LAYER4_IN_CHANNEL_NUM;
	stCnnNet->stPoolL4.inputWidth = CNN_LAYER3_OUT_SIZE;
	stCnnNet->stPoolL4.inputHeight = CNN_LAYER3_OUT_SIZE;
	stCnnNet->stPoolL4.mapSize = 2;
	stCnnNet->stPoolL4.poolType = CNN_AVE_POOL;


	cuRet = cudaMalloc((void **)&pPoolingLayer4->basicDev, CNN_LAYER4_OUT_CHANNEL_NUM*sizeof(FLOAT));
	RET_CHEAK_ZERO(cuRet);
	cuRet = cudaMalloc((void **)&pPoolingLayer4->yDev, CNN_LAYER4_OUT_SIZE*CNN_LAYER4_OUT_SIZE*CNN_LAYER4_OUT_CHANNEL_NUM*sizeof(FLOAT));
	RET_CHEAK_ZERO(cuRet);
	cuRet = cudaMalloc((void **)&pPoolingLayer4->dDev, CNN_LAYER4_OUT_SIZE*CNN_LAYER4_OUT_SIZE*CNN_LAYER4_OUT_CHANNEL_NUM*sizeof(FLOAT));
	RET_CHEAK_ZERO(cuRet);

	//########################################################################//
	stCnnNet->stOutL5.inputNum = CNN_LAYER5_IN_DATA_NUM;
	stCnnNet->stOutL5.outputNum = CNN_LAYER5_OUT_CHANNEL_NUM;
	stCnnNet->stOutL5.isFullConnect = TRUE;
	pOutLayer = &stCnnNet->stOutL5;
	cuRet = cudaMalloc((void**)&pOutLayer->inDev, CNN_LAYER5_IN_DATA_NUM*sizeof(FLOAT));
	RET_CHEAK_ZERO(cuRet);
	cuRet = cudaMalloc((void**)&pOutLayer->basicDev, CNN_LAYER5_OUT_CHANNEL_NUM*sizeof(FLOAT));
	RET_CHEAK_ZERO(cuRet);
	cuRet = cudaMalloc((void**)&pOutLayer->vDev, CNN_LAYER5_OUT_CHANNEL_NUM*sizeof(FLOAT));
	RET_CHEAK_ZERO(cuRet);
	cuRet = cudaMalloc((void**)&pOutLayer->yDev, CNN_LAYER5_OUT_CHANNEL_NUM*sizeof(FLOAT));
	RET_CHEAK_ZERO(cuRet);
	cuRet = cudaMalloc((void**)&pOutLayer->dDev, CNN_LAYER5_OUT_CHANNEL_NUM*sizeof(FLOAT));
	RET_CHEAK_ZERO(cuRet);

	cuRet = cudaMalloc((void**)&pOutLayer->wDev, CNN_LAYER5_OUT_CHANNEL_NUM*CNN_LAYER5_IN_DATA_NUM*sizeof(FLOAT));
	RET_CHEAK_ZERO(cuRet);
}

/*******************************************************************************
Function:		cuSigmaActiveFun
Description:
Input:
Output:		N/A
Return:		0:			Successful
ohters:		Failed
*******************************************************************************/
__global__ void cuSigmaActiveFun_kernel(float * input, float *bas, float *output) /* sigma激活函数 */
{	
	int i = threadIdx.x;
	output[i] = (float)1.0 / ((float)(1.0 + exp(-(input[i] + (float)*bas))));
}

__global__ void cuSigmaActiveL5_kernel(float * input, float *bas, float *output) /* sigma激活函数 */
{
	int i = threadIdx.x;
	output[i] = (float)1.0 / ((float)(1.0 + exp(-(input[i] + bas[i]))));
}

__global__ void cuAverageL2_kernel(float * input, float *output)
{
	FLOAT sum = 0.0;
	int w = CNN_LAYER2_OUT_SIZE;
	int mapSize = 2;
	int m, n;
	int i = threadIdx.x;
	int j = threadIdx.y;

	for (m = j * mapSize; m < (j+1) * mapSize; m++)
	{
		for (n = i * mapSize; n < (i+1) * mapSize; n++)
		{
			sum += input[m * w *mapSize + n];
		}
	}

	output[i+j*w] = sum / (float)(mapSize * mapSize * 1.0);
}
/*******************************************************************************
Function:		cuPoolingAverage
Description:  求平均值 
Input:
Output:		N/A
Return:		0:			Successful
ohters:		Failed
*******************************************************************************/
void cuL2PoolingAverage(float *output, nSize outputSize, float *input, nSize inputSize, int mapSize)
{
	int i, j, m, n;
	float sum = 0.0;
	int outputW = 0;
	int outputH = 0;
	CHEAK_POINT_NULL(input);
	CHEAK_POINT_NULL(output);
	CHEAK_VALUE_ZERO(mapSize);

	outputW = inputSize.c / mapSize;
	outputH = inputSize.r / mapSize;

	if (outputSize.c != outputW || outputSize.r != outputH)
	{
		PRT("ERROR: output size is wrong!!");
		return;
	}

	FLOAT * data1 = NULL;
	FLOAT * data2 = NULL;

	cudaMalloc((void**)&data1, (inputSize.c*inputSize.r)*sizeof(FLOAT));
	cudaMalloc((void**)&data2, (outputW*outputH)*sizeof(FLOAT));

	cudaMemcpy((void*)(data1), (void*)(input), (inputSize.c*inputSize.r)*sizeof(FLOAT), cudaMemcpyHostToDevice);
	dim3 dimBlock(CNN_LAYER2_OUT_SIZE, CNN_LAYER2_OUT_SIZE);
	cuAverageL2_kernel << <1, dimBlock >> >(data1, data2);
	cudaThreadSynchronize();

	cudaMemcpy((void*)(output), (void*)(data2), (CNN_LAYER2_OUT_SIZE*CNN_LAYER2_OUT_SIZE)*sizeof(FLOAT), cudaMemcpyDeviceToHost);
	cudaFree((void*)data1);
	cudaFree((void*)data2);
}

__global__ void cuAverageL4_kernel(float * input, float *output)
{
	FLOAT sum = 0.0;
	int w = CNN_LAYER4_OUT_SIZE;
	int mapSize = 2;
	int m, n;
	int i = threadIdx.x;
	int j = threadIdx.y;

	for (m = j * mapSize; m < (j + 1) * mapSize; m++)
	{
		for (n = i * mapSize; n < (i + 1) * mapSize; n++)
		{
			sum += input[m * w *mapSize + n];
		}
	}

	output[i + j*w] = sum / (float)(mapSize * mapSize * 1.0);
}

void cuL4PoolingAverage(float *output, nSize outputSize, float *input, nSize inputSize, int mapSize)
{
	int i, j, m, n;
	float sum = 0.0;
	int outputW = 0;
	int outputH = 0;
	CHEAK_POINT_NULL(input);
	CHEAK_POINT_NULL(output);
	CHEAK_VALUE_ZERO(mapSize);

	outputW = inputSize.c / mapSize;
	outputH = inputSize.r / mapSize;

	if (outputSize.c != outputW || outputSize.r != outputH)
	{
		PRT("ERROR: output size is wrong!!");
		return;
	}

	FLOAT * data1 = NULL;
	FLOAT * data2 = NULL;

	cudaMalloc((void**)&data1, (inputSize.c*inputSize.r)*sizeof(FLOAT));
	cudaMalloc((void**)&data2, (outputW*outputH)*sizeof(FLOAT));

	cudaMemcpy((void*)(data1), (void*)(input), (inputSize.c*inputSize.r)*sizeof(FLOAT), cudaMemcpyHostToDevice);
	dim3 dimBlock(CNN_LAYER4_OUT_SIZE, CNN_LAYER4_OUT_SIZE);
	cuAverageL4_kernel << <1, dimBlock >> >(data1, data2);
	cudaThreadSynchronize();

	cudaMemcpy((void*)(output), (void*)(data2), (CNN_LAYER4_OUT_SIZE*CNN_LAYER4_OUT_SIZE)*sizeof(FLOAT), cudaMemcpyDeviceToHost);
	cudaFree((void*)data1);
	cudaFree((void*)data2);
}

/*******************************************************************************
Function:		cuCnnTrainProc
Description:
Input:
Output:		N/A
Return:		0:			Successful
ohters:		Failed
*******************************************************************************/
void cuCnnTrainProc(CNN_NET_STR *pCnnNet, MinstImgArr *inputData, MinstLabelArr *outputData, CNNOpts opts, int trainNum)
{
	int e = 0;
	int i = 0;
	int j = 0;
	int k = 0;
	int n = 0;
	float iee = 0.0;
	char fileName[128] = { '\0' };
	FLOAT imgDataFloat[CNN_LAYER1_IN_SIZE][CNN_LAYER1_IN_SIZE] = { { 0.0 } };/* 灰度图 */

	if (!pCnnNet || !inputData || !outputData)
	{
		return;
	}

	for (e = 0; e < opts.numepochs; e++)
	{
		for (n = 0; n < trainNum; n++)
		{
			PRT("numepochs:%d, img:%d \n", e, n);
			for (j = 0; j < CNN_LAYER1_IN_SIZE; j++)
			{
				for (k = 0; k < CNN_LAYER1_IN_SIZE; k++)
				{
					imgDataFloat[j][k] = inputData->ImgPtr[n].ImgData[j][k];
				}
			}

			CnnForwardPass(pCnnNet, (FLOAT *)imgDataFloat);  /* 前向传播，这里主要计算各 */

			CnnBackPass(pCnnNet, outputData->LabelPtr[n].LabelData); /* 后向传播，这里主要计算各神经元的误差梯度 */

			/*            sprintf(fileName, "/mnt/hgfs/share/cnnDemo/PicTrans/CnnData1/%d.cnn", n); */
			/*            SaveCnnMidData(pCnnNet, fileName, inputData->ImgPtr[n].ImgData); */

			CnnApplyGrads(pCnnNet, opts, (FLOAT *)imgDataFloat); /* 更新权重 */

			CnnParamClear(pCnnNet);
			/* 计算并保存误差能量 */
			iee = 0.0;
			for (i = 0; i < pCnnNet->stOutL5.outputNum; i++)
			{
				iee = iee + pCnnNet->e[i] * pCnnNet->e[i];
			}

			if (n == 0)
				pCnnNet->L[n] = iee / (float)2.0;
			else
				pCnnNet->L[n] = pCnnNet->L[n - 1] * 0.99 + 0.01 * iee / (float)2.0;
		}
	}
}

/* 这里InputData是图像数据，inputData[r][c],r行c列，这里根各权重模板是一致的 */
void cuCnnff(CNN *cnn, float **inputData)
{
	int i, j, r, c;

	if (!cnn || !inputData)
	{
		return;
	}

	int outSizeW = cnn->S2->inputWidth;
	int outSizeH = cnn->S2->inputHeight;
	/* 第一层的传播 */

	/* 第一层输出数据 */
	nSize mapSize = { cnn->C1->mapSize, cnn->C1->mapSize };
	nSize inSize = { cnn->C1->inputWidth, cnn->C1->inputHeight };
	nSize outSize = { cnn->S2->inputWidth, cnn->S2->inputHeight };
	PRT("C1: insize[%d-%d], mapsize[%d-%d], outsize[%d-%d]\n", inSize.c, inSize.r, mapSize.c, mapSize.r, outSize.c, outSize.r);
	for (i = 0; i < (cnn->C1->outChannels); i++)
	{
		for (j = 0; j < (cnn->C1->inChannels); j++)
		{
			float **mapout = cov(cnn->C1->mapData[j][i], mapSize, inputData, inSize, COV_VALID);
			addmat(cnn->C1->v[i], cnn->C1->v[i], outSize, mapout, outSize);
			for (r = 0; r < outSize.r; r++)
			{
				free(mapout[r]);
			}

			free(mapout);
		}

		for (r = 0; r < outSize.r; r++)
		{
			for (c = 0; c < outSize.c; c++)
				cnn->C1->y[i][r][c] = activation_Sigma(cnn->C1->v[i][r][c], cnn->C1->basicData[i]);
		}
	}

	CovLayerPrint(cnn->C1);

	/* 第二层的输出传播S2，采样层 */
	outSize.c = cnn->C3->inputWidth;
	outSize.r = cnn->C3->inputHeight;
	inSize.c = cnn->S2->inputWidth;
	inSize.r = cnn->S2->inputHeight;
	PRT("S2: insize[%d-%d], outsize[%d-%d]\n", inSize.c, inSize.r, outSize.c, outSize.r);
	for (i = 0; i < (cnn->S2->outChannels); i++)
	{
		if (cnn->S2->poolType == CNN_AVE_POOL)
			avgPooling(cnn->S2->y[i], outSize, cnn->C1->y[i], inSize, cnn->S2->mapSize);
	}

	PoolLayerPrint(cnn->S2);
	/* 第三层输出传播,这里是全连接 */
	outSize.c = cnn->S4->inputWidth;
	outSize.r = cnn->S4->inputHeight;
	inSize.c = cnn->C3->inputWidth;
	inSize.r = cnn->C3->inputHeight;
	mapSize.c = cnn->C3->mapSize;
	mapSize.r = cnn->C3->mapSize;
	PRT("C3: insize[%d-%d], mapsize[%d-%d], outsize[%d-%d]\n", inSize.c, inSize.r, mapSize.c, mapSize.r, outSize.c, outSize.r);
	for (i = 0; i < (cnn->C3->outChannels); i++)
	{
		for (j = 0; j < (cnn->C3->inChannels); j++)
		{
			float **mapout = cov(cnn->C3->mapData[j][i], mapSize, cnn->S2->y[j], inSize, COV_VALID);
			addmat(cnn->C3->v[i], cnn->C3->v[i], outSize, mapout, outSize);
			for (r = 0; r < outSize.r; r++)
				free(mapout[r]);

			free(mapout);
		}

		for (r = 0; r < outSize.r; r++)
		for (c = 0; c < outSize.c; c++)
			cnn->C3->y[i][r][c] = activation_Sigma(cnn->C3->v[i][r][c], cnn->C3->basicData[i]);
	}

	CovLayerPrint(cnn->C3);

	/* 第四层的输出传播 */
	inSize.c = cnn->S4->inputWidth;
	inSize.r = cnn->S4->inputHeight;
	outSize.c = inSize.c / cnn->S4->mapSize;
	outSize.r = inSize.r / cnn->S4->mapSize;
	PRT("S4: insize[%d-%d], outsize[%d-%d]\n", inSize.c, inSize.r, outSize.c, outSize.r);
	for (i = 0; i < (cnn->S4->outChannels); i++)
	{
		if (cnn->S4->poolType == AvePool)
			avgPooling(cnn->S4->y[i], outSize, cnn->C3->y[i], inSize, cnn->S4->mapSize);
	}

	PoolLayerPrint(cnn->S4);
	/* 输出层O5的处理 */
	/* 首先需要将前面的多维输出展开成一维向量 */
	float *O5inData = (float *)malloc((cnn->O5->inputNum) * sizeof(float));
	for (i = 0; i < (cnn->S4->outChannels); i++)
	{
		for (r = 0; r < outSize.r; r++)
		{
			for (c = 0; c < outSize.c; c++)
				O5inData[i * outSize.r * outSize.c + r * outSize.c + c] = cnn->S4->y[i][r][c];
		}
	}

	nSize cnnL5nSize = { cnn->O5->inputNum, cnn->O5->outputNum };/* 192-10 */
	nnff(cnn->O5->v, O5inData, cnn->O5->wData, cnn->O5->basicData, cnnL5nSize);
	for (i = 0; i < cnn->O5->outputNum; i++)
	{
		cnn->O5->y[i] = activation_Sigma(cnn->O5->v[i], cnn->O5->basicData[i]);
	}

	free(O5inData);
	OutLayerPrint(cnn->O5);
}

/*******************************************************************************
Function:		cuNn2f_kernel
Description:
Input:
Output:		N/A
Return:		0:			Successful
ohters:		Failed
*******************************************************************************/
__global__ void cuNn2f_kernel(float *output, float *input, float *wdata, float *bas, nSize nnSize)
{
	int w = nnSize.c;
	int h = nnSize.r;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = 0, ty = 0;


	if (!output || !input || !wdata || !bas)
	{
		PRT_ERR("param error\n");
		return;
	}
	tx = threadIdx.x;
	ty = threadIdx.y;

	//input 1*192->192*1
	// w 10*192
	// output 10*1

	__shared__ float tmp[192];

	tmp[tx] = wdata[192 * bx + tx] * input[tx];
	__syncthreads();

	if (0 == tx)
	{
		output[bx] = 0.0;
		for (int i = 0; i < 192; i++)
		{
			output[bx] += tmp[i];
		}
		output[bx] += bas[bx];
	}
	__syncthreads();
}

/* 这里InputData是图像数据，inputData[r][c],r行c列，这里根各权重模板是一致的 */
void cuCnnForwardPass(CNN_NET_STR *pCnnNet, float *inputData)
{
	int i, j, r, c;
	int kr, kc;
	int idx = 0;
	FILE *fp = NULL;
	unsigned char imgName[128] = { '\0' };
	unsigned char imgData[1024] = { 0 };
	nSize dstSize = { 0, 0 };
	FLOAT mapDataRotL1[CNN_LAYER1_MAP_SIZE][CNN_LAYER1_MAP_SIZE] = { { 0.0 } };
	FLOAT mapDataRotL3[CNN_LAYER3_MAP_SIZE][CNN_LAYER3_MAP_SIZE] = { { 0.0 } };
	cudaError_t cuRet = cudaSuccess;
	CNN_OUT_LAYER_L5 *pOutLayer = NULL;
	if (!pCnnNet || !inputData)
	{
		return;
	}

	int outSizeW = pCnnNet->stPoolL2.inputWidth;
	int outSizeH = pCnnNet->stPoolL2.inputHeight;
	/* 第一层的传播 */

	/* 第一层输出数据 */
	nSize mapSize = { pCnnNet->stCovL1.mapSize, pCnnNet->stCovL1.mapSize };
	nSize inSize = { pCnnNet->stCovL1.inputWidth, pCnnNet->stCovL1.inputHeight };
	nSize outSize = { pCnnNet->stPoolL2.inputWidth, pCnnNet->stPoolL2.inputHeight };
	/*    PRT("L1: insize[%d-%d], mapsize[%d-%d], outsize[%d-%d]\n", inSize.c, inSize.r, mapSize.c, mapSize.r, outSize.c, outSize.r); */
	/*	PRT("L1:inChannels:%d, outChannels:%d, mapSize:%d\n", pCnnNet->stCovL1.inChannels, pCnnNet->stCovL1.outChannels, pCnnNet->stCovL1.mapSize); */
	cudaMemcpy((void*)(pCnnNet->stCovL1.basicDev), (void*)(pCnnNet->stCovL1.basicData), CNN_LAYER1_OUT_CHANNEL_NUM * sizeof(float), cudaMemcpyHostToDevice);
	for (i = 0; i < (pCnnNet->stCovL1.outChannels); i++)
	{
		for (j = 0; j < (pCnnNet->stCovL1.inChannels); j++)
		{

			/* 采用COV_VALID类型卷积 */
			if (mapSize.r % 2 == 0 && mapSize.c % 2 == 0)/* 偶数，一般 r c一样 */
			{
				dstSize.c = inSize.c + 3 - mapSize.c;
				dstSize.r = inSize.r + 3 - mapSize.r;
			}
			else/* 奇数 */
			{
				dstSize.c = inSize.c + 1 - mapSize.c;
				dstSize.r = inSize.r + 1 - mapSize.r;
			}

			dstSize.c = CNN_LAYER1_OUT_SIZE;
			dstSize.r = CNN_LAYER1_OUT_SIZE;
			/*			Mat2dRotate_180(pCnnNet->stCovL1.mapData[j][i], mapDataRotL1, mapSize); */
#if 0
			/* 映射矩阵旋转，将卷积变为相关计算 */
			for (kr = 0; kr < CNN_LAYER1_MAP_SIZE; kr++)
			{
				for (kc = 0; kc < CNN_LAYER1_MAP_SIZE; kc++)
				{
					mapDataRotL1[kr][kc] = pCnnNet->stCovL1.mapData[j][i][CNN_LAYER1_MAP_SIZE - kr - 1][CNN_LAYER1_MAP_SIZE - kc - 1];
				}
			}
			memset(pCnnNet->stCovL1.mapOut, 0, CNN_LAYER1_OUT_SIZE * CNN_LAYER1_OUT_SIZE * sizeof(float));
			Mat2dCorrelation_Valid(inputData, inSize, (FLOAT *)mapDataRotL1, mapSize, (FLOAT *)pCnnNet->stCovL1.mapOut, dstSize);
			Mat2D_Add((FLOAT *)pCnnNet->stCovL1.v[i], (FLOAT *)pCnnNet->stCovL1.v[i], (FLOAT *)pCnnNet->stCovL1.mapOut, dstSize);
#else
			float * mapDataRotL1Dev = NULL;
			float * mapDataDev = NULL;
			cuRet = cudaMalloc((void**)&mapDataRotL1Dev, CNN_LAYER1_MAP_SIZE*CNN_LAYER1_MAP_SIZE*sizeof(FLOAT));
			RET_CHEAK_ZERO(cuRet);
			cuRet = cudaMalloc((void **)&mapDataDev, CNN_LAYER1_MAP_SIZE*CNN_LAYER1_MAP_SIZE*sizeof(FLOAT));
			RET_CHEAK_ZERO(cuRet);

			cudaMemcpy((void*)mapDataDev, (void*)pCnnNet->stCovL1.mapData[j][i], CNN_LAYER1_MAP_SIZE*CNN_LAYER1_MAP_SIZE*sizeof(FLOAT), cudaMemcpyHostToDevice);
			cuMat2DRolate180_kernel << <CNN_LAYER1_OUT_SIZE, CNN_LAYER1_OUT_SIZE >> >(mapDataRotL1Dev, mapDataDev, CNN_LAYER1_MAP_SIZE);
			cudaThreadSynchronize();
			cudaMemcpy((void*)mapDataRotL1, (void*)mapDataRotL1Dev, CNN_LAYER1_MAP_SIZE*CNN_LAYER1_MAP_SIZE*sizeof(FLOAT), cudaMemcpyDeviceToHost);

			memset(pCnnNet->stCovL1.mapOut, 0, CNN_LAYER1_OUT_SIZE * CNN_LAYER1_OUT_SIZE * sizeof(float));
			cudaMemset(pCnnNet->stCovL1.mapOutDev, 0, CNN_LAYER1_OUT_SIZE * CNN_LAYER1_OUT_SIZE*sizeof(FLOAT));

			cudaMemcpy((void*)pCnnNet->pInDataDev, inputData, CNN_LAYER1_IN_SIZE*CNN_LAYER1_IN_SIZE*sizeof(FLOAT), cudaMemcpyHostToDevice);
			cuMat2dCorrelation_Valid(pCnnNet->pInDataDev, inSize, (FLOAT *)mapDataRotL1Dev, mapSize, (FLOAT *)pCnnNet->stCovL1.mapOutDev, dstSize);
			cudaMemcpy((void*)pCnnNet->stCovL1.mapOut, pCnnNet->stCovL1.mapOutDev, CNN_LAYER1_OUT_SIZE * CNN_LAYER1_OUT_SIZE*sizeof(FLOAT), cudaMemcpyDeviceToHost);
			cuRet = cudaFree(mapDataDev);
			RET_CHEAK_ZERO(cuRet);
			cudaFree(mapDataRotL1Dev);

			cudaMemcpy((void*)(pCnnNet->stCovL1.vDev[i]), (void*)(pCnnNet->stCovL1.v[i]), CNN_LAYER1_OUT_SIZE * CNN_LAYER1_OUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
			cuMat2DAdd_kernel << <CNN_LAYER1_OUT_SIZE, CNN_LAYER1_OUT_SIZE>> >(pCnnNet->stCovL1.vDev[i], pCnnNet->stCovL1.vDev[i], pCnnNet->stCovL1.mapOutDev, CNN_LAYER1_OUT_SIZE);
			cudaThreadSynchronize();
			cudaMemcpy((void*)(pCnnNet->stCovL1.v[i]), (void*)(pCnnNet->stCovL1.vDev[i]), CNN_LAYER1_OUT_SIZE * CNN_LAYER1_OUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
#endif
		}
#if 0
		for (r = 0; r < outSize.r; r++)
		{
			for (c = 0; c < outSize.c; c++)
			{
				pCnnNet->stCovL1.y[i][r][c] = SigmaActiveFun(pCnnNet->stCovL1.v[i][r][c], pCnnNet->stCovL1.basicData[i]);
			}
		}
#else		
		cuSigmaActiveFun_kernel << <1, CNN_LAYER1_OUT_SIZE * CNN_LAYER1_OUT_SIZE>> >(pCnnNet->stCovL1.vDev[i], (float*)&pCnnNet->stCovL1.basicDev[i], pCnnNet->stCovL1.yDev[i]);
		cudaThreadSynchronize();
		cudaMemcpy(pCnnNet->stCovL1.y[i], pCnnNet->stCovL1.yDev[i], CNN_LAYER1_OUT_SIZE * CNN_LAYER1_OUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
#endif
	}

	/*
	PRT("L1: output data ");
	for (i = 0; i < (pCnnNet->stCovL1.outChannels); i++){
	idx = 0;
	sprintf(imgName, "/mnt/hgfs/share/cnnDemo/output/imgL1_%d.yuv", i);
	fp = fopen(imgName, "wb");
	for (r = 0; r < CNN_LAYER1_OUT_SIZE; r++){
	for (c = 0; c < CNN_LAYER1_OUT_SIZE; c++){
	PRT("%f ", pCnnNet->stCovL1.y[i][r][c]);
	imgData[idx] = (unsigned char)(pCnnNet->stCovL1.y[i][r][c] * 255);
	idx++;
	}
	PRT("\n");
	}
	PRT("\n");

	fwrite(imgData, CNN_LAYER1_OUT_SIZE*CNN_LAYER1_OUT_SIZE, 1 ,fp);
	fclose(fp);
	}
	*/

	/* 第二层的输出传播S2，采样层 */
	inSize.c = pCnnNet->stPoolL2.inputWidth;
	inSize.r = pCnnNet->stPoolL2.inputHeight;
	outSize.c = pCnnNet->stCovL3.inputWidth;
	outSize.r = pCnnNet->stCovL3.inputHeight;
	/*    PRT("S2: insize[%d-%d], outsize[%d-%d]\n", inSize.c, inSize.r, outSize.c, outSize.r); */

	for (i = 0; i < (pCnnNet->stPoolL2.outChannels); i++)
	{
		if (pCnnNet->stPoolL2.poolType == CNN_AVE_POOL)
		{
#if 0
			PoolingAverage((FLOAT *)pCnnNet->stPoolL2.y[i], outSize, (FLOAT *)pCnnNet->stCovL1.y[i], inSize, pCnnNet->stPoolL2.mapSize);
#else
			cuL2PoolingAverage((FLOAT *)pCnnNet->stPoolL2.y[i], outSize, (FLOAT *)pCnnNet->stCovL1.y[i], inSize, pCnnNet->stPoolL2.mapSize);
#endif
		}
	}

	/*
	PRT("L2: output data \n");
	for (i = 0; i < (pCnnNet->stPoolL2.outChannels); i++){
	for (r = 0; r < CNN_LAYER2_OUT_SIZE; r++){
	for (c = 0; c < CNN_LAYER2_OUT_SIZE; c++){
	PRT("%f ", pCnnNet->stPoolL2.y[i][r][c]);
	}
	PRT("\n");
	}
	PRT("\n");
	}
	*/

	/* 第三层输出传播,这里是全连接 */
	inSize.c = pCnnNet->stCovL3.inputWidth;
	inSize.r = pCnnNet->stCovL3.inputHeight;
	mapSize.c = pCnnNet->stCovL3.mapSize;
	mapSize.r = pCnnNet->stCovL3.mapSize;
	outSize.c = pCnnNet->stPoolL4.inputWidth;
	outSize.r = pCnnNet->stPoolL4.inputHeight;
	/*    PRT("L3: insize[%d-%d], mapsize[%d-%d], outsize[%d-%d]\n", inSize.c, inSize.r, mapSize.c, mapSize.r, outSize.c, outSize.r); */
	/*	PRT("L3:inChannels:%d, outChannels:%d, mapSize:%d\n", pCnnNet->stCovL3.inChannels, pCnnNet->stCovL3.outChannels, pCnnNet->stCovL3.mapSize); */
	cudaMemcpy((void*)(pCnnNet->stCovL3.basicDev), (void*)(pCnnNet->stCovL3.basicData), CNN_LAYER3_OUT_CHANNEL_NUM * sizeof(float), cudaMemcpyHostToDevice);
	for (i = 0; i < (pCnnNet->stCovL3.outChannels); i++)
	{
		for (j = 0; j < (pCnnNet->stCovL3.inChannels); j++)
		{
			/* 采用COV_VALID类型卷积 */
			if (mapSize.r % 2 == 0 && mapSize.c % 2 == 0)/* 偶数，一般 r c一样 */
			{
				dstSize.c = inSize.c + 3 - mapSize.c;
				dstSize.r = inSize.r + 3 - mapSize.r;
			}
			else/* 奇数 */
			{
				dstSize.c = inSize.c + 1 - mapSize.c;
				dstSize.r = inSize.r + 1 - mapSize.r;
			}

			dstSize.c = CNN_LAYER3_OUT_SIZE;
			dstSize.r = CNN_LAYER3_OUT_SIZE;
#if 0
			/*			Mat2dRotate_180(pCnnNet->stCovL3.mapData[j][i], mapDataRotL3, mapSize); */
			for (kr = 0; kr < CNN_LAYER3_MAP_SIZE; kr++)
			{
				for (kc = 0; kc < CNN_LAYER3_MAP_SIZE; kc++)
				{
					mapDataRotL3[kr][kc] = pCnnNet->stCovL3.mapData[j][i][CNN_LAYER3_MAP_SIZE - kr - 1][CNN_LAYER3_MAP_SIZE - kc - 1];
				}
			}
			memset(pCnnNet->stCovL3.mapOut, 0, CNN_LAYER3_OUT_SIZE * CNN_LAYER3_OUT_SIZE * sizeof(float));
			Mat2dCorrelation_Valid((FLOAT *)pCnnNet->stPoolL2.y[j], inSize, (FLOAT *)mapDataRotL3, mapSize, (FLOAT *)pCnnNet->stCovL3.mapOut, dstSize);
			Mat2D_Add((FLOAT *)pCnnNet->stCovL3.v[i], (FLOAT *)pCnnNet->stCovL3.v[i], (FLOAT *)pCnnNet->stCovL3.mapOut, outSize);
#else
			float * mapDataRotL3Dev = NULL;
			float * mapDataL3Dev = NULL;
			cuRet = cudaMalloc((void**)&mapDataRotL3Dev, CNN_LAYER3_MAP_SIZE*CNN_LAYER3_MAP_SIZE*sizeof(FLOAT));
			RET_CHEAK_ZERO(cuRet);
			cuRet = cudaMalloc((void **)&mapDataL3Dev, CNN_LAYER3_MAP_SIZE*CNN_LAYER3_MAP_SIZE*sizeof(FLOAT));
			RET_CHEAK_ZERO(cuRet);

			cudaMemcpy((void*)mapDataL3Dev, (void*)pCnnNet->stCovL3.mapData[j][i], CNN_LAYER3_MAP_SIZE*CNN_LAYER3_MAP_SIZE*sizeof(FLOAT), cudaMemcpyHostToDevice);
			cuMat2DRolate180_kernel << <CNN_LAYER3_OUT_SIZE, CNN_LAYER3_OUT_SIZE >> >(mapDataRotL3Dev, mapDataL3Dev, CNN_LAYER3_MAP_SIZE);
			cudaThreadSynchronize();
			cudaMemcpy((void*)mapDataRotL3, (void*)mapDataRotL3Dev, CNN_LAYER3_MAP_SIZE*CNN_LAYER3_MAP_SIZE*sizeof(FLOAT), cudaMemcpyDeviceToHost);

			memset(pCnnNet->stCovL3.mapOut, 0, CNN_LAYER3_OUT_SIZE * CNN_LAYER3_OUT_SIZE * sizeof(float));
			cudaMemset(pCnnNet->stCovL3.mapOutDev, 0, CNN_LAYER3_OUT_SIZE * CNN_LAYER3_OUT_SIZE*sizeof(FLOAT));

			cudaMemcpy((void*)pCnnNet->stPoolL2.yDev[j], pCnnNet->stPoolL2.y[j], CNN_LAYER3_IN_SIZE*CNN_LAYER3_IN_SIZE*sizeof(FLOAT), cudaMemcpyHostToDevice);
			cuMat2dCorrelation_Valid(pCnnNet->stPoolL2.yDev[j], inSize, (FLOAT *)mapDataRotL3Dev, mapSize, (FLOAT *)pCnnNet->stCovL3.mapOutDev, dstSize);
			cudaMemcpy((void*)pCnnNet->stCovL3.mapOut, pCnnNet->stCovL3.mapOutDev, CNN_LAYER3_OUT_SIZE * CNN_LAYER3_OUT_SIZE*sizeof(FLOAT), cudaMemcpyDeviceToHost);

			cudaFree(mapDataL3Dev);
			cudaFree(mapDataRotL3Dev);

			cudaMemcpy((void*)(pCnnNet->stCovL3.vDev[i]), (void*)(pCnnNet->stCovL3.v[i]), CNN_LAYER3_OUT_SIZE * CNN_LAYER3_OUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
			cuMat2DAdd_kernel << <CNN_LAYER3_OUT_SIZE, CNN_LAYER3_OUT_SIZE >> >(pCnnNet->stCovL3.vDev[i], pCnnNet->stCovL3.vDev[i], pCnnNet->stCovL3.mapOutDev, CNN_LAYER3_OUT_SIZE);
			cudaThreadSynchronize();
			cudaMemcpy((void*)(pCnnNet->stCovL3.v[i]), (void*)(pCnnNet->stCovL3.vDev[i]), CNN_LAYER3_OUT_SIZE * CNN_LAYER3_OUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

			/*
			cudaMemcpy((void*)(pCnnNet->stCovL3.vDev[i]), (void*)(pCnnNet->stCovL3.v[i]), CNN_LAYER3_OUT_SIZE * CNN_LAYER3_OUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy((void*)(pCnnNet->stCovL3.mapOutDev), (void*)(pCnnNet->stCovL3.mapOut), CNN_LAYER3_OUT_SIZE * CNN_LAYER3_OUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
			cuMat2DAdd_kernel << <CNN_LAYER3_OUT_SIZE, CNN_LAYER3_OUT_SIZE >> >(pCnnNet->stCovL3.vDev[i], pCnnNet->stCovL3.vDev[i], pCnnNet->stCovL3.mapOutDev, CNN_LAYER3_OUT_SIZE);
			cudaThreadSynchronize();
			cudaMemcpy((void*)(pCnnNet->stCovL3.v[i]), (void*)(pCnnNet->stCovL3.vDev[i]), CNN_LAYER3_OUT_SIZE * CNN_LAYER3_OUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
			*/
#endif
		}
#if 0
		for (r = 0; r < outSize.r; r++)
		{
			for (c = 0; c < outSize.c; c++)
			{
				pCnnNet->stCovL3.y[i][r][c] = SigmaActiveFun(pCnnNet->stCovL3.v[i][r][c], pCnnNet->stCovL3.basicData[i]);
			}
		}
#else		
		cuSigmaActiveFun_kernel << <1, CNN_LAYER3_OUT_SIZE * CNN_LAYER3_OUT_SIZE >> >(pCnnNet->stCovL3.vDev[i], (float*)&pCnnNet->stCovL3.basicDev[i], pCnnNet->stCovL3.yDev[i]);
		cudaThreadSynchronize();
		cudaMemcpy(pCnnNet->stCovL3.y[i], pCnnNet->stCovL3.yDev[i], CNN_LAYER3_OUT_SIZE * CNN_LAYER3_OUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
#endif
	}

	/*
	PRT("L3: output data \n");
	for (i = 0; i < (pCnnNet->stCovL3.outChannels); i++){
	for (r = 0; r < CNN_LAYER3_OUT_SIZE; r++){
	for (c = 0; c < CNN_LAYER3_OUT_SIZE; c++){
	PRT("%f ", pCnnNet->stCovL3.y[i][r][c]);
	}
	PRT("\n");
	}
	PRT("\n");
	}
	*/

	/* 第四层的输出传播 */
	inSize.c = pCnnNet->stPoolL4.inputWidth;
	inSize.r = pCnnNet->stPoolL4.inputHeight;
	outSize.c = inSize.c / pCnnNet->stPoolL4.mapSize;
	outSize.r = inSize.r / pCnnNet->stPoolL4.mapSize;
	/*    PRT("S4: insize[%d-%d], outsize[%d-%d]\n", inSize.c, inSize.r, outSize.c, outSize.r); */
	for (i = 0; i < (pCnnNet->stPoolL4.outChannels); i++)
	{
		if (pCnnNet->stPoolL4.poolType == AvePool)
		{
#if 0
			PoolingAverage((FLOAT *)pCnnNet->stPoolL4.y[i], outSize, (FLOAT *)pCnnNet->stCovL3.y[i], inSize, pCnnNet->stPoolL4.mapSize);
#else
			cuL4PoolingAverage((FLOAT *)pCnnNet->stPoolL4.y[i], outSize, (FLOAT *)pCnnNet->stCovL3.y[i], inSize, pCnnNet->stPoolL4.mapSize);
#endif
		}
	}

	/* 输出层O5的处理
	1. 首先需要将前面的多维输出展开成一维向量
	*/
	pOutLayer = &pCnnNet->stOutL5;
	/*
	PRT("L4: output data \n");
	for (i = 0; i < (pCnnNet->stPoolL4.outChannels); i++){
	for (r = 0; r < outSize.r; r++){
	for (c = 0; c < outSize.c; c++){
	pCnnNet->stOutL5.inData[i*outSize.r*outSize.c + r*outSize.c + c] = pCnnNet->stPoolL4.y[i][r][c];
	PRT("%f ", pCnnNet->stPoolL4.y[i][r][c]);
	}
	}
	PRT("\n");
	}

	PRT("L5: inputNum-outputNum[%d-%d] \n", pCnnNet->stOutL5.inputNum, pCnnNet->stOutL5.outputNum);
	PRT("L5: input data \n");
	for(i = 0; i < CNN_LAYER5_IN_DATA_NUM; i++)
	{
	PRT("%f ", pCnnNet->stOutL5.inData[i]);
	if((i+1)%16 == 0)
	{
	PRT("\n");
	}
	}
	PRT("\n");
	*/
	nSize pCnnNetL5nSize = { pCnnNet->stOutL5.inputNum, pCnnNet->stOutL5.outputNum };/* (12*4*4)192-10 */
#if 0
	nn2f(pCnnNet->stOutL5.v, (FLOAT *)pCnnNet->stPoolL4.y, (FLOAT *)pCnnNet->stOutL5.wData, pCnnNet->stOutL5.basicData, pCnnNetL5nSize);
#else
	cudaMemcpy((void*)(pCnnNet->stPoolL4.yDev), (void*)(pCnnNet->stPoolL4.y), 12 * 4 * 4 * sizeof(FLOAT), cudaMemcpyHostToDevice);

	cudaMemcpy((void*)(pOutLayer->wDev), (void*)(pOutLayer->wData), 10 * 192 * sizeof(FLOAT), cudaMemcpyHostToDevice);
	cudaMemcpy((void*)(pOutLayer->basicDev), (void*)(pOutLayer->basicData), 10 * sizeof(FLOAT), cudaMemcpyHostToDevice);
	cuNn2f_kernel << <pOutLayer->outputNum, pOutLayer->inputNum >> >(pOutLayer->vDev, pCnnNet->stPoolL4.yDev, pOutLayer->wDev, pOutLayer->basicDev, pCnnNetL5nSize);
	cudaMemcpy((void*)(pCnnNet->stOutL5.v), (void*)(pOutLayer->vDev), 10 * sizeof(FLOAT), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
#endif
	/*
	for(i = 0; i < CNN_LAYER5_OUT_CHANNEL_NUM; i++)
	{
	PRT("%f ", pCnnNet->stOutL5.v[i]);
	}
	PRT("\n");
	*/
#if 0
	for (i = 0; i < pCnnNet->stOutL5.outputNum; i++)
	{
		pCnnNet->stOutL5.y[i] = SigmaActiveFun(pCnnNet->stOutL5.v[i], pCnnNet->stOutL5.basicData[i]);
	}
#else
	INT outputDataSize = pCnnNet->stOutL5.outputNum * sizeof(FLOAT);
//	cudaMemcpy((void*)(pOutLayer->vDev), (void*)(pCnnNet->stOutL5.v), outputDataSize, cudaMemcpyHostToDevice);
//	cudaMemcpy((void*)(pOutLayer->basicDev), (void*)(pCnnNet->stOutL5.basicData), outputDataSize, cudaMemcpyHostToDevice);
	cuSigmaActiveL5_kernel << <1, CNN_LAYER5_OUT_CHANNEL_NUM >> >(pOutLayer->vDev, pOutLayer->basicDev, pOutLayer->yDev);
	cudaThreadSynchronize();
	cudaMemcpy((void*)(pCnnNet->stOutL5.y), (void*)(pOutLayer->yDev), outputDataSize, cudaMemcpyDeviceToHost);
#endif
}

