/*

meimaokui@126.com
*/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "inc/com_type_def.h"
#include "inc/common.h"
#include "inc/cnn.h"
#include "inc/minst.h"
#include "inc/cnn_inference.h"
#include "inc/cuCnn.h"

void GetGpuDeviceInfo();

//global variable 
CNN_NET_STR stCnn;
CNN_NET_STR stCnnTrain;



int main()
{
	int i = 0;
	int j = 0;
	int k = 0;
	int idx = 0;
	int iRet = 0;
	FILE *fp = NULL;
	int maxIndex = 0;
	int iLabelIndex = 0;
	int incorrectnum = 0;  /* 错误预测的数目 */
	CHAR imgName[128] = { '\0' };
	UNCHAR imgData[CNN_LAYER1_IN_SIZE][CNN_LAYER1_IN_SIZE] = { { 0 } };/* 灰度图 */
	FLOAT imgDataFloat[CNN_LAYER1_IN_SIZE][CNN_LAYER1_IN_SIZE] = { { 0.0 } };/* 灰度图 */
	MinstImgArr testImg;
	MinstLabelArr testLabel;
	
	/* CNN结构的初始化 */
	memset(&stCnn, 0, sizeof(stCnn));
	memset(&stCnnTrain, 0, sizeof(stCnnTrain));
	memset(&testImg, 0, sizeof(MinstImgArr));
	memset(&testLabel, 0, sizeof(MinstLabelArr));

	//输出GPU信息
	GetGpuDeviceInfo();
	PRT("LeNet-5 run ......\n");
	minstReadLable(&testLabel, CNN_TEST_LABELS_PATH);
	ReadMinstImg(&testImg, CNN_TEST_IMAGES_PATH);

	nSize inputSize = { testImg.ImgPtr[0].c, testImg.ImgPtr[0].r };
	INT32 outSize = testLabel.LabelPtr[0].len;

	PRT("input c:%d, r:%d, outSize:%d\n", inputSize.c, inputSize.r, outSize);/* 28,28,10 */
	PRT("ImgNum:%d\n", testImg.ImgNum);

	PRT("loadind model file ...\n");
#if 0
	CnnSetUp(&stCnn);
#else
	cuCnnSetUp(&stCnn);
#endif

	iRet = ImportCnnModelFile(&stCnn, CNN_MODEL_FILE_SAVE_PATH);
	if (iRet)
	{
		PRT_ERR("import cnn error !\n");
		system("pause");
		return -1;
	}

	for (idx = 0; idx < PIC_TEST_NUM; idx++)
	{
		sprintf(imgName, "./output/img_%d.yuv", idx);
		fp = fopen(imgName, "wb");
		for (j = 0; j < CNN_LAYER1_IN_SIZE; j++)
		{
			for (k = 0; k < CNN_LAYER1_IN_SIZE; k++)
			{
				imgDataFloat[j][k] = testImg.ImgPtr[idx].ImgData[j][k];
				imgData[j][k] = (unsigned char)(testImg.ImgPtr[idx].ImgData[j][k] * 255);
			}
		}

		fwrite(imgData, CNN_LAYER1_IN_SIZE * CNN_LAYER1_IN_SIZE, 1, fp);
		fclose(fp);
#if 0
		CnnForwardPass(&stCnn, (FLOAT *)imgDataFloat);
#else
		cuCnnForwardPass(&stCnn, (FLOAT *)imgDataFloat);
#endif
		maxIndex = vecMaxIndex(stCnn.stOutL5.y, stCnn.stOutL5.outputNum);
		iLabelIndex = vecMaxIndex(testLabel.LabelPtr[idx].LabelData, stCnn.stOutL5.outputNum);

		/*
		PRT("Test:");
		for(i = 0; i < CNN_LAYER5_OUT_CHANNEL_NUM; i++)
		{
		PRT("[%d]:%f ", i, stCnn.stOutL5.y[i]);
		}
		PRT("\n");

		PRT("Real:");
		for(i = 0; i < CNN_LAYER5_OUT_CHANNEL_NUM; i++)
		{
		PRT("[%d]:%f ", i, testLabel.LabelPtr[n].LabelData[i]);
		}
		PRT("\n");
		*/
		if (maxIndex != iLabelIndex)
		{
			incorrectnum++;
		}

		PRT("outputNum:%d, testIndex:%d, realIndex:%d\n", stCnn.stOutL5.outputNum, maxIndex, iLabelIndex);

		CnnParamClear(&stCnn);
	}

	PRT("incorrectnum/totalNUmbet: %d/%d\n", incorrectnum, PIC_TEST_NUM);
	cuCnnDestroy(&stCnn);
	system("pause");
    return 0;
}




void GetGpuDeviceInfo()
{
	int deviceCount = 0;
	int dev = 0;
	int driverVersion = 0; 
	int runtimeVersion = 0;
	cudaDeviceProp deviceProp;

	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
	if (error_id != cudaSuccess) {
		printf("cudaGetDeviceCount returned %d\n-> %s\n",(int)error_id, cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
		exit(EXIT_FAILURE);
	}

	if (deviceCount == 0) {
		PRT("There are no available device(s) that support CUDA\n");
	}
	else {
		PRT("Detected %d CUDA Capable device(s)\n", deviceCount);
	}

	cudaSetDevice(dev);
	cudaGetDeviceProperties(&deviceProp, dev);
	PRT("Device %d: \"%s\"\n", dev, deviceProp.name);
	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);
	PRT(" CUDA Driver Version / Runtime Version %d.%d / %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);
	PRT(" CUDA Capability Major/Minor version number: %d.%d\n", deviceProp.major, deviceProp.minor);
	PRT(" Total amount of global memory: %.2f GBytes (%llu bytes)\n", (float)deviceProp.totalGlobalMem / (pow(1024.0, 3)), (unsigned long long) deviceProp.totalGlobalMem);
	PRT(" GPU Clock rate: %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
	PRT(" Memory Clock rate: %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
	PRT(" Memory Bus Width: %d-bit\n", deviceProp.memoryBusWidth);

	if (deviceProp.l2CacheSize) {
		PRT(" L2 Cache Size: %d bytes\n",
			deviceProp.l2CacheSize);
	}

	PRT(" Max Texture Dimension Size (x,y,z) 1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
		deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
		deviceProp.maxTexture2D[1],
		deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1],
		deviceProp.maxTexture3D[2]);

	PRT(" Max Layered Texture Size (dim) x layers 1D=(%d) x %d, 2D=(%d,%d) x %d\n",
		deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
		deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
		deviceProp.maxTexture2DLayered[2]);

	PRT(" Total amount of constant memory: %lu bytes\n", deviceProp.totalConstMem);
	PRT(" Total amount of shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);
	PRT(" Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
	PRT(" Warp size: %d\n", deviceProp.warpSize);
	PRT(" Maximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
	PRT(" Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
	PRT(" Maximum sizes of each dimension of a block: %d x %d x %d\n",
		deviceProp.maxThreadsDim[0],
		deviceProp.maxThreadsDim[1],
		deviceProp.maxThreadsDim[2]);
	PRT(" Maximum sizes of each dimension of a grid: %d x %d x %d\n",
		deviceProp.maxGridSize[0],
		deviceProp.maxGridSize[1],
		deviceProp.maxGridSize[2]);

	PRT(" Maximum memory pitch: %lu bytes\n", deviceProp.memPitch);
}

