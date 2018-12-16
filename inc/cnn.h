<<<<<<< HEAD
#ifndef __CNN_
#define __CNN_
=======
/******************************************************************************
Copyright 2018-2028 @
All Rights Reserved
FileName:    cnn.h
Description:
Author:		meimaokui@126.com
Date:		$(Time)
Modification History: <version>      <time>      <author>        <desc>
a)					  v1.0.0	   $(time)	  meimaokui@126.com	 Creat
******************************************************************************/
#ifndef _MMK_CNN_H_
#define _MMK_CNN_H_

#ifdef __cplusplus
extern "C" {
#endif
>>>>>>> init files

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
<<<<<<< HEAD
#include <random>
#include <time.h>
#include "mat.h"
=======
	//#include <random>
#include <time.h>
#include "mat.h"
#include "../inc/com_type_def.h"
#include "../inc/common.h"
>>>>>>> init files
#include "minst.h"

#define AvePool 0
#define MaxPool 1
#define MinPool 2

<<<<<<< HEAD
#define CNN_AVE_POOL 0
#define CNN_MAX_POOL 1
#define CNN_MIN_POOL 2

#define CNN_DATA_FILE_DIR "D:\\Embedded Work\\Machine Learning\\CNN\\demoCnn\\PicTrans\\CNNData\\"

// 卷积层
typedef struct convolutional_layer{
	int inputWidth;   //输入图像的宽
	int inputHeight;  //输入图像的长
	int mapSize;      //特征模板的大小，模板一般都是正方形

	int inChannels;   //输入图像的数目
	int outChannels;  //输出图像的数目

	/* 
	关于特征模板的权重分布，这里是一个四维数组,其大小为inChannels*outChannels*mapSize*mapSize大小
	这里用四维数组，主要是为了表现全连接的形式，实际上卷积层并没有用到全连接的形式
	这里的例子是DeapLearningToolboox里的CNN例子，其用到就是全连接
	*/
	float**** mapData;     //存放特征模块的数据
	float**** dmapData;    //存放特征模块的数据的局部梯度

	float* basicData;   //偏置，偏置的大小，为outChannels
	bool isFullConnect; //是否为全连接
	bool* connectModel; //连接模式（默认为全连接）

	// 下面三者的大小同输出的维度相同
	float*** v; // 进入激活函数的输入值
	float*** y; // 激活函数后神经元的输出

	// 输出像素的局部梯度
	float*** d; // 网络的局部梯度,δ值  
}CovLayer;

// 采样层 pooling
typedef struct pooling_layer{
	int inputWidth;   //输入图像的宽
	int inputHeight;  //输入图像的长
	int mapSize;      //特征模板的大小

	int inChannels;   //输入图像的数目
	int outChannels;  //输出图像的数目

	int poolType;     //Pooling的方法
	float* basicData;   //偏置

	float*** y; // 采样函数后神经元的输出,无激活函数
	float*** d; // 网络的局部梯度,δ值
}PoolLayer;

// 输出层 全连接的神经网络
typedef struct nn_layer{
	int 	inputNum;   //输入数据的数目
	int 	outputNum;  //输出数据的数目

	float** wData; // 权重数据，为一个inputNum*outputNum大小
	float* 	basicData;   //偏置，大小为outputNum大小

	// 下面三者的大小同输出的维度相同
	float* 	v; // 进入激活函数的输入值
	float* 	y; // 激活函数后神经元的输出
	float* 	d; // 网络的局部梯度,δ值

	bool 	isFullConnect; //是否为全连接
}OutLayer;

typedef struct cnn_network{
	int 		layerNum;
	CovLayer* 	C1;
	PoolLayer* 	S2;
	CovLayer* 	C3;
	PoolLayer* 	S4;
	OutLayer* 	O5;

	float* 		e; // 训练误差
	float* 		L; // 瞬时误差能量
}CNN;

typedef struct _CNN_NET_STR_{
	int 		layerNum;
	int 		res[3];
	CovLayer 	C1;
	PoolLayer 	S2;
	CovLayer 	C3;
	PoolLayer 	S4;
	OutLayer 	O5;

	float* 		e; // 训练误差
	float* 		L; // 瞬时误差能量
}CNN_NET_STR;


typedef struct train_opts{
	int 		numepochs; // 训练的迭代次数
	float 		alpha; // 学习速率
}CNNOpts;

void cnnsetup(CNN* cnn,nSize inputSize,int outputSize);
/*	
	CNN网络的训练函数
	inputData，outputData分别存入训练数据
	dataNum表明数据数目
*/
void cnnTrain(CNN* cnn,	ImgArr inputData,LabelArr outputData,CNNOpts opts,int trainNum);
// 测试cnn函数
float cnnTest(CNN* cnn, ImgArr inputData, LabelArr outputData, unsigned int testNum);
// 保存cnn
void savecnn(CNN* cnn, const char* filename);
// 导入cnn的数据
int importcnn(CNN* cnn, const char* filename);

// 初始化卷积层
CovLayer* initCovLayer(int inputWidth,int inputHeight,int mapSize,int inChannels,int outChannels);
void CovLayerConnect(CovLayer* covL,bool* connectModel);
// 初始化采样层
PoolLayer* initPoolLayer(int inputWidth,int inputHeigh,int mapSize,int inChannels,int outChannels,int poolType);
void PoolLayerConnect(PoolLayer* poolL,bool* connectModel);
// 初始化输出层
OutLayer* initOutLayer(int inputNum,int outputNum);

// 激活函数 input是数据，inputNum说明数据数目，bas表明偏置
float activation_Sigma(float input,float bas); // sigma激活函数
float CnnActiv_Sigma(float input, float bas);

void cnnff(CNN* cnn,float** inputData); // 网络的前向传播
void cnnbp(CNN* cnn,float* outputData); // 网络的后向传播
void cnnapplygrads(CNN* cnn,CNNOpts opts,float** inputData);
void cnnclear(CNN* cnn); // 将数据vyd清零

/*
=======
#define CNN_AVE_POOL	0
#define CNN_MAX_POOL	1
#define CNN_MIN_POOL	2

#define CNN_LAYER_NUM 5

#define CNN_LAYER1_IN_SIZE			28
#define CNN_MAP_SIZE				5
#define CNN_LAYER1_MAP_SIZE			5
#define CNN_LAYER1_IN_CHANNEL_NUM	1
#define CNN_LAYER1_OUT_CHANNEL_NUM	6
#define CNN_LAYER1_OUT_SIZE			24

#define CNN_LAYER2_IN_CHANNEL_NUM 6
#define CNN_LAYER2_OUT_CHANNEL_NUM 6
#define CNN_LAYER2_OUT_SIZE 12


#define CNN_LAYER3_IN_SIZE 12
#define CNN_LAYER3_MAP_SIZE 5
#define CNN_LAYER3_IN_CHANNEL_NUM 6
#define CNN_LAYER3_OUT_CHANNEL_NUM 12
#define CNN_LAYER3_OUT_SIZE 8

#define CNN_LAYER4_IN_CHANNEL_NUM 12
#define CNN_LAYER4_OUT_CHANNEL_NUM 12
#define CNN_LAYER4_OUT_SIZE 4

#define CNN_LAYER5_IN_CHANNEL_NUM 12
#define CNN_LAYER5_OUT_CHANNEL_NUM 10
#define CNN_LAYER5_IN_DATA_NUM (CNN_LAYER5_IN_CHANNEL_NUM * CNN_LAYER4_OUT_SIZE * CNN_LAYER4_OUT_SIZE)

#define CNN_DATA_FILE_DIR "/mnt/hgfs/share/cnnDemo/PicTrans/CNNData"


	/* 卷积层 */
	typedef struct convolutional_layer
	{
		int inputWidth;   /* 输入图像的宽 */
		int inputHeight;  /* 输入图像的长 */
		int mapSize;      /* 特征模板的大小，模板一般都是正方形 */

		int inChannels;   /* 输入图像的数目 */
		int outChannels;  /* 输出图像的数目 */

		/*
		关于特征模板的权重分布，这里是一个四维数组,其大小为inChannels*outChannels*mapSize*mapSize大小
		这里用四维数组，主要是为了表现全连接的形式，实际上卷积层并没有用到全连接的形式
		这里的例子是DeapLearningToolboox里的CNN例子，其用到就是全连接
		*/
		float ****mapData;     /* 存放特征模块的数据 */
		float ****dmapData;    /* 存放特征模块的数据的局部梯度 */

		float *basicData;   /* 偏置，偏置的大小，为outChannels */
		BOOL isFullConnect; /* 是否为全连接 */
		BOOL *connectModel; /* 连接模式（默认为全连接） */

		/* 下面三者的大小同输出的维度相同 */
		float ***v; /* 进入激活函数的输入值 */
		float ***y; /* 激活函数后神经元的输出 */

		/* 输出像素的局部梯度 */
		float ***d; /* 网络的局部梯度,δ值 */
	} CovLayer;

	// 卷积层
	typedef struct _CNN_COV_LAYER_L1_{
		int inputWidth;   //输入图像的宽 28
		int inputHeight;  //输入图像的长 28
		int mapSize;      //特征模板的大小，模板一般都是正方形 5*5

		int inChannels;   //输入图像的数目 1
		int outChannels;  //输出图像的数目 6

		BOOL isFullConnect; //是否为全连接
		BOOL* connectModel; //连接模式（默认为全连接）

		/*
		关于特征模板的权重分布，这里是一个四维数组,其大小为inChannels*outChannels*mapSize*mapSize大小
		这里用四维数组，主要是为了表现全连接的形式，实际上卷积层并没有用到全连接的形式
		这里的例子是DeapLearningToolboox里的CNN例子，其用到就是全连接
		*/
		//存放特征模块的数据
		float mapData[CNN_LAYER1_IN_CHANNEL_NUM][CNN_LAYER1_OUT_CHANNEL_NUM][CNN_LAYER1_MAP_SIZE][CNN_LAYER1_MAP_SIZE];
		//存放特征模块的数据的局部梯度
		float dMapData[CNN_LAYER1_IN_CHANNEL_NUM][CNN_LAYER1_OUT_CHANNEL_NUM][CNN_LAYER1_MAP_SIZE][CNN_LAYER1_MAP_SIZE];
		//偏置，偏置的大小，为outChannels
		float basicData[CNN_LAYER1_OUT_CHANNEL_NUM];

		// 下面三者的大小同输出的维度相同
		// 进入激活函数的输入值
		float v[CNN_LAYER1_OUT_CHANNEL_NUM][CNN_LAYER1_OUT_SIZE][CNN_LAYER1_OUT_SIZE];
		// 激活函数后神经元的输出
		float y[CNN_LAYER1_OUT_CHANNEL_NUM][CNN_LAYER1_OUT_SIZE][CNN_LAYER1_OUT_SIZE];
		// 输出像素的局部梯度, 网络的局部梯度,δ值  
		float d[CNN_LAYER1_OUT_CHANNEL_NUM][CNN_LAYER1_OUT_SIZE][CNN_LAYER1_OUT_SIZE];

		float mapOut[CNN_LAYER1_OUT_SIZE][CNN_LAYER1_OUT_SIZE]; //存放卷积输出临时数据

		//GPU
		float *vDev[CNN_LAYER1_OUT_CHANNEL_NUM];
		float *yDev[CNN_LAYER1_OUT_CHANNEL_NUM];
		float *dDev[CNN_LAYER1_OUT_CHANNEL_NUM];
		float *mapOutDev;
		float* basicDev;
	}CNN_COV_LAYER_L1;

	// 采样层 pooling 池化
	typedef struct _CNN_POOLING_LAYER_L2_{
		int inputWidth;   //输入图像的宽 24
		int inputHeight;  //输入图像的长 24
		int mapSize;      //特征模板的大小 2*2

		int inChannels;   //输入图像的数目 6
		int outChannels;  //输出图像的数目 6

		int poolType;     //Pooling的方法
		float basicData[CNN_LAYER2_OUT_CHANNEL_NUM];   //偏置

		float y[CNN_LAYER2_OUT_CHANNEL_NUM][CNN_LAYER2_OUT_SIZE][CNN_LAYER2_OUT_SIZE]; // 采样函数后神经元的输出,无激活函数
		float d[CNN_LAYER2_OUT_CHANNEL_NUM][CNN_LAYER2_OUT_SIZE][CNN_LAYER2_OUT_SIZE]; // 网络的局部梯度,δ值

		//##############################GPU###########################################//
		float * basicDev;
		float * yDev[CNN_LAYER2_OUT_CHANNEL_NUM];
		float * dDev;

	}CNN_POOLING_LAYER_L2;

	typedef struct _CNN_COV_LAYER_L3_{
		int inputWidth;   //输入图像的宽 12
		int inputHeight;  //输入图像的长 12
		int mapSize;      //特征模板的大小，模板一般都是正方形 5*5

		int inChannels;   //输入图像的数目 6
		int outChannels;  //输出图像的数目 12

		BOOL isFullConnect; //是否为全连接
		BOOL* connectModel; //连接模式（默认为全连接）
		//##############################CPU###########################################//
		/*
		关于特征模板的权重分布，这里是一个四维数组,其大小为inChannels*outChannels*mapSize*mapSize大小
		这里用四维数组，主要是为了表现全连接的形式，实际上卷积层并没有用到全连接的形式
		这里的例子是DeapLearningToolboox里的CNN例子，其用到就是全连接
		*/
		//存放特征模块的数据
		float mapData[CNN_LAYER3_IN_CHANNEL_NUM][CNN_LAYER3_OUT_CHANNEL_NUM][CNN_LAYER3_MAP_SIZE][CNN_LAYER3_MAP_SIZE];
		//存放特征模块的数据的局部梯度
		float dMapData[CNN_LAYER3_IN_CHANNEL_NUM][CNN_LAYER3_OUT_CHANNEL_NUM][CNN_LAYER3_MAP_SIZE][CNN_LAYER3_MAP_SIZE];
		//偏置，偏置的大小，为outChannels
		float basicData[CNN_LAYER3_OUT_CHANNEL_NUM];

		// 下面三者的大小同输出的维度相同
		// 进入激活函数的输入值
		float v[CNN_LAYER3_OUT_CHANNEL_NUM][CNN_LAYER3_OUT_SIZE][CNN_LAYER3_OUT_SIZE];
		// 激活函数后神经元的输出
		float y[CNN_LAYER3_OUT_CHANNEL_NUM][CNN_LAYER3_OUT_SIZE][CNN_LAYER3_OUT_SIZE];
		// 输出像素的局部梯度, 网络的局部梯度,δ值  
		float d[CNN_LAYER3_OUT_CHANNEL_NUM][CNN_LAYER3_OUT_SIZE][CNN_LAYER3_OUT_SIZE];

		float mapOut[CNN_LAYER3_OUT_SIZE][CNN_LAYER3_OUT_SIZE]; //存放卷积输出临时数据

		//##############################GPU###########################################//
		float *	basicDev;   //偏置，大小为outputNum大小
		float *mapOutDev;
		float *vDev[CNN_LAYER3_OUT_CHANNEL_NUM];
		float *yDev[CNN_LAYER3_OUT_CHANNEL_NUM];
		float *dDev[CNN_LAYER3_OUT_CHANNEL_NUM];


	}CNN_COV_LAYER_L3;


	// 采样层 pooling 池化
	typedef struct _CNN_POOLING_LAYER_L4_{
		int inputWidth;   //输入图像的宽 8
		int inputHeight;  //输入图像的长 8
		int mapSize;      //特征模板的大小 2*2

		int inChannels;   //输入图像的数目 12
		int outChannels;  //输出图像的数目 12

		int poolType;     //Pooling的方法
		//##############################CPU###########################################//
		float basicData[CNN_LAYER4_OUT_CHANNEL_NUM];   //偏置

		float y[CNN_LAYER4_OUT_CHANNEL_NUM][CNN_LAYER4_OUT_SIZE][CNN_LAYER4_OUT_SIZE]; // 采样函数后神经元的输出,无激活函数
		float d[CNN_LAYER4_OUT_CHANNEL_NUM][CNN_LAYER4_OUT_SIZE][CNN_LAYER4_OUT_SIZE]; // 网络的局部梯度,δ值

		//##############################GPU###########################################//
		float * basicDev;
		float * yDev;
		float * dDev;

	}CNN_POOLING_LAYER_L4;

	// 采样层 pooling
	typedef struct pooling_layer{
		int inputWidth;   //输入图像的宽
		int inputHeight;  //输入图像的长
		int mapSize;      //特征模板的大小

		int inChannels;   /* 输入图像的数目 */
		int outChannels;  /* 输出图像的数目 */

		int poolType;     /* Pooling的方法 */
		float *basicData;   /* 偏置 */

		float ***y; /* 采样函数后神经元的输出,无激活函数 */
		float ***d; /* 网络的局部梯度,δ值 */
	} PoolLayer;

	/* 输出层 全连接的神经网络 */
	typedef struct nn_layer
	{
		int	inputNum;       /* 输入数据的数目 */
		int	outputNum;      /* 输出数据的数目 */

		float **wData; /* 权重数据，为一个inputNum*outputNum大小 */
		float *basicData;    /* 偏置，大小为outputNum大小 */

		/* 下面三者的大小同输出的维度相同 */
		float *v;  /* 进入激活函数的输入值 */
		float *y;  /* 激活函数后神经元的输出 */
		float *d;  /* 网络的局部梯度,δ值 */

		BOOL isFullConnect;    /* 是否为全连接 */
	} OutLayer;

	// 输出层 全连接的神经网络
	typedef struct _CNN_OUT_LAYER_L5_{
		int 	inputNum;   //输入数据的数目 192
		int 	outputNum;  //输出数据的数目 10
		BOOL 	isFullConnect; //是否为全连接
		//##############################CPU###########################################//
		float	inData[CNN_LAYER5_IN_DATA_NUM];//前面数据转为一维数据
		float 	wData[CNN_LAYER5_OUT_CHANNEL_NUM][CNN_LAYER5_IN_DATA_NUM]; // 权重数据，为一个inputNum*outputNum大小
		float 	basicData[CNN_LAYER5_OUT_CHANNEL_NUM];   //偏置，大小为outputNum大小

		// 下面三者的大小同输出的维度相同
		float 	v[CNN_LAYER5_OUT_CHANNEL_NUM]; // 进入激活函数的输入值
		float 	y[CNN_LAYER5_OUT_CHANNEL_NUM]; // 激活函数后神经元的输出
		float 	d[CNN_LAYER5_OUT_CHANNEL_NUM]; // 网络的局部梯度,δ值

		//##############################GPU###########################################//
		float * inDev;//前面数据转为一维数据
		float * wDev;
		float *	basicDev;   //偏置，大小为outputNum大小
		float *	vDev; // 进入激活函数的输入值
		float *	yDev; // 激活函数后神经元的输出
		float *	dDev; // 网络的局部梯度,δ值	
	}CNN_OUT_LAYER_L5;


	typedef struct cnn_net{
		int 		layerNum;
		CovLayer* 	C1;
		PoolLayer* 	S2;
		CovLayer* 	C3;
		PoolLayer* 	S4;
		OutLayer* 	O5;

		float *e;      /* 训练误差 */
		float *L;      /* 瞬时误差能量 */
	} CNN;

	typedef struct _CNN_NET_STR_{
		int 		layerNum;
		float * pInDataDev;
		int 		res[2];
		CNN_COV_LAYER_L1 		stCovL1;
		CNN_POOLING_LAYER_L2 	stPoolL2;
		CNN_COV_LAYER_L3		stCovL3;
		CNN_POOLING_LAYER_L4	stPoolL4;
		CNN_OUT_LAYER_L5		stOutL5;

		float e[CNN_LAYER5_OUT_CHANNEL_NUM];      /* 训练误差 */
		float L[CNN_PIC_TRAIN_NUM];      /* 瞬时误差能量 */
	} CNN_NET_STR;


	typedef struct train_opts
	{
		int	numepochs;         /* 训练的迭代次数 */
		float alpha;       /* 学习速率 */
	} CNNOpts;

	void cnnsetup(CNN *cnn, nSize inputSize, int outputSize);
	void CnnSetUp(CNN_NET_STR* stCnnNet);
	/*******************************************************************************
	Function:		CnnTrainProc
	Description:
	Input:
	Output:		N/A
	Return:		0:			Successful
	ohters:		Failed
	*******************************************************************************/
	void CnnTrainProc(CNN_NET_STR *pCnnNet, MinstImgArr *inputData, MinstLabelArr *outputData, CNNOpts opts, int trainNum);
	/*
	CNN网络的训练函数
	inputData，outputData分别存入训练数据
	dataNum表明数据数目
	*/
	void cnnTrain(CNN *cnn, MinstImgArr* inputData, MinstLabelArr* outputData, CNNOpts opts, int trainNum);

	/* 测试cnn函数 */
	float cnnTest(CNN *cnn, MinstImgArr* inputData, MinstLabelArr* outputData, unsigned int testNum);

	/* 保存cnn */
	void savecnn(CNN *cnn, const char *filename);
	/* 导入cnn的数据 */
	int importcnn(CNN *cnn, const char *filename);


	void SaveCnnModelFile(CNN_NET_STR* pCnnNet, const char* fileName);



	/* 初始化卷积层 */
	CovLayer *initCovLayer(int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels);
	void CovLayerConnect(CovLayer *covL, BOOL *connectModel);

	/* 初始化采样层 */
	PoolLayer *initPoolLayer(int inputWidth, int inputHeigh, int mapSize, int inChannels, int outChannels, int poolType);
	void PoolLayerConnect(PoolLayer *poolL, BOOL *connectModel);

	/* 初始化输出层 */
	OutLayer *initOutLayer(int inputNum, int outputNum);

	/* 激活函数 input是数据，inputNum说明数据数目，bas表明偏置 */
	float activation_Sigma(float input, float bas); /* sigma激活函数 */
	float SigmaActiveFun(float input, float bas); // sigma激活函数


	void cnnff(CNN *cnn, float **inputData); /* 网络的前向传播 */
	void CnnForwardPass(CNN_NET_STR* pCnnNet, float* inputData);
	void CnnBackPass(CNN_NET_STR *pCnnNet, float *outputData); /* 网络的后向传播 */

	void cnnbp(CNN *cnn, float *outputData); /* 网络的后向传播 */
	void cnnapplygrads(CNN *cnn, CNNOpts opts, float **inputData);
	void CnnApplyGrads(CNN_NET_STR *pCnnNet, CNNOpts opts, float *inputData); /* 更新权重 */

	void cnnclear(CNN *cnn); /* 将数据vyd清零 */
	/*******************************************************************************
	Function:		CnnParamClear
	Description:
	Input:
	Output:		N/A
	Return:		0:			Successful
	ohters:		Failed
	*******************************************************************************/
	void CnnParamClear(CNN_NET_STR *pCnnNet);

	/*
>>>>>>> init files
	Pooling Function
	input 输入数据
	inputNum 输入数据数目
	mapSize 求平均的模块区域
<<<<<<< HEAD
*/
void avgPooling(float** output,nSize outputSize,float** input,nSize inputSize,int mapSize); // 求平均值

/* 
	单层全连接神经网络的处理
	nnSize是网络的大小
*/
void nnff(float* output,float* input,float** wdata,float* bas,nSize nnSize); // 单层全连接神经网络的前向传播

void saveCnnData(CNN* cnn, const char* filename, float** inputdata); // 保存CNN网络中的相关数据

void CNNOptsPrint(CNNOpts *pCNNOpts);
void CovLayerPrint(CovLayer *pCovLayer);
void PoolLayerPrint(PoolLayer *pPoolLayer);
void OutLayerPrint(OutLayer *pOutLayer);
=======
	*/
	void avgPooling(float **output, nSize outputSize, float **input, nSize inputSize, int mapSize); /* 求平均值 */
	void PoolingAverage(float* output, nSize outputSize, float* input, nSize inputSize, int mapSize);

	/*
	单层全连接神经网络的处理
	nnSize是网络的大小
	*/
	void nnff(float *output, float *input, float **wdata, float *bas, nSize nnSize); /* 单层全连接神经网络的前向传播 */
	void nn2f(float* output, float* input, float* wdata, float* bas, nSize nnSize);

	void saveCnnData(CNN *cnn, const char *filename, float **inputdata); /* 保存CNN网络中的相关数据 */
	void SaveCnnMidData(CNN_NET_STR *pCnnNet, const char *filename, float **inputdata);

	/*******************************************************************************
	Function:		CNNOptsPrint
	Description:
	Input:
	Output:		N/A
	Return:		0:			Successful
	ohters:		Failed
	*******************************************************************************/
	void CNNOptsPrint(CNNOpts *pCNNOpts);

	void CovLayerPrint(CovLayer *pCovLayer);
	void PoolLayerPrint(PoolLayer *pPoolLayer);
	void OutLayerPrint(OutLayer *pOutLayer);
	INT32 vecMaxIndex(PFLOAT vec, UINT32 veclength);

#ifdef __cplusplus
}
#endif
>>>>>>> init files

#endif
