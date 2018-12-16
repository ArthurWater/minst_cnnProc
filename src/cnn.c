/******************************************************************************
   Copyright 2018-2028 @
   All Rights Reserved
   FileName:    cnn.c
   Description:
   Author:		meimaokui@126.com
   Date:		$(Time)
   Modification History: <version>      <time>      <author>        <desc>
   a)					  v1.0.0	   $(time)	  meimaokui@126.com	 Creat
******************************************************************************/
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
/* #include <random.h> */
#include <time.h>
#include "../inc/cnn.h"

/*******************************************************************************
   Function:		cnnsetup
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
void cnnsetup(CNN *cnn, nSize inputSize, int outputSize)
{
    nSize inSize;
    int mapSize = CNN_MAP_SIZE;

    if (!cnn)
    {
        PRT_ERR("error\n");
    }

    cnn->layerNum = CNN_LAYER_NUM;

    inSize.c = inputSize.c; /* 28 */
    inSize.r = inputSize.r; /* 28 */
    cnn->C1 = initCovLayer(inSize.c, inSize.r, CNN_LAYER1_MAP_SIZE, 1, 6);

    inSize.c = inSize.c - mapSize + 1;
    inSize.r = inSize.r - mapSize + 1;
    cnn->S2 = initPoolLayer(inSize.c, inSize.r, 2, 6, 6, AvePool);

    inSize.c = inSize.c / 2;
    inSize.r = inSize.r / 2;
    cnn->C3 = initCovLayer(inSize.c, inSize.r, 5, 6, 12);

    inSize.c = inSize.c - mapSize + 1;
    inSize.r = inSize.r - mapSize + 1;
    cnn->S4 = initPoolLayer(inSize.c, inSize.r, 2, 12, 12, AvePool);

    inSize.c = inSize.c / 2;
    inSize.r = inSize.r / 2;
    cnn->O5 = initOutLayer(inSize.c * inSize.r * 12, outputSize);

    cnn->e = (float *)calloc(cnn->O5->outputNum, sizeof(float));
    PRT("outputNum:%d\n", cnn->O5->outputNum);
    if (!cnn->e)
    {
        PRT_ERR("calloc failed\n");
    }
}

/* CNN初始化函数 */
void CnnSetUp(CNN_NET_STR *stCnnNet)
{
    if (!stCnnNet)
    {
        PRT_ERR("param error\n");
    }

    stCnnNet->layerNum = CNN_LAYER_NUM;

    stCnnNet->stCovL1.inChannels = CNN_LAYER1_IN_CHANNEL_NUM;
    stCnnNet->stCovL1.outChannels = CNN_LAYER1_OUT_CHANNEL_NUM;
    stCnnNet->stCovL1.isFullConnect = TRUE;
    stCnnNet->stCovL1.inputWidth = CNN_LAYER1_IN_SIZE;
    stCnnNet->stCovL1.inputHeight = CNN_LAYER1_IN_SIZE;
    stCnnNet->stCovL1.mapSize = CNN_LAYER1_MAP_SIZE;

    stCnnNet->stPoolL2.inChannels = CNN_LAYER2_IN_CHANNEL_NUM;
    stCnnNet->stPoolL2.outChannels = CNN_LAYER2_IN_CHANNEL_NUM;
    stCnnNet->stPoolL2.inputWidth = CNN_LAYER1_OUT_SIZE;
    stCnnNet->stPoolL2.inputHeight = CNN_LAYER1_OUT_SIZE;
    stCnnNet->stPoolL2.mapSize = 2;
    stCnnNet->stPoolL2.poolType = CNN_AVE_POOL;

    stCnnNet->stCovL3.inChannels = CNN_LAYER3_IN_CHANNEL_NUM;
    stCnnNet->stCovL3.outChannels = CNN_LAYER3_OUT_CHANNEL_NUM;
    stCnnNet->stCovL3.isFullConnect = TRUE;
    stCnnNet->stCovL3.inputWidth = CNN_LAYER3_IN_SIZE;
    stCnnNet->stCovL3.inputHeight = CNN_LAYER3_IN_SIZE;
    stCnnNet->stCovL3.mapSize = CNN_LAYER3_MAP_SIZE;

    stCnnNet->stPoolL4.inChannels = CNN_LAYER4_IN_CHANNEL_NUM;
    stCnnNet->stPoolL4.outChannels = CNN_LAYER4_IN_CHANNEL_NUM;
    stCnnNet->stPoolL4.inputWidth = CNN_LAYER3_OUT_SIZE;
    stCnnNet->stPoolL4.inputHeight = CNN_LAYER3_OUT_SIZE;
    stCnnNet->stPoolL4.mapSize = 2;
    stCnnNet->stPoolL4.poolType = CNN_AVE_POOL;

    stCnnNet->stOutL5.inputNum = CNN_LAYER5_IN_DATA_NUM;
    stCnnNet->stOutL5.outputNum = CNN_LAYER5_OUT_CHANNEL_NUM;
    stCnnNet->stOutL5.isFullConnect = TRUE;

}

/* 初始化卷积层 */
CovLayer *initCovLayer(int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels)
{
    int i, j, c, r;
    int outW = 0;
    int outH = 0;
    float randnum = 1.0;
    CovLayer *covL = (CovLayer *)malloc(sizeof(CovLayer));

    if (!covL)
    {
        return NULL;
    }

    covL->inputHeight = inputHeight;
    covL->inputWidth = inputWidth;
    covL->mapSize = mapSize;
    covL->inChannels = inChannels;
    covL->outChannels = outChannels;

    covL->isFullConnect = TRUE; /* 默认为全连接 */

    /* 权重空间的初始化，先行再列调用，[r][c] */
    srand((unsigned)time(NULL));
    covL->mapData = (float ****)malloc(inChannels * sizeof(float ***));
    for (i = 0; i < inChannels; i++)
    {
        covL->mapData[i] = (float ***)malloc(outChannels * sizeof(float **));
        for (j = 0; j < outChannels; j++)
        {
            covL->mapData[i][j] = (float **)malloc(mapSize * sizeof(float *));
            for (r = 0; r < mapSize; r++)
            {
                covL->mapData[i][j][r] = (float *)malloc(mapSize * sizeof(float));
                for (c = 0; c < mapSize; c++)
                {
                    randnum = (((float)rand() / (float)RAND_MAX) - 0.5) * 2;
                    covL->mapData[i][j][r][c] = randnum * sqrt((float)6.0 / (float)(mapSize * mapSize * (inChannels + outChannels)));
                }
            }
        }
    }

    /* 权重梯度变化 */
    covL->dmapData = (float ****)malloc(inChannels * sizeof(float ***));
    for (i = 0; i < inChannels; i++)
    {
        covL->dmapData[i] = (float ***)malloc(outChannels * sizeof(float **));
        for (j = 0; j < outChannels; j++)
        {
            covL->dmapData[i][j] = (float **)malloc(mapSize * sizeof(float *));
            for (r = 0; r < mapSize; r++)
            {
                covL->dmapData[i][j][r] = (float *)calloc(mapSize, sizeof(float));
            }
        }
    }

    covL->basicData = (float *)calloc(outChannels, sizeof(float));

    outW = inputWidth - mapSize + 1;
    outH = inputHeight - mapSize + 1;

    covL->d = (float ***)malloc(outChannels * sizeof(float **));
    covL->v = (float ***)malloc(outChannels * sizeof(float **));
    covL->y = (float ***)malloc(outChannels * sizeof(float **));
    for (j = 0; j < outChannels; j++)
    {
        covL->d[j] = (float **)malloc(outH * sizeof(float *));
        covL->v[j] = (float **)malloc(outH * sizeof(float *));
        covL->y[j] = (float **)malloc(outH * sizeof(float *));
        for (r = 0; r < outH; r++)
        {
            covL->d[j][r] = (float *)calloc(outW, sizeof(float));
            covL->v[j][r] = (float *)calloc(outW, sizeof(float));
            covL->y[j][r] = (float *)calloc(outW, sizeof(float));
        }
    }

    return covL;
}

/*******************************************************************************
   Function:		initPoolLayer
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
PoolLayer *initPoolLayer(int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels, int poolType)
{
    int j, r;
    PoolLayer *poolL = (PoolLayer *)malloc(sizeof(PoolLayer));

    poolL->inputHeight = inputHeight;
    poolL->inputWidth = inputWidth;
    poolL->mapSize = mapSize;
    poolL->inChannels = inChannels;
    poolL->outChannels = outChannels;
    poolL->poolType = poolType;

    poolL->basicData = (float *)calloc(outChannels, sizeof(float));

    int outW = inputWidth / mapSize;
    int outH = inputHeight / mapSize;


    poolL->d = (float ***)malloc(outChannels * sizeof(float **));
    poolL->y = (float ***)malloc(outChannels * sizeof(float **));
    for (j = 0; j < outChannels; j++)
    {
        poolL->d[j] = (float **)malloc(outH * sizeof(float *));
        poolL->y[j] = (float **)malloc(outH * sizeof(float *));
        for (r = 0; r < outH; r++)
        {
            poolL->d[j][r] = (float *)calloc(outW, sizeof(float));
            poolL->y[j][r] = (float *)calloc(outW, sizeof(float));
        }
    }

    return poolL;
}

/*******************************************************************************
   Function:		initOutLayer
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
OutLayer *initOutLayer(int inputNum, int outputNum)
{
    OutLayer *outL = (OutLayer *)malloc(sizeof(OutLayer));

    outL->inputNum = inputNum;
    outL->outputNum = outputNum;


    outL->basicData = (float *)calloc(outputNum, sizeof(float));

    outL->d = (float *)calloc(outputNum, sizeof(float));
    outL->v = (float *)calloc(outputNum, sizeof(float));
    outL->y = (float *)calloc(outputNum, sizeof(float));

    /* 权重的初始化 */
    outL->wData = (float **)malloc(outputNum * sizeof(float *)); /* 输入行，输出列 */
    int i, j;
    srand((unsigned)time(NULL));
    for (i = 0; i < outputNum; i++)
    {
        outL->wData[i] = (float *)malloc(inputNum * sizeof(float));
        for (j = 0; j < inputNum; j++)
        {
            float randnum = (((float)rand() / (float)RAND_MAX) - 0.5) * 2; /* 产生一个-1到1的随机数 */
            outL->wData[i][j] = randnum * sqrt((float)6.0 / (float)(inputNum + outputNum));
        }
    }

    outL->isFullConnect = TRUE;

    return outL;
}

/* 返回向量最大数的序号 */
INT32 vecMaxIndex(PFLOAT vec, UINT32 veclength)
{
    UINT32 i = 0;
    FLOAT maxData = -1.0;
    UINT32 maxIndex = 0;

    for (i = 0; i < veclength; i++)
    {
        if (maxData < vec[i])
        {
            maxData = vec[i];
            maxIndex = i;
        }
    }

    return maxIndex;
}

/* 测试cnn函数 */
float cnnTest(CNN *cnn, MinstImgArr *inputData, MinstLabelArr *outputData, unsigned int testNum)
{
    int n = 0;
    int maxIndex = 0;
    int iLabelIndex = 0;
    int incorrectnum = 0;  /* 错误预测的数目 */

    if (0 == testNum || !cnn)
    {
        PRT_ERR("param error\n");
        return -1;
    }

    for (n = 0; n < testNum; n++)
    {
        cnnff(cnn, inputData->ImgPtr[n].ImgData);
        maxIndex = vecMaxIndex(cnn->O5->y, cnn->O5->outputNum);
        iLabelIndex = vecMaxIndex(outputData->LabelPtr[n].LabelData, cnn->O5->outputNum);
        if (maxIndex != iLabelIndex)
        {
            incorrectnum++;
        }

        PRT("outputNum:%d, index[%d-%d]:%f\n", cnn->O5->outputNum, maxIndex, iLabelIndex, cnn->O5->y[maxIndex]);
        cnnclear(cnn);
    }

    return (float)incorrectnum / (float)testNum;
}

/* 保存cnn */
void savecnn(CNN *cnn, const char *filename)
{
    FILE *fp = NULL;

    fp = fopen(filename, "wb");
    if (fp == NULL)
        printf("%s[%d]write file failed\n", __FILE__, __LINE__);

    int i, j, r;
    /* C1网络的数据 */
    for (i = 0; i < cnn->C1->inChannels; i++)
        for (j = 0; j < cnn->C1->outChannels; j++)
            for (r = 0; r < cnn->C1->mapSize; r++)
                fwrite(cnn->C1->mapData[i][j][r], sizeof(float), cnn->C1->mapSize, fp);

    fwrite(cnn->C1->basicData, sizeof(float), cnn->C1->outChannels, fp);

    /* C3网络 */
    for (i = 0; i < cnn->C3->inChannels; i++)
        for (j = 0; j < cnn->C3->outChannels; j++)
            for (r = 0; r < cnn->C3->mapSize; r++)
                fwrite(cnn->C3->mapData[i][j][r], sizeof(float), cnn->C3->mapSize, fp);

    fwrite(cnn->C3->basicData, sizeof(float), cnn->C3->outChannels, fp);

    /* O5输出层 */
    for (i = 0; i < cnn->O5->outputNum; i++)
        fwrite(cnn->O5->wData[i], sizeof(float), cnn->O5->inputNum, fp);

    fwrite(cnn->O5->basicData, sizeof(float), cnn->O5->outputNum, fp);

    fclose(fp);
}

/* 导入cnn的数据 */
int importcnn(CNN *cnn, const char *filename)
{
    int i, j, c, r;
    float inData = 0.0;
    FILE *fp = NULL;

    if (!cnn || !filename)
    {
        return -1;
    }

    fp = fopen(filename, "rb");
    if (fp == NULL)
    {
        PRT_ERR("write file failed\n");
        return -1;
    }

    /* C1的数据 */
    PRT("C1->inChannels:%d, C1->outChannels:%d, C1->mapSize:%d\n", cnn->C1->inChannels, cnn->C1->outChannels, cnn->C1->mapSize);
    for (i = 0; i < cnn->C1->inChannels; i++)
    {
        for (j = 0; j < cnn->C1->outChannels; j++)
            for (r = 0; r < cnn->C1->mapSize; r++)
                for (c = 0; c < cnn->C1->mapSize; c++)
                {
        #if 0
                    float *in = (float *)malloc(sizeof(float));
                    fread(in, sizeof(float), 1, fp);
                    cnn->C1->mapData[i][j][r][c] = *in;
        #else
                    fread(&inData, sizeof(float), 1, fp);
                    cnn->C1->mapData[i][j][r][c] = inData;
        #endif
                }
    }

    for (i = 0; i < cnn->C1->outChannels; i++)
    {
        fread(&cnn->C1->basicData[i], sizeof(float), 1, fp);
        PRT("%f ", cnn->C1->basicData[i]);
    }

    PRT("\n");


    /* C3网络 */
    PRT("C3->inChannels:%d, C3->outChannels:%d, C3->mapSize:%d\n", cnn->C3->inChannels, cnn->C3->outChannels, cnn->C3->mapSize);
    for (i = 0; i < cnn->C3->inChannels; i++)
        for (j = 0; j < cnn->C3->outChannels; j++)
            for (r = 0; r < cnn->C3->mapSize; r++)
                for (c = 0; c < cnn->C3->mapSize; c++)
                {
                    fread(&cnn->C3->mapData[i][j][r][c], sizeof(float), 1, fp);
                }

    for (i = 0; i < cnn->C3->outChannels; i++)
        fread(&cnn->C3->basicData[i], sizeof(float), 1, fp);

    /* O5输出层 */
    PRT("O5->inputNum:%d, O5->outputNum:%d\n", cnn->O5->inputNum, cnn->O5->outputNum);
    for (i = 0; i < cnn->O5->outputNum; i++)
        for (j = 0; j < cnn->O5->inputNum; j++)
            fread(&cnn->O5->wData[i][j], sizeof(float), 1, fp);

    PRT("O5->basicData:");
    for (i = 0; i < cnn->O5->outputNum; i++)
    {
        fread(&cnn->O5->basicData[i], sizeof(float), 1, fp);
        PRT(" %f", cnn->O5->basicData[i]);
    }

    PRT("\n");

    fclose(fp);
    return 0;
}

/*******************************************************************************
   Function:		SaveCnnModelFile
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
void SaveCnnModelFile(CNN_NET_STR *pCnnNet, const char *fileName)
{
    int i, j, r;
    FILE *fp = NULL;

    if (!pCnnNet || !fileName)
    {
        return;
    }

    fp = fopen(fileName, "wb");
    if (fp == NULL)
    {
        PRT_ERR("write file failed\n");
    }

    /* C1网络的数据 */
    for (i = 0; i < pCnnNet->stCovL1.inChannels; i++)
    {
        for (j = 0; j < pCnnNet->stCovL1.outChannels; j++)
        {
            for (r = 0; r < pCnnNet->stCovL1.mapSize; r++)
            {
                fwrite(pCnnNet->stCovL1.mapData[i][j][r], sizeof(FLOAT), pCnnNet->stCovL1.mapSize, fp);
            }
        }
    }

    fwrite(pCnnNet->stCovL1.basicData, sizeof(FLOAT), pCnnNet->stCovL1.outChannels, fp);

    /* C3网络 */
    for (i = 0; i < pCnnNet->stCovL3.inChannels; i++)
    {
        for (j = 0; j < pCnnNet->stCovL3.outChannels; j++)
        {
            for (r = 0; r < pCnnNet->stCovL3.mapSize; r++)
            {
                fwrite(pCnnNet->stCovL3.mapData[i][j][r], sizeof(FLOAT), pCnnNet->stCovL3.mapSize, fp);
            }
        }
    }

    fwrite(pCnnNet->stCovL3.basicData, sizeof(FLOAT), pCnnNet->stCovL3.outChannels, fp);

    /* O5输出层 */
    for (i = 0; i < pCnnNet->stOutL5.outputNum; i++)
    {
        fwrite(pCnnNet->stOutL5.wData[i], sizeof(FLOAT), pCnnNet->stOutL5.inputNum, fp);
    }

    fwrite(pCnnNet->stOutL5.basicData, sizeof(FLOAT), pCnnNet->stOutL5.outputNum, fp);

    fclose(fp);
}

/*******************************************************************************
   Function:		cnnTrain
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
void cnnTrain(CNN *cnn,	MinstImgArr *inputData, MinstLabelArr *outputData, CNNOpts opts, int trainNum)
{
    int e = 0;
    int i = 0;
    int n = 0;
    float iee = 0.0;

    /* 学习训练误差曲线 */
    cnn->L = (float *)malloc(trainNum * sizeof(float));
    for (e = 0; e < opts.numepochs; e++)
    {
        for (n = 0; n < trainNum; n++)
        {
            /* printf("%d\n",n); */
            cnnff(cnn, inputData->ImgPtr[n].ImgData);  /* 前向传播，这里主要计算各 */

            cnnbp(cnn, outputData->LabelPtr[n].LabelData); /* 后向传播，这里主要计算各神经元的误差梯度 */

            const char *filename = combine_strings(CNN_DATA_FILE_DIR, combine_strings(intTochar(n), ".cnn"));
            saveCnnData(cnn, filename, inputData->ImgPtr[n].ImgData);

            cnnapplygrads(cnn, opts, inputData->ImgPtr[n].ImgData); /* 更新权重 */

            cnnclear(cnn);
            /* 计算并保存误差能量 */
            iee = 0.0;
            for (i = 0; i < cnn->O5->outputNum; i++)
                iee = iee + cnn->e[i] * cnn->e[i];

            if (n == 0)
                cnn->L[n] = iee / (float)2.0;
            else
                cnn->L[n] = cnn->L[n - 1] * 0.99 + 0.01 * iee / (float)2.0;
        }
    }
}

/*******************************************************************************
   Function:		CnnTrainProc
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
void CnnTrainProc(CNN_NET_STR *pCnnNet,	MinstImgArr *inputData, MinstLabelArr *outputData, CNNOpts opts, int trainNum)
{
    int e = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    int n = 0;
    float iee = 0.0;
    char fileName[128] = {'\0'};
    FLOAT imgDataFloat[CNN_LAYER1_IN_SIZE][CNN_LAYER1_IN_SIZE] = {{0.0}};/* 灰度图 */

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
void cnnff(CNN *cnn, float **inputData)
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
    nSize mapSize = {cnn->C1->mapSize, cnn->C1->mapSize};
    nSize inSize = {cnn->C1->inputWidth, cnn->C1->inputHeight};
    nSize outSize = {cnn->S2->inputWidth, cnn->S2->inputHeight};
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

    nSize cnnL5nSize = {cnn->O5->inputNum, cnn->O5->outputNum};/* 192-10 */
    nnff(cnn->O5->v, O5inData, cnn->O5->wData, cnn->O5->basicData, cnnL5nSize);
    for (i = 0; i < cnn->O5->outputNum; i++)
    {
        cnn->O5->y[i] = activation_Sigma(cnn->O5->v[i], cnn->O5->basicData[i]);
    }

    free(O5inData);
    OutLayerPrint(cnn->O5);
}

/* 这里InputData是图像数据，inputData[r][c],r行c列，这里根各权重模板是一致的 */
void CnnForwardPass(CNN_NET_STR *pCnnNet, float *inputData)
{
    int i, j, r, c;
    int kr, kc;
    int idx = 0;
    FILE *fp = NULL;
    unsigned char imgName[128] = {'\0'};
    unsigned char imgData[1024] = {0};
    nSize dstSize = {0, 0};
    FLOAT mapDataRotL1[CNN_LAYER1_MAP_SIZE][CNN_LAYER1_MAP_SIZE] = {{0.0}};
    FLOAT mapDataRotL3[CNN_LAYER3_MAP_SIZE][CNN_LAYER3_MAP_SIZE] = {{0.0}};

    if (!pCnnNet || !inputData)
    {
        return;
    }

    int outSizeW = pCnnNet->stPoolL2.inputWidth;
    int outSizeH = pCnnNet->stPoolL2.inputHeight;
    /* 第一层的传播 */

    /* 第一层输出数据 */
    nSize mapSize = {pCnnNet->stCovL1.mapSize, pCnnNet->stCovL1.mapSize};
    nSize inSize = {pCnnNet->stCovL1.inputWidth, pCnnNet->stCovL1.inputHeight};
    nSize outSize = {pCnnNet->stPoolL2.inputWidth, pCnnNet->stPoolL2.inputHeight};
/*    PRT("L1: insize[%d-%d], mapsize[%d-%d], outsize[%d-%d]\n", inSize.c, inSize.r, mapSize.c, mapSize.r, outSize.c, outSize.r); */
/*	PRT("L1:inChannels:%d, outChannels:%d, mapSize:%d\n", pCnnNet->stCovL1.inChannels, pCnnNet->stCovL1.outChannels, pCnnNet->stCovL1.mapSize); */
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
            /* 映射矩阵旋转，将卷积变为相关计算 */
            for (kr = 0; kr < CNN_LAYER1_MAP_SIZE; kr++)
            {
                for (kc = 0; kc < CNN_LAYER1_MAP_SIZE; kc++)
                {
                    mapDataRotL1[kr][kc] = pCnnNet->stCovL1.mapData[j][i][CNN_LAYER1_MAP_SIZE - kr - 1][CNN_LAYER1_MAP_SIZE - kc - 1];
                }
            }

            memset(pCnnNet->stCovL1.mapOut, 0, CNN_LAYER1_OUT_SIZE * CNN_LAYER1_OUT_SIZE * sizeof(float));
            Mat2dCorrelation_Valid(inputData, inSize, mapDataRotL1, mapSize, pCnnNet->stCovL1.mapOut, dstSize);

            Mat2D_Add((FLOAT *)pCnnNet->stCovL1.v[i], (FLOAT *)pCnnNet->stCovL1.v[i], (FLOAT *)pCnnNet->stCovL1.mapOut, dstSize);
        }

        for (r = 0; r < outSize.r; r++)
        {
            for (c = 0; c < outSize.c; c++)
            {
                pCnnNet->stCovL1.y[i][r][c] = SigmaActiveFun(pCnnNet->stCovL1.v[i][r][c], pCnnNet->stCovL1.basicData[i]);
            }
        }
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
            PoolingAverage(pCnnNet->stPoolL2.y[i], outSize, pCnnNet->stCovL1.y[i], inSize, pCnnNet->stPoolL2.mapSize);
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

/*			Mat2dRotate_180(pCnnNet->stCovL3.mapData[j][i], mapDataRotL3, mapSize); */
            for (kr = 0; kr < CNN_LAYER3_MAP_SIZE; kr++)
            {
                for (kc = 0; kc < CNN_LAYER3_MAP_SIZE; kc++)
                {
                    mapDataRotL3[kr][kc] = pCnnNet->stCovL3.mapData[j][i][CNN_LAYER3_MAP_SIZE - kr - 1][CNN_LAYER3_MAP_SIZE - kc - 1];
                }
            }

            memset(pCnnNet->stCovL3.mapOut, 0, CNN_LAYER3_OUT_SIZE * CNN_LAYER3_OUT_SIZE * sizeof(float));

            Mat2dCorrelation_Valid(pCnnNet->stPoolL2.y[j], inSize, mapDataRotL3, mapSize, pCnnNet->stCovL3.mapOut, dstSize);

            Mat2D_Add((FLOAT *)pCnnNet->stCovL3.v[i], (FLOAT *)pCnnNet->stCovL3.v[i], (FLOAT *)pCnnNet->stCovL3.mapOut, outSize);

        }

        for (r = 0; r < outSize.r; r++)
        {
            for (c = 0; c < outSize.c; c++)
            {
                pCnnNet->stCovL3.y[i][r][c] = SigmaActiveFun(pCnnNet->stCovL3.v[i][r][c], pCnnNet->stCovL3.basicData[i]);
            }
        }
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
            PoolingAverage(pCnnNet->stPoolL4.y[i], outSize, pCnnNet->stCovL3.y[i], inSize, pCnnNet->stPoolL4.mapSize);
        }
    }

    /* 输出层O5的处理
       1. 首先需要将前面的多维输出展开成一维向量
     */

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
    nSize pCnnNetL5nSize = {pCnnNet->stOutL5.inputNum, pCnnNet->stOutL5.outputNum};/* (12*4*4)192-10 */
    nn2f(pCnnNet->stOutL5.v, pCnnNet->stPoolL4.y, (FLOAT *)pCnnNet->stOutL5.wData, pCnnNet->stOutL5.basicData, pCnnNetL5nSize);

/*
    for(i = 0; i < CNN_LAYER5_OUT_CHANNEL_NUM; i++)
    {
        PRT("%f ", pCnnNet->stOutL5.v[i]);
    }
    PRT("\n");
 */
    for (i = 0; i < pCnnNet->stOutL5.outputNum; i++)
    {
        pCnnNet->stOutL5.y[i] = SigmaActiveFun(pCnnNet->stOutL5.v[i], pCnnNet->stOutL5.basicData[i]);
    }
}

/* 激活函数 input是数据，inputNum说明数据数目，bas表明偏置 */
float activation_Sigma(float input, float bas) /* sigma激活函数 */
{
    float temp = input + bas;

    return (float)1.0 / ((float)(1.0 + exp(-temp)));
}

/*******************************************************************************
   Function:		SigmaActiveFun
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
float SigmaActiveFun(float input, float bas) /* sigma激活函数 */
{
    float temp = input + bas;

    return (float)1.0 / ((float)(1.0 + exp(-temp)));
}

/*******************************************************************************
   Function:		avgPooling
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
void avgPooling(float **output, nSize outputSize, float **input, nSize inputSize, int mapSize) /* 求平均值 */
{
    int i, j, m, n;
    float sum = 0.0;
    int outputW = inputSize.c / mapSize;
    int outputH = inputSize.r / mapSize;

    if (!input || !output)
    {
        return;
    }

    if (outputSize.c != outputW || outputSize.r != outputH)
    {
        PRT("ERROR: output size is wrong!!");
    }

    for (i = 0; i < outputH; i++)
    {
        for (j = 0; j < outputW; j++)
        {
            sum = 0.0;
            for (m = i * mapSize; m < i * mapSize + mapSize; m++)
            {
                for (n = j * mapSize; n < j * mapSize + mapSize; n++)
                {
                    sum = sum + input[m][n];
                }
            }

            output[i][j] = sum / (float)(mapSize * mapSize);
        }
    }
}

/*******************************************************************************
   Function:		PoolingAverage
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
void PoolingAverage(float *output, nSize outputSize, float *input, nSize inputSize, int mapSize) /* 求平均值 */
{
    int i, j, m, n;
    float sum = 0.0;
    int outputW = 0;
    int outputH = 0;

    if (!input || !output || mapSize == 0)
    {
        return;
    }

    outputW = inputSize.c / mapSize;
    outputH = inputSize.r / mapSize;

    if (outputSize.c != outputW || outputSize.r != outputH)
    {
        PRT("ERROR: output size is wrong!!");
        return;
    }

    for (i = 0; i < outputH; i++)
    {
        for (j = 0; j < outputW; j++)
        {
            sum = 0.0;
            for (m = i * mapSize; m < i * mapSize + mapSize; m++)
            {
                for (n = j * mapSize; n < j * mapSize + mapSize; n++)
                {
                    sum += input[m * inputSize.c + n];
                }
            }

            output[i * outputW + j] = sum / (float)(mapSize * mapSize * 1.0);
        }
    }
}

/* 单层全连接神经网络的前向传播,
   向量点乘结果
 */
float vecMulti(float *vec1, float *vec2, int vecL)/* 两向量相乘 */
{
    int i = 0;
    float m = 0.0;

    if (!vec1 || !vec2 || !vecL)
    {
        return 0;
    }

    for (i = 0; i < vecL; i++)
    {
        m = m + vec1[i] * vec2[i];
    }

    return m;
}

/*******************************************************************************
   Function:		nnff
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
void nnff(float *output, float *input, float **wdata, float *bas, nSize nnSize)
{
    int i = 0;
    int w = nnSize.c;
    int h = nnSize.r;

    if (!output)
    {
        return;
    }

    for (i = 0; i < h; i++)
    {
        output[i] = vecMulti(input, wdata[i], w) + bas[i];
    }
}

/*******************************************************************************
   Function:		nn2f
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
void nn2f(float *output, float *input, float *wdata, float *bas, nSize nnSize)
{
    int i = 0, j = 0;
    float m = 0.0;
    int w = nnSize.c;
    int h = nnSize.r;

    if (!output || !input || !wdata || !bas)
    {
        PRT_ERR("param error\n");
        return;
    }

    for (i = 0; i < h; i++)
    {
        m = 0.0;
        for (j = 0; j < w; j++)
        {
            m += input[j] * wdata[i * w + j];
        }

        output[i] = m + bas[i];
    }
}

/* Logic激活函数的自变量微分 sigmoid函数 */
float sigma_derivation(float y)
{
    return y * (1 - y); /* 这里y是指经过激活函数的输出值，而不是自变量 */
}

/* 正向的时候是激活函数，该函数为激活的微分形式 */
float SigmaDerivationFun(float y)
{
    return y * (1 - y); /* 这里y是指经过激活函数的输出值，而不是自变量 */
}

/*******************************************************************************
   Function:		cnnbp
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
void cnnbp(CNN *cnn, float *outputData) /* 网络的后向传播 */
{
    int i, j, c, r; /* 将误差保存到网络中 */

    if (!cnn || !outputData)
    {
        return;
    }

    for (i = 0; i < cnn->O5->outputNum; i++)
    {
        cnn->e[i] = cnn->O5->y[i] - outputData[i];/* 当前值与实际标记值的差 */
    }

    /*从后向前反向计算*/
    /* 输出层O5 */
    PRT("L5 d:\n");
    for (i = 0; i < cnn->O5->outputNum; i++)
    {
        cnn->O5->d[i] = cnn->e[i] * sigma_derivation(cnn->O5->y[i]);
        PRT("%f ", cnn->O5->d[i]);
    }

    PRT("\n");

    /* S4层，传递到S4层的误差 */
    /* 这里没有激活函数 */
    nSize outSize = {cnn->S4->inputWidth / cnn->S4->mapSize, cnn->S4->inputHeight / cnn->S4->mapSize};
    for (i = 0; i < cnn->S4->outChannels; i++)
        for (r = 0; r < outSize.r; r++)
            for (c = 0; c < outSize.c; c++)
                for (j = 0; j < cnn->O5->outputNum; j++)
                {
                    int wInt = i * outSize.c * outSize.r + r * outSize.c + c;
                    cnn->S4->d[i][r][c] = cnn->S4->d[i][r][c] + cnn->O5->d[j] * cnn->O5->wData[j][wInt];
                }

    /* C3层 */
    /* 由S4层传递的各反向误差,这里只是在S4的梯度上扩充一倍 */
    int mapdata = cnn->S4->mapSize;
    nSize S4dSize = {cnn->S4->inputWidth / cnn->S4->mapSize, cnn->S4->inputHeight / cnn->S4->mapSize};
    /* 这里的Pooling是求平均，所以反向传递到下一神经元的误差梯度没有变化 */
    for (i = 0; i < cnn->C3->outChannels; i++)
    {
        float **C3e = UpSample(cnn->S4->d[i], S4dSize, cnn->S4->mapSize, cnn->S4->mapSize);
        for (r = 0; r < cnn->S4->inputHeight; r++)
            for (c = 0; c < cnn->S4->inputWidth; c++)
                cnn->C3->d[i][r][c] = C3e[r][c] * sigma_derivation(cnn->C3->y[i][r][c]) / (float)(cnn->S4->mapSize * cnn->S4->mapSize);

        for (r = 0; r < cnn->S4->inputHeight; r++)
            free(C3e[r]);

        free(C3e);
    }

    /* S2层，S2层没有激活函数，这里只有卷积层有激活函数部分 */
    /* 由卷积层传递给采样层的误差梯度，这里卷积层共有6*12个卷积模板 */
    outSize.c = cnn->C3->inputWidth;
    outSize.r = cnn->C3->inputHeight;
    nSize inSize = {cnn->S4->inputWidth, cnn->S4->inputHeight};
    nSize mapSize = {cnn->C3->mapSize, cnn->C3->mapSize};
    for (i = 0; i < cnn->S2->outChannels; i++)
    {
        for (j = 0; j < cnn->C3->outChannels; j++)
        {
            float **corr = correlation(cnn->C3->mapData[i][j], mapSize, cnn->C3->d[j], inSize, full);
            addmat(cnn->S2->d[i], cnn->S2->d[i], outSize, corr, outSize);
            for (r = 0; r < outSize.r; r++)
                free(corr[r]);

            free(corr);
        }

        /*
           for(r=0;r<cnn->C3->inputHeight;r++)
            for(c=0;c<cnn->C3->inputWidth;c++)
                // 这里本来用于采样的激活
         */
    }

    /* C1层，卷积层 */
    mapdata = cnn->S2->mapSize;
    nSize S2dSize = {cnn->S2->inputWidth / cnn->S2->mapSize, cnn->S2->inputHeight / cnn->S2->mapSize};
    /* 这里的Pooling是求平均，所以反向传递到下一神经元的误差梯度没有变化 */
    for (i = 0; i < cnn->C1->outChannels; i++)
    {
        float **C1e = UpSample(cnn->S2->d[i], S2dSize, cnn->S2->mapSize, cnn->S2->mapSize);
        for (r = 0; r < cnn->S2->inputHeight; r++)
            for (c = 0; c < cnn->S2->inputWidth; c++)
                cnn->C1->d[i][r][c] = C1e[r][c] * sigma_derivation(cnn->C1->y[i][r][c]) / (float)(cnn->S2->mapSize * cnn->S2->mapSize);

        for (r = 0; r < cnn->S2->inputHeight; r++)
            free(C1e[r]);

        free(C1e);
    }
}

/*******************************************************************************
   Function:		CnnBackPass
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
void CnnBackPass(CNN_NET_STR *pCnnNet, float *outputData) /* 网络的后向传播 */
{
    int i, j, c, r; /* 将误差保存到网络中 */
    nSize dstSize = {0, 0};
    FLOAT *pTmpD = NULL;

    if (!pCnnNet || !outputData)
    {
        return;
    }

    for (i = 0; i < pCnnNet->stOutL5.outputNum; i++)
    {
        pCnnNet->e[i] = pCnnNet->stOutL5.y[i] - outputData[i];/* 当前值与实际标记值的差 */
    }

    /*从后向前反向计算*/
    /* 输出层O5 */
    PRT("L5 d:\n");
    for (i = 0; i < pCnnNet->stOutL5.outputNum; i++)
    {
        pCnnNet->stOutL5.d[i] = pCnnNet->e[i] * SigmaDerivationFun(pCnnNet->stOutL5.y[i]);
        PRT("%f ", pCnnNet->stOutL5.d[i]);
    }

    PRT("\n");
    /* S4层，传递到S4层的误差 */
    /* 这里没有激活函数 */
    nSize outSize = {pCnnNet->stPoolL4.inputWidth / pCnnNet->stPoolL4.mapSize, pCnnNet->stPoolL4.inputHeight / pCnnNet->stPoolL4.mapSize};
    for (i = 0; i < pCnnNet->stPoolL4.outChannels; i++)
    {
        for (r = 0; r < outSize.r; r++)
        {
            for (c = 0; c < outSize.c; c++)
            {
                for (j = 0; j < pCnnNet->stOutL5.outputNum; j++)
                {
                    int wInt = i * outSize.c * outSize.r + r * outSize.c + c;
                    pCnnNet->stPoolL4.d[i][r][c] += pCnnNet->stOutL5.d[j] * pCnnNet->stOutL5.wData[j][wInt];
                }
            }
        }
    }

    /* C3层 */
    /* 由S4层传递的各反向误差,这里只是在S4的梯度上扩充一倍 */
    int mapdata = pCnnNet->stPoolL4.mapSize;
    nSize S4dSize = {pCnnNet->stPoolL4.inputWidth / pCnnNet->stPoolL4.mapSize, pCnnNet->stPoolL4.inputHeight / pCnnNet->stPoolL4.mapSize};
    dstSize.c = pCnnNet->stPoolL4.inputWidth;
    dstSize.r = pCnnNet->stPoolL4.inputHeight;
    pTmpD = (FLOAT *)malloc(dstSize.r * dstSize.c * sizeof(FLOAT));
    /* 这里的Pooling是求平均，所以反向传递到下一神经元的误差梯度没有变化 */
    for (i = 0; i < pCnnNet->stCovL3.outChannels; i++)
    {
        Mat2dUpSample(pCnnNet->stPoolL4.d[i], S4dSize, pTmpD, dstSize);

        for (r = 0; r < pCnnNet->stPoolL4.inputHeight; r++)
        {
            for (c = 0; c < pCnnNet->stPoolL4.inputWidth; c++)
                pCnnNet->stCovL3.d[i][r][c] = pTmpD[r * dstSize.c + c] * SigmaDerivationFun(pCnnNet->stCovL3.y[i][r][c]) / (float)(pCnnNet->stPoolL4.mapSize * pCnnNet->stPoolL4.mapSize);
        }
    }

    free(pTmpD);
    pTmpD = NULL;

    /* S2层，S2层没有激活函数，这里只有卷积层有激活函数部分 */
    /* 由卷积层传递给采样层的误差梯度，这里卷积层共有6*12个卷积模板 */
    outSize.c = pCnnNet->stCovL3.inputWidth;
    outSize.r = pCnnNet->stCovL3.inputHeight;
    nSize inSize = {pCnnNet->stPoolL4.inputWidth, pCnnNet->stPoolL4.inputHeight};
    nSize mapSize = {pCnnNet->stCovL3.mapSize, pCnnNet->stCovL3.mapSize};
    dstSize.c = inSize.c + mapSize.c - 1;
    dstSize.r = inSize.r + mapSize.r - 1;
    pTmpD = (FLOAT *)malloc(dstSize.r * dstSize.c * sizeof(FLOAT));
    for (i = 0; i < pCnnNet->stPoolL2.outChannels; i++)
    {
        for (j = 0; j < pCnnNet->stCovL3.outChannels; j++)
        {
/*            float **corr = correlation(pCnnNet->stCovL3.mapData[i][j], mapSize, pCnnNet->stCovL3.d[j], inSize, full); */

            Mat2dCorrelation_Full(pCnnNet->stCovL3.d[j], inSize, pCnnNet->stCovL3.mapData[i][j], mapSize, pTmpD, dstSize);

            Mat2D_Add(pCnnNet->stPoolL2.d[i], pCnnNet->stPoolL2.d[i], pTmpD, outSize);

        }

        /*
           for(r=0;r<pCnnNet->stCovL3.inputHeight;r++)
            for(c=0;c<pCnnNet->stCovL3.inputWidth;c++)
                // 这里本来用于采样的激活
         */
    }

    free(pTmpD);
    pTmpD = NULL;

    /* C1层，卷积层 */
    mapdata = pCnnNet->stPoolL2.mapSize;
    nSize S2dSize = {pCnnNet->stPoolL2.inputWidth / pCnnNet->stPoolL2.mapSize, pCnnNet->stPoolL2.inputHeight / pCnnNet->stPoolL2.mapSize};
    dstSize.c = pCnnNet->stPoolL2.inputWidth;
    dstSize.r = pCnnNet->stPoolL2.inputHeight;
    pTmpD = (FLOAT *)malloc(dstSize.r * dstSize.c * sizeof(FLOAT));
    /* 这里的Pooling是求平均，所以反向传递到下一神经元的误差梯度没有变化 */
    for (i = 0; i < pCnnNet->stCovL1.outChannels; i++)
    {
        Mat2dUpSample(pCnnNet->stPoolL2.d[i], S2dSize, pTmpD, dstSize);
        for (r = 0; r < pCnnNet->stPoolL2.inputHeight; r++)
        {
            for (c = 0; c < pCnnNet->stPoolL2.inputWidth; c++)
                pCnnNet->stCovL1.d[i][r][c] = pTmpD[r * dstSize.c + c] * SigmaDerivationFun(pCnnNet->stCovL1.y[i][r][c]) / (float)(pCnnNet->stPoolL2.mapSize * pCnnNet->stPoolL2.mapSize);
        }
    }

    free(pTmpD);
    pTmpD = NULL;
}

/*******************************************************************************
   Function:		cnnapplygrads
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
void cnnapplygrads(CNN *cnn, CNNOpts opts, float **inputData) /* 更新权重 */
{
    /* 这里存在权重的主要是卷积层和输出层 */
    /* 更新这两个地方的权重就可以了 */
    int i, j, r, c;

    /* C1层的权重更新 */
    nSize dSize = {cnn->S2->inputHeight, cnn->S2->inputWidth};
    nSize ySize = {cnn->C1->inputHeight, cnn->C1->inputWidth};
    nSize mapSize = {cnn->C1->mapSize, cnn->C1->mapSize};

    for (i = 0; i < cnn->C1->outChannels; i++)
    {
        for (j = 0; j < cnn->C1->inChannels; j++)
        {
            float **flipInputData = rotate180(inputData, ySize);
            float **C1dk = cov(cnn->C1->d[i], dSize, flipInputData, ySize, COV_VALID);
            multifactor(C1dk, C1dk, mapSize, -1 * opts.alpha);
            addmat(cnn->C1->mapData[j][i], cnn->C1->mapData[j][i], mapSize, C1dk, mapSize);
            for (r = 0; r < (dSize.r - (ySize.r - 1)); r++)
            {
                free(C1dk[r]);
            }

            free(C1dk);
            for (r = 0; r < ySize.r; r++)
                free(flipInputData[r]);

            free(flipInputData);
        }

        cnn->C1->basicData[i] = cnn->C1->basicData[i] - opts.alpha * Mat2dSum(cnn->C1->d[i], dSize);
    }

    /* C3层的权重更新 */
    dSize.c = cnn->S4->inputWidth;
    dSize.r = cnn->S4->inputHeight;
    ySize.c = cnn->C3->inputWidth;
    ySize.r = cnn->C3->inputHeight;
    mapSize.c = cnn->C3->mapSize;
    mapSize.r = cnn->C3->mapSize;
    for (i = 0; i < cnn->C3->outChannels; i++)
    {
        for (j = 0; j < cnn->C3->inChannels; j++)
        {
            float **flipinputData = rotate180(cnn->S2->y[j], ySize);
            float **C3dk = cov(cnn->C3->d[i], dSize, flipinputData, ySize, COV_VALID);
            multifactor(C3dk, C3dk, mapSize, -1.0 * opts.alpha);
            addmat(cnn->C3->mapData[j][i], cnn->C3->mapData[j][i], mapSize, C3dk, mapSize);
            for (r = 0; r < (dSize.r - (ySize.r - 1)); r++)
                free(C3dk[r]);

            free(C3dk);
            for (r = 0; r < ySize.r; r++)
                free(flipinputData[r]);

            free(flipinputData);
        }

        cnn->C3->basicData[i] = cnn->C3->basicData[i] - opts.alpha * Mat2dSum(cnn->C3->d[i], dSize);
    }

    /* 输出层 */
    /* 首先需要将前面的多维输出展开成一维向量 */
    float *O5inData = (float *)malloc((cnn->O5->inputNum) * sizeof(float));
    nSize outSize = {cnn->S4->inputWidth / cnn->S4->mapSize, cnn->S4->inputHeight / cnn->S4->mapSize};
    for (i = 0; i < (cnn->S4->outChannels); i++)
        for (r = 0; r < outSize.r; r++)
            for (c = 0; c < outSize.c; c++)
                O5inData[i * outSize.r * outSize.c + r * outSize.c + c] = cnn->S4->y[i][r][c];

    for (j = 0; j < cnn->O5->outputNum; j++)
    {
        for (i = 0; i < cnn->O5->inputNum; i++)
            cnn->O5->wData[j][i] = cnn->O5->wData[j][i] - opts.alpha * cnn->O5->d[j] * O5inData[i];

        cnn->O5->basicData[j] = cnn->O5->basicData[j] - opts.alpha * cnn->O5->d[j];
    }

    free(O5inData);
}

/*******************************************************************************
   Function:		CnnApplyGrads
   Description: 这里存在权重的主要是卷积层和输出层 ,更新这两个地方的权重就可以了
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
void CnnApplyGrads(CNN_NET_STR *pCnnNet, CNNOpts opts, float *inputData) /* 更新权重 */
{
    int i, j, r, c;
    FLOAT *pTmpMat = NULL;
    FLOAT *pDstMat = NULL;
    FLOAT *pTmpMapMat = NULL;

    /* 参数校验 */
    if (!pCnnNet || !inputData)
    {
        return;
    }

    /* C1层的权重更新 */
    nSize ySize = {pCnnNet->stCovL1.inputWidth, pCnnNet->stCovL1.inputHeight};
    nSize dSize = {pCnnNet->stPoolL2.inputWidth, pCnnNet->stPoolL2.inputHeight};
    nSize mapSize = {pCnnNet->stCovL1.mapSize, pCnnNet->stCovL1.mapSize};
    pTmpMapMat = malloc(dSize.r * dSize.c * sizeof(FLOAT));
    pTmpMat = malloc(ySize.r * ySize.c * sizeof(FLOAT));
    pDstMat = malloc(mapSize.r * mapSize.c * sizeof(FLOAT));
    for (i = 0; i < pCnnNet->stCovL1.outChannels; i++)
    {
        for (j = 0; j < pCnnNet->stCovL1.inChannels; j++)
        {
            Mat2dRotate_180(inputData, pTmpMat, ySize);

            Mat2dRotate_180(pCnnNet->stCovL1.d[i], pTmpMapMat, dSize);

            Mat2dCorrelation_Valid(pTmpMapMat, dSize, pTmpMat, ySize, pDstMat, mapSize);

            Mat2dMultiFactor(pDstMat, pDstMat, mapSize, -1 * opts.alpha);

            Mat2D_Add(pCnnNet->stCovL1.mapData[j][i], pCnnNet->stCovL1.mapData[j][i], pDstMat, mapSize);

        }

        pCnnNet->stCovL1.basicData[i] = pCnnNet->stCovL1.basicData[i] - opts.alpha * Mat2dSum_float(pCnnNet->stCovL1.d[i], dSize);
    }

    free(pTmpMapMat);
    pTmpMapMat = NULL;
    free(pTmpMat);
    pTmpMat = NULL;

    free(pDstMat);
    pDstMat = NULL;

    /* C3层的权重更新 */
    dSize.c = pCnnNet->stPoolL4.inputWidth;
    dSize.r = pCnnNet->stPoolL4.inputHeight;
    ySize.c = pCnnNet->stCovL3.inputWidth;
    ySize.r = pCnnNet->stCovL3.inputHeight;
    mapSize.c = pCnnNet->stCovL3.mapSize;
    mapSize.r = pCnnNet->stCovL3.mapSize;

    pTmpMat = malloc(ySize.r * ySize.c * sizeof(FLOAT));
    pTmpMapMat = malloc(dSize.r * dSize.c * sizeof(FLOAT));
    pDstMat = malloc(mapSize.r * mapSize.c * sizeof(FLOAT));
    for (i = 0; i < pCnnNet->stCovL3.outChannels; i++)
    {
        for (j = 0; j < pCnnNet->stCovL3.inChannels; j++)
        {
            Mat2dRotate_180(pCnnNet->stPoolL2.y[j], pTmpMat, ySize);

            Mat2dRotate_180(pCnnNet->stCovL3.d[i], pTmpMapMat, dSize);

            Mat2dCorrelation_Valid(pTmpMapMat, dSize, pTmpMat, ySize, pDstMat, mapSize);

            Mat2dMultiFactor(pDstMat, pDstMat, mapSize, -1 * opts.alpha);

            Mat2D_Add(pCnnNet->stCovL3.mapData[j][i], pCnnNet->stCovL3.mapData[j][i], pDstMat, mapSize);

        }

        pCnnNet->stCovL3.basicData[i] = pCnnNet->stCovL3.basicData[i] - opts.alpha * Mat2dSum_float(pCnnNet->stCovL3.d[i], dSize);
    }

    free(pTmpMat);
    pTmpMat = NULL;
    free(pTmpMapMat);
    pTmpMapMat = NULL;
    free(pDstMat);
    pDstMat = NULL;

    /* 输出层 */
    /* 首先需要将前面的多维输出展开成一维向量 */
/*    float *O5inData = (float *)malloc((pCnnNet->stOutL5.inputNum) * sizeof(FLOAT)); */
/*    nSize outSize = {pCnnNet->stPoolL4.inputWidth / pCnnNet->stPoolL4.mapSize, pCnnNet->stPoolL4.inputHeight / pCnnNet->stPoolL4.mapSize}; */
/*    for (i = 0; i < (pCnnNet->stPoolL4.outChannels); i++){ */
/*        for (r = 0; r < outSize.r; r++) */
/*            for (c = 0; c < outSize.c; c++) */
/*                O5inData[i * outSize.r * outSize.c + r * outSize.c + c] = pCnnNet->stPoolL4.y[i][r][c]; */
/*      } */
    for (j = 0; j < pCnnNet->stOutL5.outputNum; j++)
    {
        for (i = 0; i < pCnnNet->stOutL5.inputNum; i++)
        {
            pCnnNet->stOutL5.wData[j][i] = pCnnNet->stOutL5.wData[j][i] - opts.alpha * pCnnNet->stOutL5.d[j] * (FLOAT)(*((float *)pCnnNet->stPoolL4.y + i));
        }

        pCnnNet->stOutL5.basicData[j] = pCnnNet->stOutL5.basicData[j] - opts.alpha * pCnnNet->stOutL5.d[j];
    }

/*    free(O5inData); */
}

/*******************************************************************************
   Function:		cnnclear
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
void cnnclear(CNN *cnn)
{
    /* 将神经元的部分数据清除 */
    int j, c, r;

    if (!cnn)
    {
        return;
    }

    /* C1网络 */
    for (j = 0; j < cnn->C1->outChannels; j++)
    {
        for (r = 0; r < cnn->S2->inputHeight; r++)
        {
            for (c = 0; c < cnn->S2->inputWidth; c++)
            {
                cnn->C1->d[j][r][c] = (float)0.0;
                cnn->C1->v[j][r][c] = (float)0.0;
                cnn->C1->y[j][r][c] = (float)0.0;
            }
        }
    }

    /* S2网络 */
    for (j = 0; j < cnn->S2->outChannels; j++)
    {
        for (r = 0; r < cnn->C3->inputHeight; r++)
        {
            for (c = 0; c < cnn->C3->inputWidth; c++)
            {
                cnn->S2->d[j][r][c] = (float)0.0;
                cnn->S2->y[j][r][c] = (float)0.0;
            }
        }
    }

    /* C3网络 */
    for (j = 0; j < cnn->C3->outChannels; j++)
    {
        for (r = 0; r < cnn->S4->inputHeight; r++)
        {
            for (c = 0; c < cnn->S4->inputWidth; c++)
            {
                cnn->C3->d[j][r][c] = (float)0.0;
                cnn->C3->v[j][r][c] = (float)0.0;
                cnn->C3->y[j][r][c] = (float)0.0;
            }
        }
    }

    /* S4网络 */
    for (j = 0; j < cnn->S4->outChannels; j++)
    {
        for (r = 0; r < cnn->S4->inputHeight / cnn->S4->mapSize; r++)
        {
            for (c = 0; c < cnn->S4->inputWidth / cnn->S4->mapSize; c++)
            {
                cnn->S4->d[j][r][c] = (float)0.0;
                cnn->S4->y[j][r][c] = (float)0.0;
            }
        }
    }

    /* O5输出 */
    for (j = 0; j < cnn->O5->outputNum; j++)
    {
        cnn->O5->d[j] = (float)0.0;
        cnn->O5->v[j] = (float)0.0;
        cnn->O5->y[j] = (float)0.0;
    }
}

/*******************************************************************************
   Function:		CnnParamClear
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
void CnnParamClear(CNN_NET_STR *pCnnNet)
{
    /* 将神经元的部分数据清除 */
    int j, c, r;

    if (!pCnnNet)
    {
        return;
    }

    /* C1网络 */
    memset(pCnnNet->stCovL1.v, 0, CNN_LAYER1_OUT_CHANNEL_NUM * CNN_LAYER1_OUT_SIZE * CNN_LAYER1_OUT_SIZE * sizeof(FLOAT));
    memset(pCnnNet->stCovL1.y, 0, CNN_LAYER1_OUT_CHANNEL_NUM * CNN_LAYER1_OUT_SIZE * CNN_LAYER1_OUT_SIZE * sizeof(FLOAT));
    memset(pCnnNet->stCovL1.d, 0, CNN_LAYER1_OUT_CHANNEL_NUM * CNN_LAYER1_OUT_SIZE * CNN_LAYER1_OUT_SIZE * sizeof(FLOAT));

    /* S2网络 */
    memset(pCnnNet->stPoolL2.y, 0, CNN_LAYER2_OUT_CHANNEL_NUM * CNN_LAYER2_OUT_SIZE * CNN_LAYER2_OUT_SIZE * sizeof(FLOAT));
    memset(pCnnNet->stPoolL2.d, 0, CNN_LAYER2_OUT_CHANNEL_NUM * CNN_LAYER2_OUT_SIZE * CNN_LAYER2_OUT_SIZE * sizeof(FLOAT));

    /* C3网络 */
    memset(pCnnNet->stCovL3.v, 0, CNN_LAYER3_OUT_CHANNEL_NUM * CNN_LAYER3_OUT_SIZE * CNN_LAYER3_OUT_SIZE * sizeof(FLOAT));
    memset(pCnnNet->stCovL3.y, 0, CNN_LAYER3_OUT_CHANNEL_NUM * CNN_LAYER3_OUT_SIZE * CNN_LAYER3_OUT_SIZE * sizeof(FLOAT));
    memset(pCnnNet->stCovL3.d, 0, CNN_LAYER3_OUT_CHANNEL_NUM * CNN_LAYER3_OUT_SIZE * CNN_LAYER3_OUT_SIZE * sizeof(FLOAT));

    /* S4网络 */
    memset(pCnnNet->stPoolL4.y, 0, CNN_LAYER4_OUT_CHANNEL_NUM * CNN_LAYER4_OUT_SIZE * CNN_LAYER4_OUT_SIZE * sizeof(FLOAT));
    memset(pCnnNet->stPoolL4.d, 0, CNN_LAYER4_OUT_CHANNEL_NUM * CNN_LAYER4_OUT_SIZE * CNN_LAYER4_OUT_SIZE * sizeof(FLOAT));

    /* O5输出 */
    memset(pCnnNet->stOutL5.v, 0, CNN_LAYER5_OUT_CHANNEL_NUM * sizeof(FLOAT));
    memset(pCnnNet->stOutL5.y, 0, CNN_LAYER5_OUT_CHANNEL_NUM * sizeof(FLOAT));
    memset(pCnnNet->stOutL5.d, 0, CNN_LAYER5_OUT_CHANNEL_NUM * sizeof(FLOAT));

}

/* 这是用于测试的函数 */
void saveCnnData(CNN *cnn, const char *filename, float **inputdata) /* 保存CNN网络中的相关数据 */
{
    int i, j, r;
    FILE *fp = NULL;

    fp = fopen(filename, "wb");
    if (fp == NULL)
        printf("%s[%d]write file failed\n", __FILE__, __LINE__);

    /* C1的数据 */
    for (i = 0; i < cnn->C1->inputHeight; i++)
        fwrite(inputdata[i], sizeof(float), cnn->C1->inputWidth, fp); /* 图片数据 */

    /* C1网络 */
    for (i = 0; i < cnn->C1->inChannels; i++)
        for (j = 0; j < cnn->C1->outChannels; j++)
            for (r = 0; r < cnn->C1->mapSize; r++)
                fwrite(cnn->C1->mapData[i][j][r], sizeof(float), cnn->C1->mapSize, fp);

    fwrite(cnn->C1->basicData, sizeof(float), cnn->C1->outChannels, fp);

    for (j = 0; j < cnn->C1->outChannels; j++)
    {
        for (r = 0; r < cnn->S2->inputHeight; r++)
        {
            fwrite(cnn->C1->v[j][r], sizeof(float), cnn->S2->inputWidth, fp);
        }

        for (r = 0; r < cnn->S2->inputHeight; r++)
        {
            fwrite(cnn->C1->d[j][r], sizeof(float), cnn->S2->inputWidth, fp);
        }

        for (r = 0; r < cnn->S2->inputHeight; r++)
        {
            fwrite(cnn->C1->y[j][r], sizeof(float), cnn->S2->inputWidth, fp);
        }
    }

    /* S2网络 */
    for (j = 0; j < cnn->S2->outChannels; j++)
    {
        for (r = 0; r < cnn->C3->inputHeight; r++)
        {
            fwrite(cnn->S2->d[j][r], sizeof(float), cnn->C3->inputWidth, fp);
        }

        for (r = 0; r < cnn->C3->inputHeight; r++)
        {
            fwrite(cnn->S2->y[j][r], sizeof(float), cnn->C3->inputWidth, fp);
        }
    }

    /* C3网络 */
    for (i = 0; i < cnn->C3->inChannels; i++)
        for (j = 0; j < cnn->C3->outChannels; j++)
            for (r = 0; r < cnn->C3->mapSize; r++)
                fwrite(cnn->C3->mapData[i][j][r], sizeof(float), cnn->C3->mapSize, fp);

    fwrite(cnn->C3->basicData, sizeof(float), cnn->C3->outChannels, fp);

    for (j = 0; j < cnn->C3->outChannels; j++)
    {
        for (r = 0; r < cnn->S4->inputHeight; r++)
        {
            fwrite(cnn->C3->v[j][r], sizeof(float), cnn->S4->inputWidth, fp);
        }

        for (r = 0; r < cnn->S4->inputHeight; r++)
        {
            fwrite(cnn->C3->d[j][r], sizeof(float), cnn->S4->inputWidth, fp);
        }

        for (r = 0; r < cnn->S4->inputHeight; r++)
        {
            fwrite(cnn->C3->y[j][r], sizeof(float), cnn->S4->inputWidth, fp);
        }
    }

    /* S4网络 */
    for (j = 0; j < cnn->S4->outChannels; j++)
    {
        for (r = 0; r < cnn->S4->inputHeight / cnn->S4->mapSize; r++)
        {
            fwrite(cnn->S4->d[j][r], sizeof(float), cnn->S4->inputWidth / cnn->S4->mapSize, fp);
        }

        for (r = 0; r < cnn->S4->inputHeight / cnn->S4->mapSize; r++)
        {
            fwrite(cnn->S4->y[j][r], sizeof(float), cnn->S4->inputWidth / cnn->S4->mapSize, fp);
        }
    }

    /* O5输出层 */
    for (i = 0; i < cnn->O5->outputNum; i++)
        fwrite(cnn->O5->wData[i], sizeof(float), cnn->O5->inputNum, fp);

    fwrite(cnn->O5->basicData, sizeof(float), cnn->O5->outputNum, fp);
    fwrite(cnn->O5->v, sizeof(float), cnn->O5->outputNum, fp);
    fwrite(cnn->O5->d, sizeof(float), cnn->O5->outputNum, fp);
    fwrite(cnn->O5->y, sizeof(float), cnn->O5->outputNum, fp);

    fclose(fp);
}

/*******************************************************************************
   Function:		SaveCnnMidData
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
void SaveCnnMidData(CNN_NET_STR *pCnnNet, const char *filename, float **inputdata) /* 保存CNN网络中的相关数据 */
{
    int i, j, r;
    FILE *fp = NULL;

    if (!pCnnNet)
    {
        return;
    }

    fp = fopen(filename, "wb");
    if (fp == NULL)
    {
        PRT_ERR("open file failed !\n");
    }

    /* C1的数据 */
    for (i = 0; i < pCnnNet->stCovL1.inputHeight; i++)
    {
        fwrite(inputdata[i], sizeof(FLOAT), pCnnNet->stCovL1.inputWidth, fp);/* 图片数据 */
    }

    /* C1网络 */
    for (i = 0; i < pCnnNet->stCovL1.inChannels; i++)
    {
        for (j = 0; j < pCnnNet->stCovL1.outChannels; j++)
        {
            for (r = 0; r < pCnnNet->stCovL1.mapSize; r++)
            {
                fwrite(pCnnNet->stCovL1.mapData[i][j][r], sizeof(FLOAT), pCnnNet->stCovL1.mapSize, fp);
            }
        }
    }

    fwrite(pCnnNet->stCovL1.basicData, sizeof(FLOAT), pCnnNet->stCovL1.outChannels, fp);

    for (j = 0; j < pCnnNet->stCovL1.outChannels; j++)
    {
        for (r = 0; r < pCnnNet->stPoolL2.inputHeight; r++)
        {
            fwrite(pCnnNet->stCovL1.v[j][r], sizeof(FLOAT), pCnnNet->stPoolL2.inputWidth, fp);
        }

        for (r = 0; r < pCnnNet->stPoolL2.inputHeight; r++)
        {
            fwrite(pCnnNet->stCovL1.d[j][r], sizeof(FLOAT), pCnnNet->stPoolL2.inputWidth, fp);
        }

        for (r = 0; r < pCnnNet->stPoolL2.inputHeight; r++)
        {
            fwrite(pCnnNet->stCovL1.y[j][r], sizeof(FLOAT), pCnnNet->stPoolL2.inputWidth, fp);
        }
    }

    /* S2网络 */
    for (j = 0; j < pCnnNet->stPoolL2.outChannels; j++)
    {
        for (r = 0; r < pCnnNet->stCovL3.inputHeight; r++)
        {
            fwrite(pCnnNet->stPoolL2.d[j][r], sizeof(FLOAT), pCnnNet->stCovL3.inputWidth, fp);
        }

        for (r = 0; r < pCnnNet->stCovL3.inputHeight; r++)
        {
            fwrite(pCnnNet->stPoolL2.y[j][r], sizeof(FLOAT), pCnnNet->stCovL3.inputWidth, fp);
        }
    }

    /* C3网络 */
    for (i = 0; i < pCnnNet->stCovL3.inChannels; i++)
    {
        for (j = 0; j < pCnnNet->stCovL3.outChannels; j++)
        {
            for (r = 0; r < pCnnNet->stCovL3.mapSize; r++)
            {
                fwrite(pCnnNet->stCovL3.mapData[i][j][r], sizeof(FLOAT), pCnnNet->stCovL3.mapSize, fp);
            }
        }
    }

    fwrite(pCnnNet->stCovL3.basicData, sizeof(FLOAT), pCnnNet->stCovL3.outChannels, fp);

    for (j = 0; j < pCnnNet->stCovL3.outChannels; j++)
    {
        for (r = 0; r < pCnnNet->stPoolL4.inputHeight; r++)
        {
            fwrite(pCnnNet->stCovL3.v[j][r], sizeof(FLOAT), pCnnNet->stPoolL4.inputWidth, fp);
        }

        for (r = 0; r < pCnnNet->stPoolL4.inputHeight; r++)
        {
            fwrite(pCnnNet->stCovL3.d[j][r], sizeof(FLOAT), pCnnNet->stPoolL4.inputWidth, fp);
        }

        for (r = 0; r < pCnnNet->stPoolL4.inputHeight; r++)
        {
            fwrite(pCnnNet->stCovL3.y[j][r], sizeof(FLOAT), pCnnNet->stPoolL4.inputWidth, fp);
        }
    }

    /* S4网络 */
    for (j = 0; j < pCnnNet->stPoolL4.outChannels; j++)
    {
        for (r = 0; r < pCnnNet->stPoolL4.inputHeight / pCnnNet->stPoolL4.mapSize; r++)
        {
            fwrite(pCnnNet->stPoolL4.d[j][r], sizeof(FLOAT), pCnnNet->stPoolL4.inputWidth / pCnnNet->stPoolL4.mapSize, fp);
        }

        for (r = 0; r < pCnnNet->stPoolL4.inputHeight / pCnnNet->stPoolL4.mapSize; r++)
        {
            fwrite(pCnnNet->stPoolL4.y[j][r], sizeof(FLOAT), pCnnNet->stPoolL4.inputWidth / pCnnNet->stPoolL4.mapSize, fp);
        }
    }

    /* O5输出层 */
    for (i = 0; i < pCnnNet->stOutL5.outputNum; i++)
    {
        fwrite(pCnnNet->stOutL5.wData[i], sizeof(FLOAT), pCnnNet->stOutL5.inputNum, fp);
    }

    fwrite(pCnnNet->stOutL5.basicData, sizeof(FLOAT), pCnnNet->stOutL5.outputNum, fp);
    fwrite(pCnnNet->stOutL5.v, sizeof(FLOAT), pCnnNet->stOutL5.outputNum, fp);
    fwrite(pCnnNet->stOutL5.d, sizeof(FLOAT), pCnnNet->stOutL5.outputNum, fp);
    fwrite(pCnnNet->stOutL5.y, sizeof(FLOAT), pCnnNet->stOutL5.outputNum, fp);

    fclose(fp);
}

/*******************************************************************************
   Function:		CNNOptsPrint
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
void CNNOptsPrint(CNNOpts *pCNNOpts)
{
    if (!pCNNOpts)
    {
        return;
    }

    PRT("CNNOpts:\n");
    PRT("numepochs:%d, alpha:%f\n", pCNNOpts->numepochs, pCNNOpts->alpha);
}

/*******************************************************************************
   Function:		CovLayerPrint
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
void CovLayerPrint(CovLayer *pCovLayer)
{
    if (!pCovLayer)
    {
        return;
    }

    PRT("CovLayer:\n");
    PRT("inChannels:%d, outChannels:%d\n", pCovLayer->inChannels, pCovLayer->outChannels);
    PRT("width:%d, height:%d\n", pCovLayer->inputWidth, pCovLayer->inputHeight);
    PRT("mapsize:%d\n", pCovLayer->mapSize);
    PRT("isFullConnect:%d\n", pCovLayer->isFullConnect);
}

/*******************************************************************************
   Function:		PoolLayerPrint
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
void PoolLayerPrint(PoolLayer *pPoolLayer)
{
    if (!pPoolLayer)
    {
        return;
    }

    PRT("PoolLayer:\n");
    PRT("inChannels:%d, outChannels:%d\n", pPoolLayer->inChannels, pPoolLayer->outChannels);
    PRT("width:%d, height:%d\n", pPoolLayer->inputWidth, pPoolLayer->inputHeight);
    PRT("mapsize:%d\n", pPoolLayer->mapSize);
    PRT("poolType:%d\n", pPoolLayer->poolType);
}

/*******************************************************************************
   Function:		OutLayerPrint
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
void OutLayerPrint(OutLayer *pOutLayer)
{
    int i = 0;

    if (!pOutLayer)
    {
        return;
    }

    PRT("OutLayer:\n");
    PRT("inputNum:%d, outputNum:%d\n", pOutLayer->inputNum, pOutLayer->outputNum);
    PRT("isFullConnect:%d\n", pOutLayer->isFullConnect);
    PRT("v: ");
    for (i = 0; i < pOutLayer->outputNum; i++)
    {
        PRT("%f ", pOutLayer->v[i]);
    }

    PRT("\n");
    PRT("d: ");
    for (i = 0; i < pOutLayer->outputNum; i++)
    {
        PRT("%f ", pOutLayer->d[i]);
    }

    PRT("\n");
    PRT("y: ");
    for (i = 0; i < pOutLayer->outputNum; i++)
    {
        PRT("%f ", pOutLayer->y[i]);
    }

    PRT("\n");
}

