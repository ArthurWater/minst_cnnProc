/******************************************************************************
   Copyright 2018-2028 @
   All Rights Reserved
   FileName:    mat.c
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
/* #include <random> */
#include <time.h>
#include "../inc/common.h"
#include "../inc/mat.h"


/* 申请一个二维数组空间,并初始化为0 */
float **ppMat2dMalloc_Float(unsigned int width, unsigned int height)
{
    unsigned int i = 0;

    if (!width || !height)
    {
        return NULL;
    }

    float **outputSpace = (float **)malloc(height * sizeof(float *));
    for (i = 0; i < height; i++)
    {
        outputSpace[i] = (float *)malloc(width * sizeof(float));
        if (outputSpace[i])
        {
            memset(outputSpace[i], 0, width * sizeof(float));
        }
        else
        {
            PRT_ERR("malloc failed !\n");
        }
    }

    return outputSpace;
}

/*******************************************************************************
   Function:		ppMat2dFree_Float
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
void ppMat2dFree_Float(float **ppMat, unsigned int width, unsigned int height)
{
    unsigned int i = 0;

    if (!height || !width)
    {
        return;
    }

    for (i = 0; i < height; i++)
    {
        free(ppMat[i]);
    }

    free(ppMat);
}

/* 矩阵翻转180度 */
float **rotate180(float **mat, nSize matSize)
{
    int i, c, r;
    int outSizeW = matSize.c;
    int outSizeH = matSize.r;
    float **outputData = (float **)malloc(outSizeH * sizeof(float *));

    for (i = 0; i < outSizeH; i++)
        outputData[i] = (float *)malloc(outSizeW * sizeof(float));

    for (r = 0; r < outSizeH; r++)
        for (c = 0; c < outSizeW; c++)
            outputData[r][c] = mat[outSizeH - r - 1][outSizeW - c - 1];

    return outputData;
}

/*矩阵旋转180度*/
void Mat2dRotate_180(float *matIn, float *matOut, nSize matSize)
{
    int i, j;
    int outSizeW = matSize.c;
    int outSizeH = matSize.r;

    if (!matIn || !matOut || !matSize.c || !matSize.r)
    {
        return;
    }

    for (i = 0; i < outSizeH; i++)
    {
        for (j = 0; j < outSizeW; j++)
        {
            matOut[i * outSizeW + j] = matIn[(outSizeH - i - 1) * outSizeW + outSizeW - j - 1];
        }
    }
}

/*
   关于卷积和相关操作的输出选项
   这里共有三种选择：full、same、valid，分别表示
   full指完全，操作后结果的大小为inSize+(mapSize-1)
   same指同输入相同大小
   valid指完全操作后的大小，一般为inSize-(mapSize-1)大小，其不需要将输入添0扩大。
 */
float **correlation(float **map, nSize mapSize, float **inputData, nSize inSize, int type)/* 互相关 */
{
    /* 这里的互相关是在后向传播时调用，类似于将Map反转180度再卷积 */
    /* 为了方便计算，这里先将图像扩充一圈 */
    /* 这里的卷积要分成两拨，偶数模板同奇数模板 */
    unsigned int i, j, c, r;
    int halfmapsizew;
    int halfmapsizeh;

    if (mapSize.r % 2 == 0 && mapSize.c % 2 == 0) /* 模板大小为偶数 */
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
    int outSizeW = inSize.c + (mapSize.c - 1); /* 这里的输出扩大一部分 */
    int outSizeH = inSize.r + (mapSize.r - 1);
    float **outputData = (float **)malloc(outSizeH * sizeof(float *)); /* 互相关的结果扩大了 */
    for (i = 0; i < outSizeH; i++)
        outputData[i] = (float *)calloc(outSizeW, sizeof(float));

    /* 为了方便计算，将inputData扩大一圈 */
    float **exInputData = matEdgeExpand(inputData, inSize, mapSize.c - 1, mapSize.r - 1);

    for (j = 0; j < outSizeH; j++)
        for (i = 0; i < outSizeW; i++)
            for (r = 0; r < mapSize.r; r++)
                for (c = 0; c < mapSize.c; c++)
                {
                    outputData[j][i] = outputData[j][i] + map[r][c] * exInputData[j + r][i + c];
                }

    for (i = 0; i < inSize.r + 2 * (mapSize.r - 1); i++)
        free(exInputData[i]);

    free(exInputData);

    nSize outSize = {outSizeW, outSizeH};
    switch (type) /* 根据不同的情况，返回不同的结果 */
    {
        case full: /* 完全大小的情况 */
            return outputData;
        case same: {
            float **sameres = matEdgeShrink(outputData, outSize, halfmapsizew, halfmapsizeh);
            for (i = 0; i < outSize.r; i++)
                free(outputData[i]);

            free(outputData);
            return sameres;
        }
        case COV_VALID: {
            float **validres;
            if (mapSize.r % 2 == 0 && mapSize.c % 2 == 0)
                validres = matEdgeShrink(outputData, outSize, halfmapsizew * 2 - 1, halfmapsizeh * 2 - 1);
            else
                validres = matEdgeShrink(outputData, outSize, halfmapsizew * 2, halfmapsizeh * 2);

            for (i = 0; i < outSize.r; i++)
                free(outputData[i]);

            free(outputData);
            return validres;
        }
        default:
            return outputData;
    }
}

/*
   互相关
   关于卷积和相关操作的输出选项
   这里共有三种选择：full、same、valid，分别表示
   full指完全，操作后结果的大小为inSize+(mapSize-1)
   same指同输入相同大小
   valid指完全操作后的大小，一般为inSize-(mapSize-1)大小，其不需要将输入添0扩大。

    这里的互相关是在后向传播时调用，类似于将Map反转180度再卷积,为了方便计算，这里先将图像扩充一圈
    这里的卷积要分成两拨，偶数模板同奇数模板。
 */
float **MatCorrelation(float **map, nSize mapSize, float **inputData, nSize inSize, int type)
{
	unsigned int i, j, c, r;
    int halfmapsizew;
    int halfmapsizeh;

    if (!inputData || !map)
    {
        PRT_ERR("param error !\n");
        return NULL;
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
    int outSizeW = inSize.c + (mapSize.c - 1); /* 这里的输出扩大一部分 */
    int outSizeH = inSize.r + (mapSize.r - 1);

    /*
       float** outputData=(float**)malloc(outSizeH*sizeof(float*)); // 互相关的结果扩大了
       for(i=0;i<outSizeH;i++)
        outputData[i]=(float*)calloc(outSizeW,sizeof(float));
     */
    float **outputData = ppMat2dMalloc_Float(outSizeW, outSizeH);



    /* 为了方便计算，将inputData扩大一圈 */
/*	float** exInputData = matEdgeExpand(inputData, inSize, mapSize.c-1, mapSize.r-1); */

    float **exInputData = ppMat2dMalloc_Float(inSize.c + 2 * (mapSize.c - 1), inSize.r + 2 * (mapSize.r - 1));

    Mat2dEdgeExpand(inputData, inSize, exInputData, mapSize.c - 1, mapSize.r - 1);

    /* 输出值计算 */
    for (j = 0; j < outSizeH; j++)
    {
        for (i = 0; i < outSizeW; i++)
        {/* 计算输出矩阵每个位置的值 */
            for (r = 0; r < mapSize.r; r++)
                for (c = 0; c < mapSize.c; c++)
                {
                    outputData[j][i] = outputData[j][i] + map[r][c] * exInputData[j + r][i + c];
                }
        }
    }

    ppMat2dFree_Float(exInputData, inSize.c + 2 * (mapSize.c - 1), inSize.r + 2 * (mapSize.r - 1));

    nSize outSize = {outSizeW, outSizeH};
    switch (type) /* 根据不同的情况，返回不同的结果 */
    {
        case full: /* 完全大小的情况 */
            return outputData;
        case same: {
/*		float** sameres = matEdgeShrink(outputData, outSize, halfmapsizew, halfmapsizeh); */
            float **sameres = ppMat2dMalloc_Float(inSize.c, inSize.r);
            Mat2dEdgeShrink(outputData, outSize, sameres, halfmapsizew, halfmapsizeh);
            ppMat2dFree_Float(outputData, outSizeW, outSizeH);

            return sameres;
        }
        case COV_VALID: {
            float **validres;
        #if 0
            if (mapSize.r % 2 == 0 && mapSize.c % 2 == 0)
                validres = matEdgeShrink(outputData, outSize, halfmapsizew * 2 - 1, halfmapsizeh * 2 - 1);
            else
                validres = matEdgeShrink(outputData, outSize, halfmapsizew * 2, halfmapsizeh * 2);

        #endif
            if (mapSize.r % 2 == 0 && mapSize.c % 2 == 0)/* 偶数 */
            {
                validres = ppMat2dMalloc_Float(inSize.c + 3 - mapSize.c, inSize.r + 3 - mapSize.r);
                Mat2dEdgeShrink(outputData, outSize, validres, halfmapsizew * 2 - 1, halfmapsizeh * 2 - 1);
            }
            else/* 奇数 */
            {
                validres = ppMat2dMalloc_Float(inSize.c + 1 - mapSize.c, inSize.r + 1 - mapSize.r);
                Mat2dEdgeShrink(outputData, outSize, validres, halfmapsizew * 2, halfmapsizeh * 2);
            }

            ppMat2dFree_Float(outputData, outSizeW, outSizeH);

            return validres;
        }
        default:
            return outputData;
    }
}

/*******************************************************************************
   Function:		MatCorrelation_Full
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
void MatCorrelation_Full(float **srcMat, nSize srcSize, float **mapMat, nSize mapSize, float **dstMat, nSize dstSize)
{
	unsigned int i, j, c, r;
    float tmp = 0.0;
    nSize exSize = {0, 0};

    if (!srcMat || !mapMat || !dstMat)
    {
        PRT_ERR("param error !\n");
        return;
    }

    PRT("--map size--[%d-%d]\n", mapSize.r, mapSize.c);

    /* 这里先默认进行full模式的操作，full模式的输出大小为inSize+(mapSize-1) */
    int outSizeW = srcSize.c + (mapSize.c - 1); /* 这里的输出扩大一部分 */
    int outSizeH = srcSize.r + (mapSize.r - 1);

    /* 检查输出大小是否符合要求 */
    if (outSizeW != dstSize.c || outSizeH != dstSize.r)
    {
        PRT_ERR("param error !\n");
        return;
    }

    /* 为了方便计算，将inputData扩大一圈 */
    exSize.c = srcSize.c + 2 * (mapSize.c - 1);
    exSize.r = srcSize.r + 2 * (mapSize.r - 1);
    float **exInputData = ppMat2dMalloc_Float(exSize.c, exSize.r);
    if (!exInputData)
    {
        PRT_ERR("malloc error !\n");
        return;
    }

    Mat2dEdgeExpand(srcMat, srcSize, exInputData, mapSize.c - 1, mapSize.r - 1);
    PRT_ERR("--debug--[%d-%d]--[%d-%d]\n", outSizeW, outSizeH, dstSize.r, dstSize.c);

    /* 输出值计算 */
    for (j = 0; j < outSizeH; j++)
    {
        for (i = 0; i < outSizeW; i++)
        {/* 计算输出矩阵每个位置的值 */
            for (r = 0; r < mapSize.r; r++)
            {
                for (c = 0; c < mapSize.c; c++)
                {
                    dstMat[j][i] = dstMat[j][i] + mapMat[r][c] * exInputData[j + r][i + c];
                }
            }
        }
    }

    PRT_ERR("--debug--\n");
    ppMat2dFree_Float(exInputData, exSize.c, exSize.r);
}

/* 需要确保内存连续 */
void Mat2dCorrelation_Full(float *srcMat, nSize srcSize, float *mapMat, nSize mapSize, float *dstMat, nSize dstSize)
{
	unsigned int i, j, c, r;
    float tmp = 0.0;
    float *pTmpData = NULL;
    nSize exSize = {0, 0};

    if (!srcMat || !mapMat || !dstMat)
    {
        PRT_ERR("param error !\n");
        return;
    }

    /* 这里先默认进行full模式的操作，full模式的输出大小为inSize+(mapSize-1) */
    int outSizeW = srcSize.c + (mapSize.c - 1); /* 这里的输出扩大一部分 */
    int outSizeH = srcSize.r + (mapSize.r - 1);

    /* 检查输出大小是否符合要求 */
    if (outSizeW != dstSize.c || outSizeH != dstSize.r)
    {
        PRT_ERR("param error !\n");
        return;
    }

    /* 为了方便计算，将inputData扩大一圈 */
    exSize.c = srcSize.c + 2 * (mapSize.c - 1);
    exSize.r = srcSize.r + 2 * (mapSize.r - 1);

    pTmpData = (float *)malloc(exSize.c * exSize.r * sizeof(float));
    memset(pTmpData, 0, exSize.c * exSize.r * sizeof(float));

    Mat2dEdgeExpand_2(srcMat, srcSize, pTmpData, mapSize.c - 1, mapSize.r - 1);
/*	PRT_ERR("--debug--[%d-%d]--[%d-%d]\n", outSizeW, outSizeH, dstSize.r, dstSize.c); */

    /* 输出值计算 */
    for (j = 0; j < outSizeH; j++)
    {
        for (i = 0; i < outSizeW; i++)
        {/* 计算输出矩阵每个位置的值 */
            for (r = 0; r < mapSize.r; r++)
            {
                for (c = 0; c < mapSize.c; c++)
                {
                    dstMat[j * outSizeW + i] = dstMat[j * outSizeW + i] + mapMat[r * mapSize.c + c] * pTmpData[(j + r) * exSize.c + i + c];
                }
            }
        }
    }

    free(pTmpData);
}

/* mapsize 宽高相同,此时mapsize的宽高最好为奇数，这样可以正好恢复为原来的大小，否则会有偏差 */
void MatCorrelation_Same(float **srcMat, nSize srcSize, float **mapMat, nSize mapSize, float **dstMat, nSize dstSize)
{
    int halfmapsizew;
    int halfmapsizeh;

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

    if (srcSize.c != dstSize.c || srcSize.r != dstSize.r)
    {
        PRT_ERR("param error !\n");
        return;
    }

    /* 这里先默认进行full模式的操作，full模式的输出大小为inSize+(mapSize-1) */
    int outSizeW = srcSize.c + (mapSize.c - 1); /* 这里的输出扩大一部分 */
    int outSizeH = srcSize.r + (mapSize.r - 1);
    nSize outSize = { outSizeW, outSizeH };

    float **outputData = ppMat2dMalloc_Float(outSizeW, outSizeH);

    MatCorrelation_Full(srcMat, srcSize, mapMat, mapSize, outputData, outSize);

    Mat2dEdgeShrink(outputData, outSize, dstMat, halfmapsizew, halfmapsizeh);

    ppMat2dFree_Float(outputData, outSizeW, outSizeH);
}

/* mapsize 宽高相同 */
void MatCorrelation_Valid(float **srcMat, nSize srcSize, float **mapMat, nSize mapSize, float **dstMat, nSize dstSize)
{
    int halfmapsizew;
    int halfmapsizeh;

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

    float **outputData = ppMat2dMalloc_Float(outSizeW, outSizeH);

    PRT_ERR("--debug-%f-[%d-%d]\n", mapMat[0][0], outSizeW, outSizeH);

    MatCorrelation_Full(srcMat, srcSize, mapMat, mapSize, outputData, outSize);
    PRT_ERR("--debug--[%d-%d]\n", dstSize.c, dstSize.r);

    if (mapSize.r % 2 == 0 && mapSize.c % 2 == 0)/* 模板大小为偶数 */
    {
        Mat2dEdgeShrink(outputData, outSize, dstMat, halfmapsizew * 2 - 1, halfmapsizeh * 2 - 1);
    }
    else
    {
        Mat2dEdgeShrink(outputData, outSize, dstMat, halfmapsizew * 2, halfmapsizeh * 2);
    }

    ppMat2dFree_Float(outputData, outSizeW, outSizeH);
}

/*******************************************************************************
   Function:		Mat2dCorrelation_Valid
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
void Mat2dCorrelation_Valid(float *srcMat, nSize srcSize, float *mapMat, nSize mapSize, float *dstMat, nSize dstSize)
{
    int i, j, c, r;
    float *pTmpData = NULL;
    nSize exSize = {0, 0};
    int halfmapsizew;
    int halfmapsizeh;

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

    float *outputData = malloc(outSizeW * outSizeH * sizeof(float));
    memset(outputData, 0, outSizeW * outSizeH * sizeof(float));

/*	Mat2dCorrelation_Full(srcMat, srcSize, mapMat, mapSize, outputData, outSize); */

    /* 这里先默认进行full模式的操作，full模式的输出大小为inSize+(mapSize-1) */
    /* 为了方便计算，将inputData扩大一圈 */
    exSize.c = srcSize.c + 2 * (mapSize.c - 1);
    exSize.r = srcSize.r + 2 * (mapSize.r - 1);

    pTmpData = (float *)malloc(exSize.c * exSize.r * sizeof(float));
    memset(pTmpData, 0, exSize.c * exSize.r * sizeof(float));

    Mat2dEdgeExpand_2(srcMat, srcSize, pTmpData, mapSize.c - 1, mapSize.r - 1);

    /* 输出值计算 */
    for (j = 0; j < outSizeH; j++)
    {
        for (i = 0; i < outSizeW; i++)
        {/* 计算输出矩阵每个位置的值 */
            for (r = 0; r < mapSize.r; r++)
            {
                for (c = 0; c < mapSize.c; c++)
                {
                    outputData[j * outSizeW + i] += mapMat[r * mapSize.c + c] * pTmpData[(j + r) * exSize.c + i + c];
                }
            }
        }
    }

    free(pTmpData);

    if (mapSize.r % 2 == 0 && mapSize.c % 2 == 0)/* 模板大小为偶数 */
    {
        Mat2dEdgeShrink_w(outputData, outSize, dstMat, halfmapsizew * 2 - 1, halfmapsizeh * 2 - 1);
    }
    else
    {
        Mat2dEdgeShrink_w(outputData, outSize, dstMat, halfmapsizew * 2, halfmapsizeh * 2);
    }

    free(outputData);
}

/*
   卷积操作可以用旋转180度的特征模板相关来求
 */
float **cov(float **map, nSize mapSize, float **inputData, nSize inSize, int type) /* 卷积操作 */
{
    nSize dstMatSize;
    float **dstMat = NULL;

    if (!map || !inputData)
    {
        PRT_ERR("param error !\n");
        return NULL;
    }

    /*
       int i = 0;

       float** flipmap = (float**)malloc(mapSize.r*sizeof(float*));
       for(i = 0; i < mapSize.r; i++)
        flipmap[i] = (float*)malloc(mapSize.c*sizeof(float));
     */

    /*
       float** flipmap = ppMat2dMalloc_Float(mapSize.c, mapSize.r);
       if(!flipmap)
       {
        return NULL;
       }
     */
    float **flipmap = rotate180(map, mapSize); /* 旋转180度的特征模板 */

/*	Mat2dRotate_180(map, flipmap, mapSize); */

    /* 计算输出矩阵大小 */
#if 0
    if (COV_VALID == type)
    {
        dstMatSize.c = inSize.c - (mapSize.c - 1);
        dstMatSize.r = inSize.r - (mapSize.r - 1);
        dstMat = ppMat2dMalloc_Float(dstMatSize.c, dstMatSize.r);
        MatCorrelation_Valid(inputData, inSize, map, mapSize, dstMat, dstMatSize);
    }
    else
    {
        dstMat = correlation(flipmap, mapSize, inputData, inSize, type);
        PRT_ERR("param error !\n");
    }

#else
    dstMat = correlation(flipmap, mapSize, inputData, inSize, type);


#endif

    /*

       ppMat2dFree_Float(flipmap, mapSize.c, mapSize.r);

       for (i = 0; i < mapSize.r; i++)
       {
        free(flipmap[i]);
       }
       free(flipmap);
     */
    return dstMat;
}

/* 这个是矩阵的上采样（等值内插），upc及upr是内插倍数 */
float **UpSample(float **mat, nSize matSize, int upc, int upr)
{
    int i, j, m, n;
    int c = matSize.c;
    int r = matSize.r;
    float **res = (float **)malloc((r * upr) * sizeof(float *)); /* 结果的初始化 */

    for (i = 0; i < (r * upr); i++)
        res[i] = (float *)malloc((c * upc) * sizeof(float));

    for (j = 0; j < r * upr; j = j + upr)
    {
        for (i = 0; i < c * upc; i = i + upc) /* 宽的扩充 */
            for (m = 0; m < upc; m++)
                res[j][i + m] = mat[j / upr][i / upc];

        for (n = 1; n < upr; n++) /*  高的扩充 */
            for (i = 0; i < c * upc; i++)
                res[j + n][i] = res[j][i];
    }

    return res;
}

/*******************************************************************************
   Function:		Mat2dUpSample
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
void Mat2dUpSample(float *srcMat, nSize srcSize, float *dstMat, nSize dstSize)
{
    int i, j, m, n;
    int upc = 1;
    int upr = 1;

    if (!srcMat || !dstMat || srcSize.c==0 || srcSize.r==0 || srcSize.c<dstSize.c || srcSize.r<dstSize.r)
    {
        return;
    }

    memset(dstMat, 0, (dstSize.c * dstSize.r));
    upc = dstSize.c / srcSize.c;
    upr = dstSize.r / srcSize.r;

    for (j = 0; j < dstSize.r; j = j + upr)
    {
        for (i = 0; i < dstSize.c; i = i + upc)/* 宽的扩充 */
        {
            for (m = 0; m < upc; m++)
                dstMat[j * dstSize.c + i + m] = srcMat[(j / upr) * srcSize.c + i / upc];
        }

        for (n = 1; n < upr; n++)      /*  高的扩充 */
        {
//        	memcpy();
            for (i = 0; i < dstSize.c; i++)
                dstMat[(j + n) * dstSize.c + i] = dstMat[j * dstSize.c + i];
        }
    }
}

/*
   这个是矩阵的上采样（等值内插），upc及upr是内插倍数
   matInSize 插值前的大小
   matOutSize插值后的大小，该函数要求输出大小(宽和高)必须是输入的整数倍
   复制左上角的点
 */

/*
   void Mat2dUpSample(float** matIn, nSize matInSize, float** matOut, nSize matOutSize)
   {
    int i,j,m,n;
    unsigned int wUp = 1;
    unsigned int hUp = 1;
    if(!matInSize.c || !matInSize.r)
    {
        return;
    }

    //插值不是整数倍，返回
    if(!(matOutSize.c%matInSize.c) || !(matOutSize.r%matInSize.r))
    {
        return;
    }
    //插值倍数
    wUp = matOutSize.c / matInSize.c;
    hUp = matOutSize.r / matInSize.r;

    for(j=0; j<matOutSize.r; j=j+hUp)
    {
        for(i=0; i<matOutSize.c; i=i+wUp)// 宽的扩充
        {
            for(m=0; m<wUp; m++)
                matOut[j][i+m] = matIn[j/hUp][i/wUp];
        }

        for(n=1; n<hUp; n++)      //  高的扩充
        for (i = 0; i<matOutSize.c; i++)
                matOut[j+n][i] = matOut[j][i];
    }
   }
 */
/* 给二维矩阵边缘扩大，增加addw大小的0值边 */
float **matEdgeExpand(float **mat, nSize matSize, int addc, int addr)
{ /* 向量边缘扩大 */
    int i, j;
    int c = matSize.c;
    int r = matSize.r;
    float **res = (float **)malloc((r + 2 * addr) * sizeof(float *)); /* 结果的初始化 */

    for (i = 0; i < (r + 2 * addr); i++)
        res[i] = (float *)malloc((c + 2 * addc) * sizeof(float));

    for (j = 0; j < r + 2 * addr; j++)
    {
        for (i = 0; i < c + 2 * addc; i++)
        {
            if (j < addr || i < addc || j >= (r + addr) || i >= (c + addc))
                res[j][i] = (float)0.0;
            else
                res[j][i] = mat[j - addr][i - addc]; /* 复制原向量的数据 */

        }
    }

    return res;
}

/*
   给二维矩阵边缘扩大，增加addw大小的0值边
 */
void Mat2dEdgeExpand(float **matIn, nSize matSize, float **matOut, int addc, int addr)
{
    int i, j;
    int c = matSize.c;
    int r = matSize.r;

    if (!matIn || !matOut)
    {
        return;
    }

    for (i = 0; i < (r + 2 * addr); i++)
    {
        memset((float *)matOut[i], 0, (c + 2 * addc) * sizeof(float));
    }

    for (j = 0; j < (r + 2 * addr); j++)
    {
        for (i = 0; i < (c + 2 * addc); i++)
        {
            if (j < addr || j >= (r + addr) || i < addc || i >= (c + addc))
                matOut[j][i] = (float)0.0;
            else
                matOut[j][i] = matIn[j - addr][i - addc]; /* 复制原向量的数据 */

        }
    }
}

/*******************************************************************************
   Function:		Mat2dEdgeExpand_2
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
void Mat2dEdgeExpand_2(float *matIn, nSize matSize, float *matOut, int addc, int addr)
{
    int i, j;
    int c = matSize.c;
    int r = matSize.r;

    if (!matIn || !matOut)
    {
        return;
    }

    for (j = 0; j < (r + 2 * addr); j++)
    {
        for (i = 0; i < (c + 2 * addc); i++)
        {
            if (j < addr || j >= (r + addr) || i < addc || i >= (c + addc))
                matOut[j * (c + 2 * addc) + i] = (float)0.0;
            else
                matOut[j * (c + 2 * addc) + i] = matIn[(j - addr) * c + i - addc]; /* 复制原向量的数据 */

        }
    }
}

/*
   给二维矩阵边缘缩小，擦除shrinkc大小的边
 */
float **matEdgeShrink(float **mat, nSize matSize, int shrinkc, int shrinkr)
{ /* 向量的缩小，宽缩小addw，高缩小addh */
    int i, j;
    int c = matSize.c;
    int r = matSize.r;
    float **res = (float **)malloc((r - 2 * shrinkr) * sizeof(float *)); /* 结果矩阵的初始化 */

    for (i = 0; i < (r - 2 * shrinkr); i++)
        res[i] = (float *)malloc((c - 2 * shrinkc) * sizeof(float));

    for (j = 0; j < r; j++)
    {
        for (i = 0; i < c; i++)
        {
            if (j >= shrinkr && i >= shrinkc && j < (r - shrinkr) && i < (c - shrinkc))
                res[j - shrinkr][i - shrinkc] = mat[j][i]; /* 复制原向量的数据 */

        }
    }

    return res;
}

/*
   给二维矩阵边缘缩小，擦除shrinkc大小的边
   宽左右都缩小shrinkc，
   高上下都缩小shrinkr
   备注：输出矩阵空间外部分配好，提高执行效率
 */
void Mat2dEdgeShrink(float **matIn, nSize matSize, float **matOut, int shrinkc, int shrinkr)
{
    int i, j;
    int c = matSize.c;
    int r = matSize.r;

    if (!matIn || !matOut || (shrinkc > (matSize.c / 2)) || (shrinkr > (matSize.r / 2)))
    {
        return;
    }

    for (j = 0; j < r; j++)
    {
        for (i = 0; i < c; i++)
        {
            if ((j >= shrinkr) && (i >= shrinkc) && (j < (r - shrinkr)) && (i < (c - shrinkc)))
            {
                matOut[j - shrinkr][i - shrinkc] = matIn[j][i]; /* 复制原向量的数据 */
            }
        }
    }
}

/*******************************************************************************
   Function:		Mat2dEdgeShrink_w
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
void Mat2dEdgeShrink_w(float *matIn, nSize matSize, float *matOut, int shrinkc, int shrinkr)
{
    int i, j;
    int w = matSize.c;
    int h = matSize.r;

    if (!matIn || !matOut || (shrinkc > (matSize.c / 2)) || (shrinkr > (matSize.r / 2)))
    {
        return;
    }

    for (j = 0; j < h; j++)
    {
        for (i = 0; i < w; i++)
        {
            if ((j >= shrinkr) && (i >= shrinkc) && (j < (h - shrinkr)) && (i < (w - shrinkc)))
            {
                matOut[(j - shrinkr) * (w - 2 * shrinkc) + i - shrinkc] = matIn[j * w + i]; /* 复制原向量的数据 */
            }
        }
    }
}

/*******************************************************************************
   Function:		savemat
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
void savemat(float **mat, nSize matSize, const char *filename)
{
    int i = 0;
    FILE *fp = NULL;

    fp = fopen(filename, "wb");
    if (fp == NULL)
    {
        printf("write file failed\n");
        return;
    }

    for (i = 0; i < matSize.r; i++)
        fwrite(mat[i], sizeof(float), matSize.c, fp);

    fclose(fp);
}

/* 矩阵相加 */
void addmat(float **res, float **mat1, nSize matSize1, float **mat2, nSize matSize2)
{
    int i, j;

    if (matSize1.c != matSize2.c || matSize1.r != matSize2.r)
        printf("ERROR: Size is not same!");

    for (i = 0; i < matSize1.r; i++)
    {
        for (j = 0; j < matSize1.c; j++)
        {
            res[i][j] = mat1[i][j] + mat2[i][j];
        }
    }
}

/*******************************************************************************
   Function:		Mat2D_Add
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
void Mat2D_Add(float *res, float *mat1, float *mat2, nSize matSize)
{
    int i, j;

    if (matSize.c == 0 || matSize.r == 0)
    {
        printf("ERROR: Size error!");
    }

    for (i = 0; i < matSize.r; i++)
    {
        for (j = 0; j < matSize.c; j++)
        {
            res[i * matSize.c + j] = mat1[i * matSize.c + j] + mat2[i * matSize.c + j];
        }
    }
}

/* 矩阵乘以系数 */
void multifactor(float **res, float **mat, nSize matSize, float factor)
{
    int i, j;

    for (i = 0; i < matSize.r; i++)
    {
        for (j = 0; j < matSize.c; j++)
            res[i][j] = mat[i][j] * factor;
    }
}

/*******************************************************************************
   Function:		Mat2dMultiFactor
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
void Mat2dMultiFactor(float *res, float *mat, nSize matSize, float factor)
{
    int i, j;

    for (i = 0; i < matSize.r; i++)
    {
        for (j = 0; j < matSize.c; j++)
            res[i * matSize.c + j] = mat[i * matSize.c + j] * factor;
    }
}

/* 矩阵各元素的和 */
float Mat2dSum(float **mat, nSize matSize)
{
    int i, j;
    float sum = 0.0;

    for (i = 0; i < matSize.r; i++)
    {
        for (j = 0; j < matSize.c; j++)
            sum = sum + mat[i][j];
    }

    return sum;
}
float Mat2dSum_float(float *mat, nSize matSize)
{
    int i = 0;
    float sum = 0.0;

    for (i = 0; i < (matSize.r*matSize.c); i++)
    {
        sum += mat[i];
    }

    return sum;
}

