// 这里库文件主要存在关于二维矩阵数组的操作
<<<<<<< HEAD
#ifndef __MAT_
#define __MAT_
=======
#ifndef _MMK_MAT_H_
#define _MMK_MAT_H_

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
=======
	//#include <random>
>>>>>>> init files
#include <time.h>

#define full 0
#define same 1
#define COV_VALID 2

<<<<<<< HEAD
typedef struct Mat2DSize{
	unsigned int c; // 列（宽）
	unsigned int r; // 行（高）
}nSize;


float ** ppMat2dMalloc_Float(unsigned int width, unsigned int height);

void ppMat2dFree_Float(float ** ppMat, unsigned int width, unsigned int height);

float** rotate180(float** mat, nSize matSize);// 矩阵翻转180度
void Mat2dRotate_180(float** matIn, float** matOut, nSize matSize);
	
void addmat(float** res, float** mat1, nSize matSize1, float** mat2, nSize matSize2);// 矩阵相加

float** correlation(float** map,nSize mapSize,float** inputData,nSize inSize,int type);// 互相关
//mapsize 宽高相同
void MatCorrelation_Valid(float** srcMat, nSize srcSize, float** mapMat, nSize mapSize, float** dstMat, nSize dstSize);

float** cov(float** map,nSize mapSize,float** inputData,nSize inSize,int type); // 卷积操作

// 这个是矩阵的上采样（等值内插），upc及upr是内插倍数
float** UpSample(float** mat,nSize matSize,int upc,int upr);
void Mat2dUpSample(float** matIn, nSize matInSize, float** matOut, nSize matOutSize);

// 给二维矩阵边缘扩大，增加addw大小的0值边
float** matEdgeExpand(float** mat,nSize matSize,int addc,int addr);
void Mat2dEdgeExpand(float** matIn, nSize matSize, float** matOut, int addc, int addr);

// 给二维矩阵边缘缩小，擦除shrinkc大小的边
float** matEdgeShrink(float** mat,nSize matSize,int shrinkc,int shrinkr);
void Mat2dEdgeShrink(float** matIn, nSize matSize, float** matOut, int shrinkc, int shrinkr);

void savemat(float** mat,nSize matSize,const char* filename);// 保存矩阵数据

void multifactor(float** res, float** mat, nSize matSize, float factor);// 矩阵乘以系数

float Mat2dSum(float** mat,nSize matSize);// 矩阵各元素的和

char * combine_strings(char *a, char *b);

char* intTochar(int i);
=======
	typedef struct Mat2DSize{
		unsigned int c; // 列（宽）
		unsigned int r; // 行（高）
	}nSize;


	float ** ppMat2dMalloc_Float(unsigned int width, unsigned int height);

	void ppMat2dFree_Float(float ** ppMat, unsigned int width, unsigned int height);

	float** rotate180(float** mat, nSize matSize);// 矩阵翻转180度
	void Mat2dRotate_180(float* matIn, float* matOut, nSize matSize);

	void addmat(float** res, float** mat1, nSize matSize1, float** mat2, nSize matSize2);// 矩阵相加
	void Mat2D_Add(float* res, float* mat1, float* mat2, nSize matSize);

	float** correlation(float** map, nSize mapSize, float** inputData, nSize inSize, int type);// 互相关
	float** MatCorrelation(float** map, nSize mapSize, float** inputData, nSize inSize, int type);
	//mapsize 宽高相同
	void Mat2dCorrelation_Valid(float *srcMat, nSize srcSize, float *mapMat, nSize mapSize, float *dstMat, nSize dstSize);
	void MatCorrelation_Valid(float** srcMat, nSize srcSize, float** mapMat, nSize mapSize, float** dstMat, nSize dstSize);
	void Mat2dCorrelation_Full(float* srcMat, nSize srcSize, float* mapMat, nSize mapSize, float* dstMat, nSize dstSize);
	float** cov(float** map, nSize mapSize, float** inputData, nSize inSize, int type); // 卷积操作

	// 这个是矩阵的上采样（等值内插），upc及upr是内插倍数
	float** UpSample(float** mat, nSize matSize, int upc, int upr);
	//void Mat2dUpSample(float** matIn, nSize matInSize, float** matOut, nSize matOutSize);
	void Mat2dUpSample(float* srcMat, nSize srcSize, float* dstMat, nSize dstSize);
	// 给二维矩阵边缘扩大，增加addw大小的0值边
	float** matEdgeExpand(float** mat, nSize matSize, int addc, int addr);
	void Mat2dEdgeExpand(float** matIn, nSize matSize, float** matOut, int addc, int addr);
	void Mat2dEdgeExpand_2(float* matIn, nSize matSize, float* matOut, int addc, int addr);
	// 给二维矩阵边缘缩小，擦除shrinkc大小的边
	float** matEdgeShrink(float** mat, nSize matSize, int shrinkc, int shrinkr);
	void Mat2dEdgeShrink(float** matIn, nSize matSize, float** matOut, int shrinkc, int shrinkr);
	/*******************************************************************************
	Function:		Mat2dEdgeShrink_w
	Description:
	Input:
	Output:		N/A
	Return:		0:			Successful
	ohters:		Failed
	*******************************************************************************/
	void Mat2dEdgeShrink_w(float *matIn, nSize matSize, float *matOut, int shrinkc, int shrinkr);

	void savemat(float** mat, nSize matSize, const char* filename);// 保存矩阵数据

	void multifactor(float** res, float** mat, nSize matSize, float factor);// 矩阵乘以系数
	void Mat2dMultiFactor(float* res, float* mat, nSize matSize, float factor);
	float Mat2dSum(float** mat, nSize matSize);// 矩阵各元素的和
	float Mat2dSum_float(float *mat, nSize matSize);
	char * combine_strings(char *a, char *b);

	char* intTochar(int i);


#ifdef __cplusplus
}
#endif
>>>>>>> init files

#endif