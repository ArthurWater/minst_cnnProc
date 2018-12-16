/******************************************************************************
   Copyright 2018-2028 @
   All Rights Reserved
   FileName:    minst.c
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
#include <assert.h>
#include "../inc/common.h"
#include "../inc/minst.h"

/* 英特尔处理器和其他低端机用户必须翻转头字节。 */
int ReverseInt(int i)
{
    unsigned char ch1, ch2, ch3, ch4;

    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

/*******************************************************************************
   Function:		read_Img
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
MinstImgArr *read_Img(const char *filename) /* 读入图像 */
{
    FILE *fp = NULL;
    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;

    fp = fopen(filename, "rb");
    if (fp == NULL)
        PRT("open file failed\n");

    assert(fp);


    /* 从文件中读取sizeof(magic_number) 个字符到 &magic_number */
    fread((char *)&magic_number, sizeof(magic_number), 1, fp);
    magic_number = ReverseInt(magic_number);
    /* 获取训练或测试image的个数number_of_images */
    fread((char *)&number_of_images, sizeof(number_of_images), 1, fp);
    number_of_images = ReverseInt(number_of_images);
    /* 获取训练或测试图像的高度Heigh */
    fread((char *)&n_rows, sizeof(n_rows), 1, fp);
    n_rows = ReverseInt(n_rows);
    /* 获取训练或测试图像的宽度Width */
    fread((char *)&n_cols, sizeof(n_cols), 1, fp);
    n_cols = ReverseInt(n_cols);
    /* 获取第i幅图像，保存到vec中 */
    int i, r, c;

    /* 图像数组的初始化 */
    MinstImgArr *imgarr = (MinstImgArr *)malloc(sizeof(MinstImgArr));
    imgarr->ImgNum = number_of_images;
    imgarr->ImgPtr = (MinstImg *)malloc(number_of_images * sizeof(MinstImg));

    for (i = 0; i < number_of_images; ++i)
    {
        imgarr->ImgPtr[i].r = n_rows;
        imgarr->ImgPtr[i].c = n_cols;
        imgarr->ImgPtr[i].ImgData = (float **)malloc(n_rows * sizeof(float *));
        for (r = 0; r < n_rows; ++r)
        {
            imgarr->ImgPtr[i].ImgData[r] = (float *)malloc(n_cols * sizeof(float));
            for (c = 0; c < n_cols; ++c)
            {
                unsigned char temp = 0;
                fread((char *) &temp, sizeof(temp), 1, fp);
                imgarr->ImgPtr[i].ImgData[r][c] = (float)temp / 255.0;
            }
        }
    }

    fclose(fp);
    return imgarr;
}

/*******************************************************************************
   Function:		read_Lable
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
MinstLabelArr *read_Lable(const char *filename) /* 读入图像 */
{
    int i = 0;
    FILE *fp = NULL;
    int magic_number = 0;
    int number_of_labels = 0;
    int label_long = 10;

    if (filename)
    {
        printf(filename);
        printf("\n");
    }
    else
    {
        printf("%s[%d] point error\n", __FILE__, __LINE__);
    }

    fp = fopen(filename, "rb");
    if (fp == NULL)
        printf("open file %s failed\n", filename);

    assert(fp);

    /* 从文件中读取sizeof(magic_number) 个字符到 &magic_number */
    fread((char *)&magic_number, sizeof(magic_number), 1, fp);
    magic_number = ReverseInt(magic_number);

    /* 获取训练或测试image的个数number_of_images */
    fread((char *)&number_of_labels, sizeof(number_of_labels), 1, fp);
    number_of_labels = ReverseInt(number_of_labels);
    printf("[magic:0x%x][labelNum:0x%x]\n", magic_number, number_of_labels);
    /* 图像标记数组的初始化 */
    MinstLabelArr *labarr = (MinstLabelArr *)malloc(sizeof(MinstLabelArr));
    labarr->LabelNum = number_of_labels;
    labarr->LabelPtr = (MinstLabel *)malloc(number_of_labels * sizeof(MinstLabel));

    for (i = 0; i < number_of_labels; ++i)
    {
        labarr->LabelPtr[i].len = 10;
        labarr->LabelPtr[i].LabelData = (float *)calloc(label_long, sizeof(float));
        unsigned char temp = 0;
        fread((char *) &temp, sizeof(temp), 1, fp);
        labarr->LabelPtr[i].LabelData[(int)temp] = 1.0;
    }

    fclose(fp);
    return labarr;
}

int ReadMinstImg(MinstImgArr * imgArr, const char *filename) /* 读入图像 */
{
	FILE *fp = NULL;
	int magic_number = 0;
	int number_of_images = 0;
	int n_rows = 0;
	int n_cols = 0;

	fp = fopen(filename, "rb");
	if (fp == NULL)
		PRT("open file failed\n");

	assert(fp);


	/* 从文件中读取sizeof(magic_number) 个字符到 &magic_number */
	fread((char *)&magic_number, sizeof(magic_number), 1, fp);
	magic_number = ReverseInt(magic_number);
	/* 获取训练或测试image的个数number_of_images */
	fread((char *)&number_of_images, sizeof(number_of_images), 1, fp);
	number_of_images = ReverseInt(number_of_images);
	/* 获取训练或测试图像的高度Heigh */
	fread((char *)&n_rows, sizeof(n_rows), 1, fp);
	n_rows = ReverseInt(n_rows);
	/* 获取训练或测试图像的宽度Width */
	fread((char *)&n_cols, sizeof(n_cols), 1, fp);
	n_cols = ReverseInt(n_cols);
	/* 获取第i幅图像，保存到vec中 */
	int i, r, c;

	/* 图像数组的初始化 */
	imgArr->ImgNum = number_of_images;
	imgArr->ImgPtr = (MinstImg *)malloc(number_of_images * sizeof(MinstImg));

	for (i = 0; i < number_of_images; ++i)
	{
		imgArr->ImgPtr[i].r = n_rows;
		imgArr->ImgPtr[i].c = n_cols;
		imgArr->ImgPtr[i].ImgData = (float **)malloc(n_rows * sizeof(float *));
		for (r = 0; r < n_rows; ++r)
		{
			imgArr->ImgPtr[i].ImgData[r] = (float *)malloc(n_cols * sizeof(float));
			for (c = 0; c < n_cols; ++c)
			{
				unsigned char temp = 0;
				fread((char *)&temp, sizeof(temp), 1, fp);
				imgArr->ImgPtr[i].ImgData[r][c] = (float)temp / 255.0;
			}
		}
	}

	fclose(fp);
	return 0;
}

int minstReadLable(MinstLabelArr *labArr, const char *filename) /* 读入图像 */
{
	int i = 0;
	FILE *fp = NULL;
	int magic_number = 0;
	int number_of_labels = 0;
	int label_long = 10;

	if (filename)
	{
		printf(filename);
		printf("\n");
	}
	else
	{
		printf("%s[%d] point error\n", __FILE__, __LINE__);
	}
	if (!labArr)
	{
		return -1;
	}

	fp = fopen(filename, "rb");
	if (fp == NULL)
		printf("open file %s failed\n", filename);

	assert(fp);

	/* 从文件中读取sizeof(magic_number) 个字符到 &magic_number */
	fread((char *)&magic_number, sizeof(magic_number), 1, fp);
	magic_number = ReverseInt(magic_number);

	/* 获取训练或测试image的个数number_of_images */
	fread((char *)&number_of_labels, sizeof(number_of_labels), 1, fp);
	number_of_labels = ReverseInt(number_of_labels);
	printf("[magic:0x%x][labelNum:0x%x]\n", magic_number, number_of_labels);
	/* 图像标记数组的初始化 */

	labArr->LabelNum = number_of_labels;
	labArr->LabelPtr = (MinstLabel *)malloc(number_of_labels * sizeof(MinstLabel));

	for (i = 0; i < number_of_labels; ++i)
	{
		labArr->LabelPtr[i].len = 10;
		labArr->LabelPtr[i].LabelData = (float *)calloc(label_long, sizeof(float));
		unsigned char temp = 0;
		fread((char *)&temp, sizeof(temp), 1, fp);
		labArr->LabelPtr[i].LabelData[(int)temp] = 1.0;
	}

	fclose(fp);
	return 0;
}
/*******************************************************************************
   Function:		intTochar
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
char *intTochar(int i)/* 将数字转换成字符串 */
{
    int itemp = i;
    int w = 0;

    while (itemp >= 10)
    {
        itemp = itemp / 10;
        w++;
    }

    char *ptr = (char *)malloc((w + 2) * sizeof(char));
    ptr[w + 1] = '\0';
    int r; /* 余数 */
    while (i >= 10)
    {
        r = i % 10;
        i = i / 10;
        ptr[w] = (char)(r + 48);
        w--;
    }

    ptr[w] = (char)(i + 48);
    return ptr;
}

/*******************************************************************************
   Function:		combine_strings
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
char *combine_strings(char *a, char *b)  /* 将两个字符串相连 */
{
    char *ptr = NULL;
    int lena = strlen(a), lenb = strlen(b);
    int i, l = 0;

    if (!a || !b)
    {
        return NULL;
    }

    ptr = (char *)malloc((lena + lenb + 1) * sizeof(char));
    for (i = 0; i < lena; i++)
        ptr[l++] = a[i];

    for (i = 0; i < lenb; i++)
        ptr[l++] = b[i];

    ptr[l] = '\0';
    return (ptr);
}

/*******************************************************************************
   Function:		CombineString
   Description:
   Input:
   Output:		N/A
   Return:		0:			Successful
                    ohters:		Failed
*******************************************************************************/
VOID CombineString(CHAR *src1, CHAR *src2, CHAR *dst, UINT dstLen)  /* 将两个字符串相连 */
{
    int len1 = 0, len2 = 0;
    int i, l = 0;

    if (!src1 || !src2 || !dst)
    {
        return;
    }

    len1 = strlen(src1);
    len2 = strlen(src2);
    if ((len1 + len2) > dstLen)
    {
        PRT_ERR("dst len is not enough !\n");
        return;
    }

    for (i = 0; i < len1; i++)
        dst[l++] = src1[i];

    for (i = 0; i < len2; i++)
        dst[l++] = src2[i];

    dst[l] = '\0';

}

/*******************************************************************************
   Function:		SaveImgFile
   Description:
   Input:
   Output:		N/A
   Return:		NULL
*******************************************************************************/
void SaveImgFile(MinstImgArr *imgarr, char *filedir) /* 将图像数据保存成文件 */
{
    int i, r;
    FILE *fp = NULL;
    CHAR fileName[128] = {'\0'};

    if (!imgarr || !filedir)
    {
        PRT_ERR("input param error !\n");
        return;
    }

    for (i = 0; i < imgarr->ImgNum; i++)
    {
/*        const char *filename = combine_strings(filedir, combine_strings(intTochar(i), ".gray")); */
        sprintf(fileName, "%s%d%s", filedir, i, ".gray");
        fp = fopen(fileName, "wb");
        if (fp == NULL)
        {
            PRT_ERR("write file failed\n");
        }

        assert(fp);

        for (r = 0; r < imgarr->ImgPtr[i].r; r++){
            fwrite(imgarr->ImgPtr[i].ImgData[r], sizeof(FLOAT), imgarr->ImgPtr[i].c, fp);
        	}
        fclose(fp);
    }
}

