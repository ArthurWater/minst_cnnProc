

#include "../inc/cnn_inference.h"



/* 导入cnn的数据 */
int ImportCnnModelFile(CNN_NET_STR *pCnnNet, const char *filename)
{
    int i, j, c, r;
    float inData = 0.0;
    FILE *fp = NULL;
    long len = 0;

    if (!pCnnNet || !filename)
    {
        return -1;
    }

    fp = fopen(filename, "rb");
    if (fp == NULL)
    {
        PRT_ERR("open file failed\n");
        return -1;
    }

    /* C1的数据 */
    PRT("NET_C1->inChannels:%d, C1->outChannels:%d, C1->mapSize:%d\n", pCnnNet->stCovL1.inChannels, pCnnNet->stCovL1.outChannels, pCnnNet->stCovL1.mapSize);
    PRT("L1:w cof:\n");
    for (i = 0; i < pCnnNet->stCovL1.inChannels; i++)
    {
        for (j = 0; j < pCnnNet->stCovL1.outChannels; j++)
        {
            for (r = 0; r < pCnnNet->stCovL1.mapSize; r++)
            {
                for (c = 0; c < pCnnNet->stCovL1.mapSize; c++)
                {
                    fread(&inData, sizeof(float), 1, fp);
                    pCnnNet->stCovL1.mapData[i][j][r][c] = inData;
                    PRT("%f ", pCnnNet->stCovL1.mapData[i][j][r][c]);
                }

                PRT("\n");
            }

            PRT("\n");
        }
    }

    PRT("L1:basic data: ");
    for (i = 0; i < pCnnNet->stCovL1.outChannels; i++)
    {
        fread(&pCnnNet->stCovL1.basicData[i], sizeof(float), 1, fp);
        PRT("%f ", pCnnNet->stCovL1.basicData[i]);
    }

    PRT("\n");
    len = ftell(fp);
    PRT("file position %ld \n", len);

    /* C3网络 */
    PRT("C3->inChannels:%d, C3->outChannels:%d, C3->mapSize:%d\n", pCnnNet->stCovL3.inChannels, pCnnNet->stCovL3.outChannels, pCnnNet->stCovL3.mapSize);
    for (i = 0; i < pCnnNet->stCovL3.inChannels; i++)
    {
        for (j = 0; j < pCnnNet->stCovL3.outChannels; j++)
        {
            for (r = 0; r < pCnnNet->stCovL3.mapSize; r++)
            {
                for (c = 0; c < pCnnNet->stCovL3.mapSize; c++)
                {
                    fread(&inData, sizeof(float), 1, fp);
                    pCnnNet->stCovL3.mapData[i][j][r][c] = inData;
                }
            }
        }
    }

    PRT("L3:basic data: ");
    for (i = 0; i < pCnnNet->stCovL3.outChannels; i++)
    {
        fread(&pCnnNet->stCovL3.basicData[i], sizeof(float), 1, fp);
        PRT("%f ", pCnnNet->stCovL3.basicData[i]);
    }

    PRT("\n");
    len = ftell(fp);
    PRT("file position %ld \n", len);


    /* O5输出层 */
    PRT("O5->inputNum:%d, O5->outputNum:%d\n", pCnnNet->stOutL5.inputNum, pCnnNet->stOutL5.outputNum);
    for (i = 0; i < pCnnNet->stOutL5.outputNum; i++)
        for (j = 0; j < pCnnNet->stOutL5.inputNum; j++)
        {
            fread(&pCnnNet->stOutL5.wData[i][j], sizeof(float), 1, fp);
        }

    PRT("L5 cof w data\n");
    for (i = 0; i < CNN_LAYER5_OUT_CHANNEL_NUM; i++)
    {
        PRT("--%d--\n", i);
        for (j = 0; j < CNN_LAYER5_IN_DATA_NUM; j++)
        {
            PRT("%f ", pCnnNet->stOutL5.wData[i][j]);
            if ((j + 1) % 16 == 0)
            {
                PRT("\n");
            }
        }

        PRT("\n");
    }

    PRT("L5:basicData:");
    for (i = 0; i < pCnnNet->stOutL5.outputNum; i++)
    {
        fread(&pCnnNet->stOutL5.basicData[i], sizeof(float), 1, fp);
        PRT(" %f", pCnnNet->stOutL5.basicData[i]);
    }

    PRT("\n");
    len = ftell(fp);
    PRT("file position %ld \n", len);
    fclose(fp);

    return 0;
}





