
#ifndef __COMMON__
#define __COMMON__

#ifdef __cplusplus
extern "C" {
#endif

#define TRUE 1
#define FALSE 0


#define PRT printf
#define PRT_ERR printf("error: %s[%d]", __FILE__, __LINE__);printf

#define RET_CHEAK_ZERO(x) do{if(0 != x){printf("ret value error:%d\n", x);}}while(0);
#define CHEAK_VALUE_ZERO(x) do{if(0 == x){printf("value is zero error:%d\n", x);}}while(0);
#define CHEAK_POINT_NULL(p) do{if(NULL == (void *)p){printf("%s[%d] param point null !\n", __FILE__, __LINE__); exit(0);}}while(0);


#define CNN_LAYER_NUM 5
#define CNN_MAP_SIZE 5
#define CNN_LAYER1_MAP_SIZE 5


#define PIC_TEST_NUM 100
#define CNN_PIC_TRAIN_NUM 55000


#if 0
#define CNN_TEST_LABELS_PATH "input/t10k-labels.idx1-ubyte"
#define CNN_TEST_IMAGES_PATH "input/t10k-images.idx3-ubyte"

#define CNN_TRAIN_LABELS_PATH "input/train-labels.idx1-ubyte"
#define CNN_TRAIN_IMAGES_PATH "input/train-images.idx3-ubyte"

#define CNN_MODEL_FILE_PATH "src/minst.cnn"

#define CNN_MODEL_FILE_SAVE_PATH "output/minst.cnn"

#define CNN_PIC_TRAIN_L_ERROR_DATA_PATH "PicTrans/cnnL_new.ma"

#else
#define CNN_TEST_LABELS_PATH	"D:\\Embedded Work\\GPU\\CUDA\\CudaDemo\\cnnDemo\\input\\t10k-labels.idx1-ubyte"
#define CNN_TEST_IMAGES_PATH	"D:\\Embedded Work\\GPU\\CUDA\\CudaDemo\\cnnDemo\\input\\t10k-images.idx3-ubyte"

#define CNN_TRAIN_LABELS_PATH "D:\\Embedded Work\\GPU\\CUDA\\CudaDemo\\cnnDemo\\input\\train-labels.idx1-ubyte"
#define CNN_TRAIN_IMAGES_PATH "D:\\Embedded Work\\GPU\\CUDA\\CudaDemo\\cnnDemo\\input\\train-images.idx3-ubyte"

#define CNN_MODEL_FILE_PATH "D:\\Embedded Work\\GPU\\CUDA\\CudaDemo\\cnnDemo\\input\\minst.cnn"

#define CNN_MODEL_FILE_SAVE_PATH "D:\\Embedded Work\\GPU\\CUDA\\CudaDemo\\cnnDemo\\output\\minst.cnn"

#define CNN_PIC_TRAIN_L_ERROR_DATA_PATH "D:\\Embedded Work\\GPU\\CUDA\\CudaDemo\\cnnDemo\\PicTrans\\cnnL_new.ma"



#endif

#ifdef __cplusplus
}
#endif

#endif
