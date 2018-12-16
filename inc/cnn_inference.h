/******************************************************************************
   Copyright 2018-2028 @
   All Rights Reserved
   FileName:    cnn_inferenc.h
   Description:
   Author:		meimaokui@126.com
   Date:		$(Time)
   Modification History: <version>      <time>      <author>        <desc>
   a)					  v1.0.0	   $(time)	  meimaokui@126.com	 Creat
******************************************************************************/
#ifndef _MMK_CNN_INFERENCE_H_
#define _MMK_CNN_INFERENCE_H_

#ifdef __cplusplus
extern "C" {
#endif



#include "mat.h"
#include "cnn.h"




int ImportCnnModelFile(CNN_NET_STR* pCnnNet, const char* filename);



#ifdef __cplusplus
}
#endif

#endif

