/*
 * Copyright 2021 Todd Thomson, Achilles Software.  All rights reserved.
 *
 * Please refer to the ACHILLES end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#ifndef MILA_TESTS_CUDNN_COMMON_H_
#define MILA_TESTS_CUDNN_COMMON_H_

#include "cudnn.h"

import Dnn.RnnModelOptions;

using namespace Mila::Dnn;

namespace Mila::Tests::Dnn
{
	static inline RnnModelOptions GetDefaultRnnOptions()
	{
        return RnnModelOptions 
        {
            // dataType
            CUDNN_DATA_FLOAT, 
            
            // layout
            CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
            
            // maxSeqLength
            20,
            
            // batchSize 
            64,

            // vectorSize
            512
        };
	};
}
#endif