/**
 * Copyright 2021 Todd Thomson, Achilles Software.  All rights reserved.
 *
 * Please refer to the ACHILLES end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
  */

#ifndef MILA_DNN_MODEL_OPTIONS_H_
#define MILA_DNN_MODEL_OPTIONS_H_

#include "NeuralNetType.h"

namespace Mila::Dnn
{
    class DnnModelOptions
    {
    public:
        DnnModelOptions() = default;

        neuralNetType_t nnType_ = RECURRENT_NN_TYPE;
    };
}
#endif