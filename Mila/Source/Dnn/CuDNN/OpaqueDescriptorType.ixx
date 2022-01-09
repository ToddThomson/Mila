/**
 * Copyright 2021 Todd Thomson, Achilles Software.  All rights reserved.
 *
 * Please refer to the ACHILLES end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

export module CuDnn.OpaqueDescriptorType;

namespace Mila::Dnn::CuDNN
{
    export typedef enum
    {
        CUDNN_RNN_DESCRIPTOR,
        CUDNN_RNNDATA_DESCRIPTOR,
        CUDNN_TENSOR_DESCRIPTOR,
        CUDNN_DROPOUT_DESCRIPTOR
    } opaqueDescriptorType_t;
}