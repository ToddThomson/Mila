/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

module;
#include <cudnn.h>
#include <string>
#include <stdexcept>

export module CuDnn.Utils;

import Core.Version;
import CuDnn.Status;
import CuDnn.Helpers;

namespace Mila::Dnn::CuDNN 
{
    export inline std::string to_string( cudnnForwardMode_t forwardMode )
    {
        switch ( forwardMode )
        {
        case CUDNN_FWD_MODE_INFERENCE:
            return std::string( "CUDNN_FWD_MODE_INFERENCE" );
        case CUDNN_FWD_MODE_TRAINING:
            return std::string( "CUDNN_FWD_MODE_TRAINING" );
        default:
            return std::string( "WARN: Unknown cudnnForwardMode_t");
        }
    };

    export inline std::string to_string( cudnnRNNMode_t cellMode )
    {
        switch ( cellMode )
        {
        case CUDNN_RNN_RELU:
            return std::string( "CUDNN_RNN_RELU" );
        case CUDNN_RNN_TANH:
            return std::string( "CUDNN_RNN_TANH" );
        case CUDNN_LSTM:
            return std::string( "CUDNN_RNN_LSTM" );
        case CUDNN_GRU:
            return std::string( "CUDNN_RNN_GRU" );
        default:
            return std::string( "WARN: Unknown cudnnRNNMOdet" );
        }
    };

    export inline std::string to_string( cudnnRNNBiasMode_t biasMode )
    {
        switch ( biasMode )
        {
        case CUDNN_RNN_NO_BIAS:
            return std::string( "CUDNN_RNN_NO_BIAS" );
        case CUDNN_RNN_SINGLE_INP_BIAS:
            return std::string( "CUDNN_RNN_INP_BIAS" );
        case CUDNN_RNN_DOUBLE_BIAS:
            return std::string( "CUDNN_RNN_DOUBLE_BIAS" );
        case CUDNN_RNN_SINGLE_REC_BIAS:
            return std::string( "CUDNN_RNN_SINGLE_REC_BIAS" );
        default:
            return std::string( "WARN: Unknown cudnnRNNBiasMode_t" );
        }
    };

    export inline std::string to_string( cudnnDirectionMode_t directionMode )
    {
        switch ( directionMode )
        {
        case CUDNN_UNIDIRECTIONAL:
            return std::string( "CUDNN_UNIDIRECTIONAL" );
        case CUDNN_BIDIRECTIONAL:
            return std::string( "CUDNN_BIDIRECTIONAL" );
        default:
            return std::string( "WARN: Unknown cudnnDirectionMode_t" );
        }
    };

    export inline std::string to_string( cudnnRNNInputMode_t inputMode )
    {
        switch ( inputMode )
        {
        case CUDNN_LINEAR_INPUT:
            return std::string( "CUDNN_LINEAR_INPUT" );
        case CUDNN_SKIP_INPUT:
            return std::string( "CUDNN_SKIP_INPUT" );
        default:
            return std::string( "WARN: Unknown cudnnRNNInputMode_t" );

        }
    };

    static inline std::string to_string( cudnnRNNClipMode_t clipMode )
    {
        switch ( clipMode )
        {
        case CUDNN_RNN_CLIP_NONE:
            return std::string( "CUDNN_RNN_CLIP_NONE" );
        case CUDNN_RNN_CLIP_MINMAX:
            return std::string( "CUDNN_RNN_CLIP_MINMAX" );
        default:
            return std::string( "WARN: Unknown cudnnRNNClipMode_t" );
        }
    };

    export inline std::string to_string( cudnnRNNDataLayout_t dataLayout )
    {
        switch ( dataLayout )
        {
        case CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED:
            return std::string( "CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED" );
        case CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED:
            return std::string( "CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED" );
        case CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED:
            return std::string( "CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED" );
        default:
            return std::string( "WARN: Unknown cudnnRNNDataLayout_t" );
        }
    };

    export inline std::string to_string( cudnnRNNAlgo_t algorithm )
    {
        switch ( algorithm )
        {
        case CUDNN_RNN_ALGO_STANDARD:
            return std::string( "CUDNN_RNN_ALGO_STANDARD" );
        case CUDNN_RNN_ALGO_PERSIST_STATIC:
            return std::string( "CUDNN_RNN_ALGO_PERSIST_STATIC" );
        case CUDNN_RNN_ALGO_PERSIST_DYNAMIC:
            return std::string( "CUDNN_RNN_ALGO_PERSIST_DYNAMIC" );
        default:
            throw std::runtime_error( "WARN: Unknown cudnnRNNAlgo_t" );
        }
    };

    export inline std::string to_string( cudnnDataType_t type )
    {
        switch ( type )
        {
        case CUDNN_DATA_FLOAT:
            return std::string( "CUDNN_DATA_FLOAT" );
        case CUDNN_DATA_DOUBLE:
            return std::string( "CUDNN_DATA_DOUBLE" );
        case CUDNN_DATA_HALF:
            return std::string( "CUDNN_DATA_HALF" );
        case CUDNN_DATA_INT8:
            return std::string( "CUDNN_DATA_INT8" );
        case CUDNN_DATA_INT32:
            return std::string( "CUDNN_DATA_INT32" );
        case CUDNN_DATA_INT8x4: // x4 and x32 are replaced by vectorized dimension in the v8 API 
            return std::string( "CUDNN_DATA_INT8x4" );
        case CUDNN_DATA_UINT8:
            return std::string( "CUDNN_DATA_UINT8" );
        case CUDNN_DATA_UINT8x4: // x4 and x32 are replaced by vectorized dimension in the v8 API 
            return std::string( "CUDNN_DATA_UINT8x4" );
        case CUDNN_DATA_INT8x32: // x4 and x32 are replaced by vectorized dimension in the v8 API 
            return std::string( "CUDNN_DATA_INT8x32" );
        case CUDNN_DATA_INT64:
            return std::string( "CUDNN_DATA_INT64" );
        case CUDNN_DATA_BFLOAT16:
            return std::string( "CUDNN_DATA_BFLOAT16" );
        default:
            return std::string( "WARN: Unknown cudnnDataType_t" );
        }
    };

    export inline std::string to_string( cudnnStatus_t status_ )
    {
        switch ( status_ )
        {
        case CUDNN_STATUS_SUCCESS:
            return std::string( "CUDNN_STATUS_SUCCESS" );
        case CUDNN_STATUS_NOT_INITIALIZED:
            return std::string( "CUDNN_STATUS_NOT_INITIALIZED" );
        case CUDNN_STATUS_ALLOC_FAILED:
            return std::string( "CUDNN_STATUS_ALLOC_FAILED" );
        case CUDNN_STATUS_BAD_PARAM:
            return std::string( "CUDNN_STATUS_BAD_PARAM" );
        case CUDNN_STATUS_INTERNAL_ERROR:
            return std::string( "CUDNN_STATUS_INTERNAL_ERROR" );
        case CUDNN_STATUS_INVALID_VALUE:
            return std::string( "CUDNN_STATUS_INVALID_VALUE" );
        case CUDNN_STATUS_ARCH_MISMATCH:
            return std::string( "CUDNN_STATUS_ARCH_MISMATCH" );
        case CUDNN_STATUS_MAPPING_ERROR:
            return std::string( "CUDNN_STATUS_MAPPING_ERROR" );
        case CUDNN_STATUS_EXECUTION_FAILED:
            return std::string( "CUDNN_STATUS_EXECUTION_FAILED" );
        case CUDNN_STATUS_NOT_SUPPORTED:
            return std::string( "CUDNN_STATUS_NOT_SUPPORTED" );
        case CUDNN_STATUS_LICENSE_ERROR:
            return std::string( "CUDNN_STATUS_LICENSE_ERROR" );
        case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING:
            return std::string( "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING" );
        case CUDNN_STATUS_RUNTIME_IN_PROGRESS:
            return std::string( "CUDNN_STATUS_RUNTIME_IN_PROGRESS" );
        case CUDNN_STATUS_RUNTIME_FP_OVERFLOW:
            return std::string( "CUDNN_STATUS_RUNTIME_FP_OVERFLOW" );
        case CUDNN_STATUS_VERSION_MISMATCH:
            return std::string( "CUDNN_STATUS_VERSION_MISMATCH" );
        default:
            return std::string( "WARN: Unknown cudnnStatus_t" );
        }
    };

    export inline Mila::Core::Version GetVersion()
    {
        cudnnResult<int> result;
        int major = 0, minor = 0, patch = 1;

        result = getVersionPart( MAJOR_VERSION );

        if ( result.IsSuccess() )
            major = result.getValue();
        
        result = getVersionPart( MINOR_VERSION );

        if ( result.IsSuccess() )
            minor = result.getValue();

        result = getVersionPart( PATCH_LEVEL );

        if ( result.IsSuccess() )
            patch = result.getValue();

        return Mila::Core::Version( major, minor, patch );
    };
}