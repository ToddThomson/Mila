module;
#include <vector>
#include <memory>
#include <iostream>
#include <cuda_bf16.h>
#include <stdexcept>
#include <type_traits>
#include <string>
#include "Kernels/Swiglu.cuh"

export module Compute.CudaSwigluOp:Dispatch;

import Dnn.Components.SwigluConfig;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.ComponentConfig;
import Compute.OperationBase;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.ExecutionContextTemplate;
import Compute.OperationType;
import Compute.MemoryResource;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaTensorDataType;
import Compute.CudaDevice;

namespace Mila::Dnn::Compute::Cuda::Swiglu
{
    namespace Detail
    {
        template <typename TNative>
            requires std::is_same_v<TNative, float>
                  || std::is_same_v<TNative, __nv_bfloat16>
        struct cuda_swiglu_impl;

        template <>
        struct cuda_swiglu_impl<float>
        {
            cuda_swiglu_impl( const SwigluConfig& /*config*/ )
            {
            }

            inline void forward( float* Y, const float* X, int N, int half_width, cudaStream_t stream ) const
            {
                cuda_swiglu_forward_fp32( Y, X, N, half_width, stream );
            }

            inline void backward( float* dX, const float* X, const float* dY, int N, int half_width, cudaStream_t stream ) const
            {
                cuda_swiglu_backward_fp32( dX, X, dY, N, half_width, stream );
            }
        };

        template <>
        struct cuda_swiglu_impl<__nv_bfloat16>
        {
            cuda_swiglu_impl( const SwigluConfig& /*config*/ )
            {
            }

            inline void forward( __nv_bfloat16* Y, const __nv_bfloat16* X, int N, int half_width, cudaStream_t stream ) const
            {
                cuda_swiglu_forward_bf16( Y, X, N, half_width, stream );
            }

            // dX and dY are FP32 — see Swiglu.Bf16.cu for mixed-precision backward contract.
            inline void backward( float* dX, const __nv_bfloat16* X, const float* dY, int N, int half_width, cudaStream_t stream ) const
            {
                cuda_swiglu_backward_bf16( dX, X, dY, N, half_width, stream );
            }
        };
    }
}