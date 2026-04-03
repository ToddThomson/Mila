module;
#include <vector>
#include <memory>
#include <iostream>
#include <cuda_fp16.h>
#include <stdexcept>
#include <type_traits>
#include <string>
#include "Kernels/Swiglu.cuh"

export module Compute.CudaSwigluOp:Dispatch;

import Dnn.Components.Swiglu;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.ComponentConfig;
import Compute.Precision;
import Compute.OperationBase;
import Compute.UnaryOperation;
import Compute.OperationRegistry;
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
            requires std::is_same_v<TNative, float> || std::is_same_v<TNative, half>
        struct cuda_swiglu_impl;

        template <>
        struct cuda_swiglu_impl<float>
        {
            cuda_swiglu_impl( const SwigluConfig& /*config*/ )
            {
            }

            inline void forward( float* Y, const float* X, int N, int half_width, cudaStream_t stream ) const
            {
                Mila::Dnn::Compute::Cuda::Swiglu::cuda_swiglu_forward_fp32( Y, X, N, half_width, stream );
            }

            inline void backward( float* dX, const float* X, const float* dY, int N, cudaStream_t stream ) const
            {
                Mila::Dnn::Compute::Cuda::Swiglu::cuda_swiglu_backward_fp32( dX, X, dY, N, stream );
            }
        };

        template <>
        struct cuda_swiglu_impl<half>
        {
            cuda_swiglu_impl( const SwigluConfig& /*config*/ )
            {
            }

            inline void forward( half* Y, const half* X, int N, int half_width, cudaStream_t stream ) const
            {
                throw std::runtime_error( "CudaSwigluOp: fp16 forward not implemented" );
            }

            inline void backward( half* dX, const half* X, const half* dY, int N, cudaStream_t stream ) const
            {
                throw std::runtime_error( "CudaSwigluOp: fp16 backward not implemented" );
            }
        };
    }
}