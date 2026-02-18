/**
 * @file CudaGeluOp.Dispatch.ixx
 * @brief Implementation of the CUDA GELU kernel dispatch mechanism.
 */

module;
#include <vector>
#include <memory>
#include <iostream>
#include <cuda_fp16.h>
#include <stdexcept>
#include <type_traits>
#include <string>
#include "Kernels/Gelu.cuh"

export module Compute.CudaGeluOp:Dispatch;

import Dnn.Components.Gelu;
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

namespace Mila::Dnn::Compute::Cuda::Gelu
{
    namespace Detail
    {
        // Function pointer types for dispatch
        using ForwardFp32Func = void (*)(float*, const float*, int, cudaStream_t);
        using BackwardFp32Func = void (*)(float*, const float*, const float*, int, cudaStream_t);
        using ForwardFp16Func = void (*)(half*, const half*, int, cudaStream_t);
        using BackwardFp16Func = void (*)(half*, const half*, const half*, int, cudaStream_t);

        //// Forward and Backward declarations of FP32 implementations
        ////void gelu_exact_forward_fp32( float* Y, const float* X, int N, cudaStream_t stream );
        //void gelu_tanh_forward_fp32( float* Y, const float* X, int N, cudaStream_t stream );
        ////void gelu_sigmoid_forward_fp32( float* Y, const float* X, int N, cudaStream_t stream );
        //
        ////void gelu_exact_backward_fp32( float* dX, const float* X, const float* dY, int N, cudaStream_t stream );
        //void gelu_tanh_backward_fp32( float* dX, const float* X, const float* dY, int N, cudaStream_t stream );
        ////void gelu_sigmoid_backward_fp32( float* dX, const float* X, const float* dY, int N, cudaStream_t stream );

        // Primary template - will cause a compile error if no specialization exists
        template <typename TNative>
            requires std::is_same_v<TNative, float> || std::is_same_v<TNative, half>
        struct cuda_gelu_impl;

        template <>
        struct cuda_gelu_impl<float>
        {
            ForwardFp32Func forward_func;
            BackwardFp32Func backward_func;

            cuda_gelu_impl( const GeluConfig& config )
            {
                switch ( config.getApproximationMethod() )
                {
                    /*case ApproximationMethod::Exact:
                        forward_func = &gelu_exact_forward_fp32;
                        backward_func = &gelu_exact_backward_fp32;
                        break;
                    case ApproximationMethod::Sigmoid:
                        forward_func = &gelu_sigmoid_forward_fp32;
                        backward_func = &gelu_sigmoid_backward_fp32;
                        break;*/
                    case ApproximationMethod::Tanh:
                    default:
                        forward_func = &cuda_gelu_forward_fp32;
                        backward_func = &cuda_gelu_backward_fp32;
                        break;
                }
            }

            inline void forward( float* Y, const float* X, int N, cudaStream_t stream ) const
            {
                forward_func( Y, X, N, stream );
            }

            inline void backward( float* dX, const float* X, const float* dY, int N, cudaStream_t stream ) const
            {
                backward_func( dX, X, dY, N, stream );
            }
        };

        template <>
        struct cuda_gelu_impl<half>
        {
            cuda_gelu_impl( const GeluConfig& /*config*/ )
            { /* Nothing to select for half yet */
            }

            inline void forward( half* Y, const half* X, int N, cudaStream_t stream ) const
            {
                cuda_gelu_forward_fp16( Y, X, N, stream );
            }

            inline void backward( half* dX, const half* X, const half* dY, int N, cudaStream_t stream ) const
            {
                cuda_gelu_backward_fp16( dX, X, dY, N, stream );
            }
        };
    }
}