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
        // Primary template - will cause a compile error if no specialization exists
        template <typename TNative>
            requires std::is_same_v<TNative, float> || std::is_same_v<TNative, half>
        struct cuda_swiglu_impl;

        template <>
        struct cuda_swiglu_impl<float>
        {
            cuda_swiglu_impl( const SwigluConfig& /*config*/ )
            { /* no per-config selection yet */
            }

            inline void forward( float* Y, const float* X, int N, cudaStream_t stream ) const
            {
                // N is output size (half of input). Input layout: [ x1(0..N-1), x2(N..2N-1) ]
                const float* x1 = X;
                const float* x2 = X + N;

                // Compute gelu(x2) into Y (size N)
                Mila::Dnn::Compute::Cuda::Swiglu::cuda_swiglu_forward_fp32( Y, x2, N, stream );

                // Y = x1 * gelu(x2)
                // FIXME: launch_elementwise_multiply_kernel<float>( x1, Y, Y, static_cast<size_t>(N), stream );
            }

            inline void backward( float* dX, const float* X, const float* dY, int N, cudaStream_t stream ) const
            {
                // dX layout: [dX1(0..N-1), dX2(N..2N-1)]
                float* dX1 = dX;
                float* dX2 = dX + N;
                const float* x1 = X;
                const float* x2 = X + N;

                if ( N == 0 ) return;

                // Allocate temporary buffer on device for intermediate values
                float* temp = nullptr;
                cudaError_t err = cudaMallocAsync( reinterpret_cast<void**>(&temp), sizeof( float ) * static_cast<size_t>(N), stream );
                if ( err != cudaSuccess )
                {
                    // Fallback to synchronous cudaMalloc if cudaMallocAsync not available or fails
                    if ( cudaMalloc( reinterpret_cast<void**>(&temp), sizeof( float ) * static_cast<size_t>(N) ) != cudaSuccess )
                    {
                        throw std::runtime_error( "CudaSwigluOp: failed to allocate temporary buffer for backward pass" );
                    }
                    // Note: we will free with cudaFree below (synchronous)
                }

                // temp <- gelu(x2)
                Mila::Dnn::Compute::Cuda::Swiglu::cuda_swiglu_forward_fp32( temp, x2, N, stream );

                // dX1 = dY * gelu(x2)  (write into dX1)
                // FIXME: launch_elementwise_multiply_kernel<float>( dY, temp, dX1, static_cast<size_t>(N), stream );

                // temp <- dY * x1  (reuse temp)
                // FIXME: launch_elementwise_multiply_kernel<float>( dY, x1, temp, static_cast<size_t>(N), stream );

                // dX2 = gelu_backward( x2, temp ) -> write into dX2
                Mila::Dnn::Compute::Cuda::Swiglu::cuda_swiglu_backward_fp32( dX2, x2, temp, N, stream );

                // Free temp - attempt async free when possible
                if ( cudaFreeAsync != nullptr )
                {
                    // cudaFreeAsync is available only on newer CUDA runtimes; use conditional compile would be ideal,
                    // but to keep simple, attempt cudaFreeAsync and fallback to cudaFree on failure.
                    cudaError_t fe = cudaFreeAsync( temp, stream );
                    if ( fe != cudaSuccess )
                    {
                        cudaFree( temp );
                    }
                }
                else
                {
                    cudaFree( temp );
                }
            }
        };

        template <>
        struct cuda_swiglu_impl<half>
        {
            cuda_swiglu_impl( const SwigluConfig& /*config*/ )
            { /* Nothing to select for half yet */
            }

            inline void forward( half* Y, const half* X, int N, cudaStream_t stream ) const
            {
                const half* x1 = X;
                const half* x2 = X + N;

                // TODO:Mila::Dnn::Compute::Cuda::Swiglu::cuda_swiglu_forward_fp16( Y, x2, N, stream );
            }

            inline void backward( half* dX, const half* X, const half* dY, int N, cudaStream_t stream ) const
            {
                // TODO: Implement backward for half precision
            }
        };
    }
}