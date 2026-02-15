/**
 * @file CudaSwigluOp.ixx
 * @brief CUDA SwiGLU activation implementation (out = x1 * GELU(x2))
 */

module;
#include <vector>
#include <memory>
#include <iostream>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <type_traits>
#include <string>
#include "Kernels/Swiglu.cuh"
//#include "Kernels/Math.Elementwise.h"

export module Compute.CudaSwigluOp;

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
// DEPRECATED: import Compute.CudaDeviceResources;
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
        struct cuda_swiglu_impl<float> {
            cuda_swiglu_impl( const SwigluConfig& /*config*/ ) { /* no per-config selection yet */
            }

            inline void forward( float* Y, const float* X, int N, cudaStream_t stream ) const
            {
                // N is output size (half of input). Input layout: [ x1(0..N-1), x2(N..2N-1) ]
                const float* x1 = X;
                const float* x2 = X + N;

                // Compute gelu(x2) into Y (size N)
                Mila::Dnn::Compute::Cuda::Swiglu::cuda_gelu_forward_fp32( Y, x2, N, stream );

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
                if ( err != cudaSuccess ) {
                    // Fallback to synchronous cudaMalloc if cudaMallocAsync not available or fails
                    if ( cudaMalloc( reinterpret_cast<void**>(&temp), sizeof( float ) * static_cast<size_t>(N) ) != cudaSuccess ) {
                        throw std::runtime_error( "CudaSwigluOp: failed to allocate temporary buffer for backward pass" );
                    }
                    // Note: we will free with cudaFree below (synchronous)
                }

                // temp <- gelu(x2)
                Mila::Dnn::Compute::Cuda::Swiglu::cuda_gelu_forward_fp32( temp, x2, N, stream );

                // dX1 = dY * gelu(x2)  (write into dX1)
                // FIXME: launch_elementwise_multiply_kernel<float>( dY, temp, dX1, static_cast<size_t>(N), stream );

                // temp <- dY * x1  (reuse temp)
                // FIXME: launch_elementwise_multiply_kernel<float>( dY, x1, temp, static_cast<size_t>(N), stream );

                // dX2 = gelu_backward( x2, temp ) -> write into dX2
                Mila::Dnn::Compute::Cuda::Swiglu::cuda_gelu_backward_fp32( dX2, x2, temp, N, stream );

                // Free temp - attempt async free when possible
                if ( cudaFreeAsync != nullptr ) {
                    // cudaFreeAsync is available only on newer CUDA runtimes; use conditional compile would be ideal,
                    // but to keep simple, attempt cudaFreeAsync and fallback to cudaFree on failure.
                    cudaError_t fe = cudaFreeAsync( temp, stream );
                    if ( fe != cudaSuccess ) {
                        cudaFree( temp );
                    }
                }
                else {
                    cudaFree( temp );
                }
            }
        };

        template <>
        struct cuda_swiglu_impl<half> {
            cuda_swiglu_impl( const SwigluConfig& /*config*/ ) { /* Nothing to select for half yet */
            }

            inline void forward( half* Y, const half* X, int N, cudaStream_t stream ) const
            {
                const half* x1 = X;
                const half* x2 = X + N;

                Mila::Dnn::Compute::Cuda::Swiglu::cuda_gelu_forward_fp16( Y, x2, N, stream );
                // FIXMEL launch_elementwise_multiply_kernel<half>( x1, Y, Y, static_cast<size_t>(N), stream );
            }

            inline void backward( half* dX, const half* X, const half* dY, int N, cudaStream_t stream ) const
            {
                half* dX1 = dX;
                half* dX2 = dX + N;
                const half* x1 = X;
                const half* x2 = X + N;

                if ( N == 0 ) return;

                half* temp = nullptr;
                cudaError_t err = cudaMallocAsync( reinterpret_cast<void**>(&temp), sizeof( half ) * static_cast<size_t>(N), stream );
                if ( err != cudaSuccess ) {
                    if ( cudaMalloc( reinterpret_cast<void**>(&temp), sizeof( half ) * static_cast<size_t>(N) ) != cudaSuccess ) {
                        throw std::runtime_error( "CudaSwigluOp: failed to allocate temporary buffer for backward pass (FP16)" );
                    }
                }

                Mila::Dnn::Compute::Cuda::Swiglu::cuda_gelu_forward_fp16( temp, x2, N, stream );
                // FIXME: launch_elementwise_multiply_kernel<half>( dY, temp, dX1, static_cast<size_t>(N), stream );
                //launch_elementwise_multiply_kernel<half>( dY, x1, temp, static_cast<size_t>(N), stream );
                Mila::Dnn::Compute::Cuda::Swiglu::cuda_gelu_backward_fp16( dX2, x2, temp, N, stream );

                if ( cudaFreeAsync != nullptr ) {
                    cudaError_t fe = cudaFreeAsync( temp, stream );
                    if ( fe != cudaSuccess ) {
                        cudaFree( temp );
                    }
                }
                else {
                    cudaFree( temp );
                }
            }
        };
    }

    using namespace Mila::Dnn;

    export template<TensorDataType TPrecision>
        requires ValidFloatTensorDataType<TPrecision>
    class CudaSwigluOp : public UnaryOperation<DeviceType::Cuda, TPrecision>
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using UnaryOperationBase = UnaryOperation<DeviceType::Cuda, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::native_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;

        CudaSwigluOp( IExecutionContext* context, const SwigluConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaSwigluOp" ) ), config_( config ), impl_( config )
        {
            config_.validate();
        }

        void forward( const ITensor& input, ITensor& output ) const override
        {
            if ( input.getDeviceType() != DeviceType::Cuda || output.getDeviceType() != DeviceType::Cuda ) {
                throw std::invalid_argument( "CudaSwigluOp: Input and output tensors must be on CUDA device." );
            }

            if ( input.size() % 2 != 0 ) {
                throw std::invalid_argument( "CudaSwigluOp: Input must have even number of elements (split in half for SwiGLU)." );
            }

            const size_t outSize = input.size() / 2;
            if ( output.size() != outSize ) {
                throw std::invalid_argument( "CudaSwigluOp: Output must have half the size of the input for SwiGLU." );
            }

            int N = static_cast<int>(outSize);

            auto* cuda_context = static_cast<CudaExecutionContext*>(context_);
            cudaStream_t stream = cuda_context->getStream();

            auto X = static_cast<const NativeType*>(input.rawData());
            auto Y = static_cast<NativeType*>(output.rawData());

            impl_.forward( Y, X, N, stream );
        }

        void backward( const ITensor& input, const ITensor& output_gradient, ITensor& input_gradient ) const override
        {
            if ( input.getDeviceType() != DeviceType::Cuda || output_gradient.getDeviceType() != DeviceType::Cuda || input_gradient.getDeviceType() != DeviceType::Cuda ) {
                throw std::invalid_argument( "CudaSwigluOp::backward: All tensors must be on CUDA device." );
            }

            if ( input.size() % 2 != 0 ) {
                throw std::invalid_argument( "CudaSwigluOp::backward: Input size must be even." );
            }

            const size_t outSize = input.size() / 2;
            if ( output_gradient.size() != outSize || input_gradient.size() != input.size() ) {
                throw std::invalid_argument( "CudaSwigluOp::backward: Gradient and input gradient sizes are incompatible." );
            }

            int N = static_cast<int>(outSize);

            auto* cuda_context = static_cast<CudaExecutionContext*>(context_);
            cudaStream_t stream = cuda_context->getStream();

            const NativeType* X = static_cast<const NativeType*>(input.rawData());
            const NativeType* dY = static_cast<const NativeType*>(output_gradient.rawData());
            NativeType* dX = static_cast<NativeType*>(input_gradient.rawData());

            impl_.backward( dX, X, dY, N, stream );
        }

        OperationType getOperationType() const override {
            // No dedicated SwiGLU enum; classify as GeluOp for now.
            return OperationType::GeluOp;
        }

        std::string getName() const override {
            return "Cuda::SwigluOp";
        }

    private:
        SwigluConfig config_;
        CudaExecutionContext* context_;
        Detail::cuda_swiglu_impl<NativeType> impl_;
    };

    export class CudaSwigluOpRegistrar {
    public:
        static void registerOperations()
        {
            const std::string opName = "SwigluOp";

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP32, TensorDataType::FP32>(
                opName,
                []( IExecutionContext* context, const ComponentConfig& config )
                -> std::unique_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP32>>
                {
                    const auto& swigluConfig = static_cast<const SwigluConfig&>(config);
                    return std::make_unique<CudaSwigluOp<TensorDataType::FP32>>( context, swigluConfig );
                }
            );

            OperationRegistry::instance().registerUnaryOperation<DeviceType::Cuda, TensorDataType::FP16, TensorDataType::FP16>(
                opName,
                []( IExecutionContext* context, const ComponentConfig& config )
                -> std::unique_ptr<UnaryOperation<DeviceType::Cuda, TensorDataType::FP16>>
                {
                    const auto& swigluConfig = static_cast<const SwigluConfig&>(config);
                    return std::make_unique<CudaSwigluOp<TensorDataType::FP16>>( context, swigluConfig );
                }
            );
        }
    };
}