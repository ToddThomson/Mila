/**
 * @file CudaSwigluOp.ixx
 * @brief CUDA SwiGLU activation implementation
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
import :Dispatch;

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