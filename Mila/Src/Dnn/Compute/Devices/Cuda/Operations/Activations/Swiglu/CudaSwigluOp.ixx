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

import Dnn.Components.SwigluConfig;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.ComponentConfig;
import Compute.OperationBase;
import Compute.DeviceType;
// MSVC WORKAROUND, not a design decision: a consumer instantiating this operation must
// complete ExecutionContext<Cuda>, and MSVC 14.51 demands the type be VISIBLE where the
// standard -- and Clang 19+ -- accept it being merely reachable. Restore the plain import
// when that is fixed; nothing else here depends on the export.
export import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.ExecutionContextTemplate;
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
    class CudaSwigluOp : public Operation<DeviceType::Cuda, TPrecision>
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using TensorType = Tensor<TPrecision, MR>;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::device_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;

        CudaSwigluOp( IExecutionContext* context, const SwigluConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaSwigluOp" ) ), config_( config ), impl_( config )
        {
            config_.validate();
        }

        void forward( const ITensor& input, ITensor& output ) const
        {
            if ( input.size() % 2 != 0 )
            {
                throw std::invalid_argument( "CudaSwigluOp: Input must have even number of elements (split in half for SwiGLU)." );
            }

            const dim_t outSize = input.size() / 2;
            if ( output.size() != outSize )
            {
                throw std::invalid_argument( "CudaSwigluOp: Output must have half the size of the input for SwiGLU." );
            }

            int N = static_cast<int>(outSize);
            int half_width = static_cast<int>(input.shape().back() / 2);

            auto* cuda_context = static_cast<CudaExecutionContext*>(context_);
            cudaStream_t stream = cuda_context->getStream();

            auto X = static_cast<const NativeType*>(input.rawData());
            auto Y = static_cast<NativeType*>(output.rawData());

            impl_.forward( Y, X, N, half_width, stream );
        }

        void backward( const ITensor& input, const ITensor& output_gradient, ITensor& input_gradient ) const
        {
            if ( input.getDeviceType() != DeviceType::Cuda || output_gradient.getDeviceType() != DeviceType::Cuda || input_gradient.getDeviceType() != DeviceType::Cuda )
            {
                throw std::invalid_argument( "CudaSwigluOp::backward: All tensors must be on CUDA device." );
            }

            if ( input.size() % 2 != 0 )
            {
                throw std::invalid_argument( "CudaSwigluOp::backward: Input size must be even." );
            }

            const dim_t outSize = input.size() / 2;
            if ( output_gradient.size() != outSize || input_gradient.size() != input.size() )
            {
                throw std::invalid_argument( "CudaSwigluOp::backward: Gradient and input gradient sizes are incompatible." );
            }

            int N = static_cast<int>(outSize);
            int half_width = static_cast<int>(input.shape().back() / 2);

            auto* cuda_context = static_cast<CudaExecutionContext*>(context_);
            cudaStream_t stream = cuda_context->getStream();

            const NativeType* X = static_cast<const NativeType*>(input.rawData());

            if constexpr ( TPrecision == TensorDataType::BF16 )
            {
                const float* dY = static_cast<const float*>(output_gradient.rawData());
                float* dX = static_cast<float*>(input_gradient.rawData());
                impl_.backward( dX, X, dY, N, half_width, stream );
            }
            else
            {
                const NativeType* dY = static_cast<const NativeType*>(output_gradient.rawData());
                NativeType* dX = static_cast<NativeType*>(input_gradient.rawData());
                impl_.backward( dX, X, dY, N, half_width, stream );
            }
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

}