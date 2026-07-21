/**
 * @file CudaElementwiseActivationOp.ixx
 * @brief CUDA implementation of the functor-templated elementwise activation op.
 *
 * Templated on precision and a compile-time functor (from the shared
 * ElementwiseActivation library). The op launches its single specialized kernel
 * directly -- no host switch in forward() -- so the per-element hot path is
 * branch-free (see Specifications/FfnAndMoE.md section 5.1).
 */

module;
#include <memory>
#include <stdexcept>
#include <string>
#include <cuda_runtime_api.h>
#include "Kernels/ElementwiseActivation.cuh"

export module Compute.CudaElementwiseActivationOp;

import Dnn.Components.ActivationConfig;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.Component;
import Compute.OperationBase;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.OperationType;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaTensorDataType;

namespace Mila::Dnn::Compute::Cuda::Activation
{
    using namespace Mila::Dnn;

    /**
     * @brief CUDA elementwise activation operation.
     *
     * @tparam TPrecision Activation/compute precision (FP32 or BF16).
     * @tparam TFunctor   POD functor from Mila::Dnn::Activations exposing fwd/df.
     *
     * The functor is fixed at compile time by the Activation component; this op holds
     * an instance so function-specific scalars (LeakyReLU alpha) are carried by value.
     */
    export template<TensorDataType TPrecision, typename TFunctor>
        requires ValidFloatTensorDataType<TPrecision>
    class CudaElementwiseActivationOp : public Operation<DeviceType::Cuda, TPrecision>
    {
    public:
        using MR = CudaDeviceMemoryResource;
        using TensorType = Tensor<TPrecision, MR>;
        using NativeType = typename Mila::Dnn::Compute::Cuda::TensorDataTypeMap<TPrecision>::device_type;
        using CudaExecutionContext = ExecutionContext<DeviceType::Cuda>;

        CudaElementwiseActivationOp( IExecutionContext* context, const ActivationConfig& config )
            : context_( validateExecutionContext_<DeviceType::Cuda>( context, "CudaElementwiseActivationOp" ) )
        {
            if constexpr ( requires( TFunctor f ) { f.alpha; } )
            {
                functor_.alpha = config.getLeakyReluAlpha();
            }
        }

        void forward( const ITensor& input, ITensor& output ) const
        {
            if ( input.getDeviceType() != DeviceType::Cuda || output.getDeviceType() != DeviceType::Cuda )
            {
                throw std::invalid_argument( "CudaElementwiseActivationOp: tensors must be on CUDA device." );
            }

            if ( output.size() < input.size() )
            {
                throw std::invalid_argument( "CudaElementwiseActivationOp: output size must be >= input size." );
            }

            cudaStream_t stream = context_->getStream();

            auto X = static_cast<const NativeType*>( input.rawData() );
            auto Y = static_cast<NativeType*>( output.rawData() );
            int N = static_cast<int>( input.size() );

            launch_elementwise_forward<NativeType, TFunctor>( Y, X, N, functor_, stream );
        }

        void backward( const ITensor& input, const ITensor& output_gradient, ITensor& input_gradient ) const
        {
            if ( input.getDeviceType() != DeviceType::Cuda
                || output_gradient.getDeviceType() != DeviceType::Cuda
                || input_gradient.getDeviceType() != DeviceType::Cuda )
            {
                throw std::invalid_argument( "CudaElementwiseActivationOp::backward: tensors must be on CUDA device." );
            }

            cudaStream_t stream = context_->getStream();

            const NativeType* X = static_cast<const NativeType*>( input.rawData() );
            const NativeType* dY = static_cast<const NativeType*>( output_gradient.rawData() );
            NativeType* dX = static_cast<NativeType*>( input_gradient.rawData() );
            int N = static_cast<int>( input.size() );

            launch_elementwise_backward<NativeType, TFunctor>( dX, X, dY, N, functor_, stream );
        }

        OperationType getOperationType() const override
        {
            return OperationType::ElementwiseActivationOp;
        }

        std::string getName() const override
        {
            return "Cuda::ElementwiseActivationOp";
        }

    private:
        CudaExecutionContext* context_;
        TFunctor functor_{};
    };
}
