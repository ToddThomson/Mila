/**
 * @file MetalTensorOps.Scale.ixx
 * @brief Multiplying a Metal tensor by a scalar.
 */

export module Compute.MetalTensorOps:Scale;

import std;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.MetalExecutionContext;
import Compute.MetalKernelLibrary;
import Compute.MetalTensorDataTypeMap;
import Dnn.ITensor;
import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorTypes;

namespace Mila::Dnn::Compute::Metal
{
    export struct ScaleOps
    {
        /**
         * @brief Encode result = input * scalar. The two may be the same tensor.
         *
         * With a context the caller synchronizes; without one this waits before returning.
         *
         * @throws std::invalid_argument For different shapes, or a context that is not Metal's.
         */
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static void scale(
            const Tensor<TDataType, TMemoryResource>& input,
            float scalar,
            Tensor<TDataType, TMemoryResource>& result,
            IExecutionContext* exec_context = nullptr )
        {
            if ( input.shape() != result.shape() )
            {
                throw std::invalid_argument( "scale: input and result must have the same shape" );
            }

            if ( input.size() == 0 )
            {
                return;
            }

            if ( exec_context )
            {
                encode( input, scalar, result, *validateExecutionContext_<DeviceType::Metal>( exec_context, "Metal scale" ) );

                return;
            }

            MetalExecutionContext context( input.getDeviceId() );
            encode( input, scalar, result, context );
            context.synchronize();
        }

    private:

        template<TensorDataType TDataType, typename TMemoryResource>
        static void encode(
            const Tensor<TDataType, TMemoryResource>& input,
            float scalar,
            Tensor<TDataType, TMemoryResource>& result,
            MetalExecutionContext& context )
        {
            const std::array<const void*, 2> buffers{
                static_cast<const ITensor&>( input ).rawData(), static_cast<ITensor&>( result ).rawData() };

            context.dispatch(
                context.device().kernels().pipeline( std::format( "scale_{}", MetalTensorDataTypeMap<TDataType>::kernel_type ) ),
                buffers, std::as_bytes( std::span( &scalar, 1 ) ),
                static_cast<std::size_t>( narrowToKernelIndex( input.size() ) ) );
        }
    };
}
