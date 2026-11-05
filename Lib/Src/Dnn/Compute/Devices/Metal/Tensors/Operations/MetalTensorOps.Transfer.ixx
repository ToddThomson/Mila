/**
 * @file MetalTensorOps.Transfer.ixx
 * @brief Copying to, from and between Metal tensors.
 */

export module Compute.MetalTensorOps:Transfer;

import std;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.MetalExecutionContext;
import Compute.MetalHostConversion;
import Compute.MetalTensorDataTypeMap;
import Dnn.ITensor;
import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorTypes;

namespace Mila::Dnn::Compute::Metal
{
    using namespace Mila::Dnn::Compute;

    export struct TransferOps
    {
        /**
         * @brief Copy between tensors of the same shape, at least one of them on Metal.
         *
         * Between two Metal tensors of one type this is encoded, in order with the context's other work. Everything else
         * runs on the host now, once the context's earlier work has finished, since the memory is shared and there is
         * nothing to transfer.
         *
         * @throws std::invalid_argument For different shapes, or a context that is not Metal's.
         */
        template<TensorDataType TSourceDataType, typename TSourceMemoryResource,
                 TensorDataType TDestinationDataType, typename TDestinationMemoryResource>
            requires isValidTensor<TSourceDataType, TSourceMemoryResource> && isValidTensor<TDestinationDataType, TDestinationMemoryResource>
        static void copy(
            const Tensor<TSourceDataType, TSourceMemoryResource>& source,
            Tensor<TDestinationDataType, TDestinationMemoryResource>& destination,
            IExecutionContext* exec_context = nullptr )
        {
            if ( source.shape() != destination.shape() )
            {
                throw std::invalid_argument( "Source and destination tensors must have the same shape for copy" );
            }

            if ( source.size() == 0 )
            {
                return;
            }

            MetalExecutionContext* const context = exec_context
                ? validateExecutionContext_<DeviceType::Metal>( exec_context, "Metal copy" ) : nullptr;

            const void* const from = static_cast<const ITensor&>( source ).rawData();
            void* const to = static_cast<ITensor&>( destination ).rawData();

            constexpr bool both_on_metal = TSourceMemoryResource::device_type == DeviceType::Metal
                && TDestinationMemoryResource::device_type == DeviceType::Metal;

            if constexpr ( both_on_metal && TSourceDataType == TDestinationDataType )
            {
                const std::size_t bytes = metalElementBytes<TSourceDataType>( source.size() );

                if ( context )
                {
                    context->copyBytes( from, to, bytes );

                    return;
                }

                MetalExecutionContext local( destination.getDeviceId() );
                local.copyBytes( from, to, bytes );
                local.synchronize();
            } else {
                if ( context )
                {
                    context->synchronize();
                }

                if constexpr ( TSourceDataType == TDestinationDataType )
                {
                    std::memcpy( to, from, metalElementBytes<TSourceDataType>( source.size() ) );
                } else {
                    using SourceType = typename MetalTensorDataTypeMap<TSourceDataType>::host_type;
                    using DestinationType = typename MetalTensorDataTypeMap<TDestinationDataType>::host_type;

                    convertElements( static_cast<const SourceType*>( from ), static_cast<DestinationType*>( to ),
                        static_cast<std::size_t>( source.size() ) );
                }
            }
        }
    };
}
