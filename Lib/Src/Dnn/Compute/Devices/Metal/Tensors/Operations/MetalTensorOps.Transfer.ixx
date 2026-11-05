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
import Serialization.Tensor;

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

        /**
         * @brief Copy a blob's bytes into a tensor of the blob's own element type.
         *
         * Runs on the host now, once the context's earlier work has finished: a blob is host memory and the tensor's is
         * shared, so it is complete on return and the reader may reuse the blob.
         *
         * @throws std::invalid_argument For different shapes, or a context that is not Metal's.
         */
        template<TensorDataType TDestinationDataType, typename TDestinationMemoryResource>
            requires isValidTensor<TDestinationDataType, TDestinationMemoryResource>
        static void copyFromBlob(
            const Serialization::ITensorBlob& blob,
            Tensor<TDestinationDataType, TDestinationMemoryResource>& destination,
            IExecutionContext* exec_context = nullptr )
        {
            validateBlob( blob, destination, exec_context );

            std::memcpy( static_cast<ITensor&>( destination ).rawData(), blob.data(), metalElementBytes<TDestinationDataType>( destination.size() ) );
        }

        /**
         * @brief Copy a blob into a tensor of another element type, converting each element.
         *
         * Runs on the host as `copyFromBlob` does. A value FP16 cannot hold is refused, never clipped: a wrong weight
         * would otherwise show only as a wrong answer (MetalBackend.md section 3.4).
         *
         * @throws std::invalid_argument For different shapes, a context that is not Metal's, or a value the destination
         *         cannot hold, naming the tensor.
         */
        template<TensorDataType TSourceDataType, TensorDataType TDestinationDataType, typename TDestinationMemoryResource>
            requires isValidTensor<TDestinationDataType, TDestinationMemoryResource>
        static void copyFromBlobWithConversion(
            const Serialization::ITensorBlob& blob,
            Tensor<TDestinationDataType, TDestinationMemoryResource>& destination,
            IExecutionContext* exec_context = nullptr )
        {
            validateBlob( blob, destination, exec_context );

            using SourceType = typename MetalTensorDataTypeMap<TSourceDataType>::host_type;
            using DestinationType = typename MetalTensorDataTypeMap<TDestinationDataType>::host_type;

            const auto* const from = static_cast<const SourceType*>( blob.data() );
            const auto count = static_cast<std::size_t>( destination.size() );

            if constexpr ( TDestinationDataType == TensorDataType::FP16 )
            {
                for ( std::size_t i = 0; i < count; ++i )
                {
                    const float value = static_cast<float>( from[i] );

                    if ( !( std::abs( value ) <= kLargestFp16 ) )
                    {
                        throw std::invalid_argument( std::format(
                            "Tensor '{}' holds {} at element {}, which is outside what FP16 can hold.",
                            destination.getName(), value, i ) );
                    }
                }
            }

            convertElements( from, static_cast<DestinationType*>( static_cast<ITensor&>( destination ).rawData() ), count );
        }

    private:

        static constexpr float kLargestFp16 = 65504.0f;

        template<TensorDataType TDestinationDataType, typename TDestinationMemoryResource>
        static void validateBlob(
            const Serialization::ITensorBlob& blob,
            const Tensor<TDestinationDataType, TDestinationMemoryResource>& destination,
            IExecutionContext* exec_context )
        {
            if ( blob.getMetadata().shape != destination.shape() )
            {
                throw std::invalid_argument( "Blob and destination tensor shapes must match" );
            }

            if ( exec_context )
            {
                validateExecutionContext_<DeviceType::Metal>( exec_context, "Metal copy from blob" )->synchronize();
            }
        }
    };
}
