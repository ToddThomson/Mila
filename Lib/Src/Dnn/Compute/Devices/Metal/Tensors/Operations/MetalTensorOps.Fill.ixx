/**
 * @file MetalTensorOps.Fill.ixx
 * @brief Filling a Metal tensor from a scalar, by a kernel, and from host values, on the host after a wait.
 */

export module Compute.MetalTensorOps:Fill;

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

    export struct FillOps
    {
        template<TensorDataType TDataType>
        using host_value_t = std::conditional_t<TensorDataTypeTraits<TDataType>::is_integer_type, std::int32_t, float>;

        /**
         * @brief Copy host values into a tensor, converting each to its element.
         *
         * Runs on the host now, once the context's earlier work has finished. Fills the shorter of the tensor and the span.
         */
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static void fill(
            Tensor<TDataType, TMemoryResource>& tensor,
            std::span<const host_value_t<TDataType>> host_values,
            IExecutionContext* exec_context = nullptr )
        {
            if ( tensor.size() == 0 || host_values.empty() )
            {
                return;
            }

            if ( exec_context )
            {
                validateExecutionContext_<DeviceType::Metal>( exec_context, "Metal fill" )->synchronize();
            }

            using HostType = typename MetalTensorDataTypeMap<TDataType>::host_type;

            const dim_t count = std::min( tensor.size(), static_cast<dim_t>( host_values.size() ) );

            convertElements( host_values.data(), static_cast<HostType*>( static_cast<ITensor&>( tensor ).rawData() ),
                static_cast<std::size_t>( count ) );
        }

        /**
         * @brief Encode a kernel writing a scalar to every element.
         *
         * Metal converts the value. With a context the caller synchronizes; without one this waits before returning.
         */
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static void fill(
            Tensor<TDataType, TMemoryResource>& tensor,
            host_value_t<TDataType> host_value,
            IExecutionContext* exec_context = nullptr )
        {
            if ( tensor.size() == 0 )
            {
                return;
            }

            if ( exec_context )
            {
                encodeFill( tensor, host_value, *validateExecutionContext_<DeviceType::Metal>( exec_context, "Metal fill" ) );

                return;
            }

            MetalExecutionContext context( tensor.getDeviceId() );
            encodeFill( tensor, host_value, context );
            context.synchronize();
        }

    private:

        template<TensorDataType TDataType, typename TMemoryResource>
        static void encodeFill(
            Tensor<TDataType, TMemoryResource>& tensor,
            host_value_t<TDataType> host_value,
            MetalExecutionContext& context )
        {
            const std::string kernel_name = std::format( "fill_{}", MetalTensorDataTypeMap<TDataType>::kernel_type );
            const void* const destination = static_cast<ITensor&>( tensor ).rawData();

            context.dispatch(
                context.device().kernels().pipeline( kernel_name ),
                std::span<const void* const>( &destination, 1 ),
                std::as_bytes( std::span<const host_value_t<TDataType>>( &host_value, 1 ) ),
                static_cast<std::size_t>( narrowToKernelIndex( tensor.size() ) ) );
        }
    };
}
