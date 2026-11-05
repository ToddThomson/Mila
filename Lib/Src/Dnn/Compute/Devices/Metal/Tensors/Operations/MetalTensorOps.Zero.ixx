/**
 * @file MetalTensorOps.Zero.ixx
 * @brief Zeroing a Metal tensor.
 */

export module Compute.MetalTensorOps:Zero;

import std;
import Compute.DeviceType;
import Compute.ExecutionContext;
import Compute.MetalExecutionContext;
import Compute.MetalTensorDataTypeMap;
import Dnn.ITensor;
import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;

namespace Mila::Dnn::Compute::Metal
{
    using namespace Mila::Dnn::Compute;

    export struct ZeroOps
    {
        /// Encode writing zero to the tensor's storage. With a context the caller synchronizes; without one this waits.
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static void zero( Dnn::Tensor<TDataType, TMemoryResource>& tensor, IExecutionContext* exec_context = nullptr )
        {
            void* const destination = static_cast<ITensor&>( tensor ).rawData();
            const std::size_t bytes = metalElementBytes<TDataType>( tensor.size() );

            if ( bytes == 0 || !destination )
            {
                return;
            }

            if ( exec_context )
            {
                validateExecutionContext_<DeviceType::Metal>( exec_context, "Metal zero" )->zeroBytes( destination, bytes );

                return;
            }

            MetalExecutionContext context( tensor.getDeviceId() );
            context.zeroBytes( destination, bytes );
            context.synchronize();
        }
    };
}
