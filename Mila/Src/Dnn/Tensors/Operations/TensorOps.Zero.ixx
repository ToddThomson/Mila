/**
 * @file TensorOps.Zero.ixx
 * @brief Device-dispatched fast zero operation for tensor buffers.
 *
 * Provides the high-level, device-agnostic entry point `zero(...)` that
 * forwards to the backend `TensorOps<device>::zero(...)` implementation.
 */

module;
#include <type_traits>

export module Dnn.TensorOps:Zero;

import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeMap;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorOps.Base;
import Compute.DeviceType;
import Compute.ExecutionContext;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Zero a tensor using the fastest backend implementation.
     *
     * Forwards to the device-specific `TensorOps<device>::zero` implementation.
     *
     * @tparam TDataType Abstract tensor data type.
     * @tparam TMemoryResource Memory resource type backing the tensor.
     * @param tensor Destination tensor to be zeroed.
     * @param exec_context Optional execution context for stream control (borrowed).
     *
     * @note If exec_context is provided the backend should schedule the zero on the
     *       context's stream and avoid synchronizing. If exec_context is null the
     *       backend may synchronize before returning to provide synchronous semantics.
     */
    export template<TensorDataType TDataType, typename TMemoryResource>
        requires isValidTensor<TDataType, TMemoryResource>
    void zero( Tensor<TDataType, TMemoryResource>& tensor, IExecutionContext* exec_context = nullptr )
    {
        constexpr DeviceType device = TMemoryResource::device_type;
        TensorOps<device>::zero( tensor, exec_context );
    }
}