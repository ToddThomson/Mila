/**
 * @file TensorOps.Random.ixx
 * @brief Device-dispatching random initialization for tensors.
 *
 * Provides fill_normal and fill_uniform entry points, forwarding to device-specific
 * implementations (CPU, CUDA). For CUDA, uses cuRAND for efficient device-side fill.
 */

module;
#include <concepts>
#include <span>
#include <cstdint>

export module Dnn.TensorOps:Random;

import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorOps.Base;
import Compute.DeviceTraits;
import Compute.ExecutionContext;
import Compute.DeviceType;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    export template<TensorDataType TDataType, typename TMemoryResource>
        requires isValidTensor<TDataType, TMemoryResource>
    void fill_normal(
        Tensor<TDataType, TMemoryResource>& tensor,
        float mean,
        float stddev,
        IExecutionContext* exec_context = nullptr )
    {
        constexpr DeviceType device = TMemoryResource::device_type;
        TensorOps<device>::fill_normal( tensor, mean, stddev, exec_context );
    }

    export template<TensorDataType TDataType, typename TMemoryResource>
        requires isValidTensor<TDataType, TMemoryResource>
    void fill_uniform(
        Tensor<TDataType, TMemoryResource>& tensor,
        host_value_t<TDataType> min_val,
        host_value_t<TDataType> max_val,
        IExecutionContext* exec_context = nullptr )
    {
        constexpr DeviceType device = TMemoryResource::device_type;
        TensorOps<device>::fill_uniform( tensor, min_val, max_val, exec_context );
    }
}