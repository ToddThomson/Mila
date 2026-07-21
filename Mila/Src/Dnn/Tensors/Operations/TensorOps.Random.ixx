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
#include <cmath>

export module Dnn.TensorOps:Random;

import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorOps.Base;
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

    /**
     * @brief Xavier/Glorot uniform initialization: U(-limit, limit), limit = sqrt(6 / (fan_in + fan_out)).
     *
     * Convenience over fill_uniform that computes the Glorot bound. Device-agnostic;
     * dispatches through fill_uniform to the backend RandomOps.
     */
    export template<TensorDataType TDataType, typename TMemoryResource>
        requires isValidTensor<TDataType, TMemoryResource> && TensorDataTypeTraits<TDataType>::is_float_type
    void xavier(
        Tensor<TDataType, TMemoryResource>& tensor,
        size_t fan_in,
        size_t fan_out,
        IExecutionContext* exec_context = nullptr )
    {
        const float limit = std::sqrt( 6.0f / static_cast<float>( fan_in + fan_out ) );

        fill_uniform( tensor,
            static_cast<host_value_t<TDataType>>( -limit ),
            static_cast<host_value_t<TDataType>>( limit ),
            exec_context );
    }
}
