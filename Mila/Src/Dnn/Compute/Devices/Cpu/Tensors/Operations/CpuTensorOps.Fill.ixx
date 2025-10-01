/**
 * @file CpuTensorOps.Fill.ixx
 * @brief CPU tensor fill operations partition
 */

module;
#include <cstring>
#include <algorithm>
#include <span>
#include <type_traits>
#include <stdexcept>

export module Dnn.TensorOps:Fill.Cpu;

import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeMap;
import Dnn.TensorDataTypeTraits;
import Compute.DeviceTraits;
import Compute.CpuMemoryResource;
import Compute.CpuTensorDataTypeTraits;

namespace Mila::Dnn
{
	using namespace Mila::Dnn::Compute;

    /**
     * @brief CPU specialization of TensorOps for initialization operations.
     *
     * This specialization provides CPU-specific implementations of tensor
     * fill operations for the compute device tag
     * `Compute::CpuComputeDeviceTag`.
     *
     * The implementations in this specialization operate on tensors that
     * satisfy the `isValidTensor` concept and use the tensor's device
     * context and memory resource to perform the operation.
     */
    template<typename TComputeDeviceTag> struct TensorOps;

    export template<>
    struct TensorOps<Compute::CpuComputeDeviceTag>
    {
        template<TensorDataType TDataType>
        using host_value_t = std::conditional_t<TensorDataTypeTraits<TDataType>::is_integer_type, int32_t, float>;
        
        /**
         * @brief Copy contiguous host values (host_type derived from TDataType) into tensor.
         *
         * Host span element type is selected from TensorDataTypeTraits<TDataType>::host_type,
         * ensuring the host-provided values match the expected host representation for the
         * tensor data type (float for floating tensors, int32_t for integer tensors).
         *
         * Implemented as `fill` overload that accepts a host span to match generic dispatch.
         */
        template<TensorDataType TDataType>
        static void fill(
            Tensor<TDataType, CpuMemoryResource>& tensor,
            std::span<const host_value_t<TDataType>> host_values )
        {
            if (tensor.size() == 0 || host_values.empty())
                return;

            using HostType = typename TensorDataTypeTraits<TDataType>::host_type;
            using TargetType = typename CpuTensorDataTypeTraits::template native_type<TDataType>;

            const size_t count = std::min( tensor.size(), host_values.size() );

            void* raw = tensor.rawData();
            if (!raw)
                return; // REVIEW: defensive

            TargetType* typed_dst = static_cast<TargetType*>(raw);

            if constexpr (std::is_same_v<TargetType, HostType>) {
                std::copy_n( host_values.data(), count, typed_dst );
            }
            else {
                for (size_t i = 0; i < count; ++i) {
                    typed_dst[i] = static_cast<TargetType>( host_values[i] );
                }
            }
        }

        /**
         * @brief Fill tensor with scalar host value.
         *
         * Host scalar type is the `host_type` defined by `TensorDataTypeTraits<TDataType>`.
         * Callers should pass the matching host scalar (float for float tensors, int32_t for integers).
         */
        template<TensorDataType TDataType>
        static void fill( Tensor<TDataType, CpuMemoryResource>& tensor, host_value_t<TDataType> host_value )
        {
            if (tensor.size() == 0)
                return;

            //using HostType = typename TensorDataTypeTraits<TDataType>::host_type;
            using TargetType = typename CpuTensorDataTypeTraits::template native_type<TDataType>;

            void* raw = tensor.rawData();
            if (!raw)
                return; // REVIEW: defensive

            TargetType* typed_dst = static_cast<TargetType*>(raw);

            TargetType quantized_value = static_cast<TargetType>(host_value);

            std::fill_n( typed_dst, tensor.size(), quantized_value );
        }
    };
}