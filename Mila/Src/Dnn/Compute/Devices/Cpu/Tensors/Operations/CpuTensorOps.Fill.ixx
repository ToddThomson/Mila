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
import Dnn.TensorTypeMap;
import Dnn.TensorTypeTraits;
import Compute.DeviceTraits;
import Compute.CpuMemoryResource;
import Compute.CpuTensorDataTypeTraits;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief CPU specialization of TensorOps for initialization operations.
     *
     * Provides CPU-specific implementations of tensor fill operations using
     * optimized standard library algorithms for host memory. All operations
     * execute synchronously with no device synchronization overhead.
     *
     * Key features:
     * - Direct memory access to CPU tensors
     * - STL algorithm optimizations (vectorization, cache efficiency)
     * - Automatic type conversion when needed
     * - Compile-time dispatch for optimal code generation
     */
    template<typename TComputeDeviceTag> struct TensorOps;

    export template<>
    struct TensorOps<Compute::CpuComputeDeviceTag>
    {
        template<TensorDataType TDataType>
        using host_value_t = std::conditional_t<
            TensorDataTypeTraits<TDataType>::is_integer_type, int32_t, float>;

        /**
         * @brief Fill tensor with array of host values
         *
         * Copies host values into CPU tensor with automatic type conversion.
         * Uses optimized STL algorithms for performance.
         *
         * Implementation:
         * - Direct copy when types match (zero conversion overhead)
         * - Element-wise transform when conversion needed
         * - Compile-time dispatch based on type compatibility
         *
         * @tparam TDataType Abstract tensor data type
         * @param tensor Destination CPU tensor to fill
         * @param host_values Span of host values in canonical representation
         *
         * @note Handles size mismatches gracefully (uses minimum size)
         * @note Type conversion handled automatically via static_cast
         * @note No synchronization needed - operations are synchronous on CPU
         */
        template<TensorDataType TDataType>
        static void fill(Tensor<TDataType, CpuMemoryResource>& tensor, 
                        std::span<const host_value_t<TDataType>> host_values)
        {
            if (tensor.size() == 0 || host_values.empty())
                return;

            using HostValueType = host_value_t<TDataType>;
            using NativeType = typename CpuTensorDataTypeTraits::template native_type<TDataType>;

            const size_t count = std::min(tensor.size(), host_values.size());
            NativeType* typed_dst = static_cast<NativeType*>(tensor.rawData());

            // Optimization: Direct copy when types match
            if constexpr (std::is_same_v<NativeType, HostValueType>) {
                std::copy_n(host_values.data(), count, typed_dst);
            }
            else {
                // Element-wise conversion when types differ
                std::transform(host_values.begin(), 
                             host_values.begin() + count,
                             typed_dst,
                             [](HostValueType val) { 
                                 return static_cast<NativeType>(val); 
                             });
            }
        }

        /**
         * @brief Fill tensor with scalar host value
         *
         * Broadcasts a single scalar value to all tensor elements using
         * optimized STL fill algorithm.
         *
         * Implementation:
         * - Single type conversion (scalar -> native type)
         * - Optimized std::fill_n (vectorized by compiler)
         * - Compile-time type selection
         *
         * @tparam TDataType Abstract tensor data type
         * @param tensor Destination CPU tensor to fill
         * @param host_value Scalar value in canonical host representation
         *
         * @note Conversion happens once before fill operation
         * @note No synchronization needed - operations are synchronous on CPU
         */
        template<TensorDataType TDataType>
        static void fill(Tensor<TDataType, CpuMemoryResource>& tensor, 
                        host_value_t<TDataType> host_value)
        {
            if (tensor.size() == 0)
                return;

            using NativeType = typename CpuTensorDataTypeTraits::template native_type<TDataType>;

            NativeType* typed_dst = static_cast<NativeType*>(tensor.rawData());
            NativeType native_value = static_cast<NativeType>(host_value);

            std::fill_n(typed_dst, tensor.size(), native_value);
        }
    };
}