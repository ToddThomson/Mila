/**
 * @file CpuTensorOps.Fill.ixx
 * @brief CPU tensor fill operations partition
 *
 * Provides CPU-specific implementations of tensor fill operations using
 * optimized standard library algorithms for host memory. All operations
 * execute synchronously with no device synchronization overhead.
 *
 * ExecutionContext handling:
 * - Accepts ExecutionContext parameter for API consistency with device implementations
 * - Parameter is unused for CPU operations (all operations are synchronous)
 * - No stream management needed on CPU
 */

module;
#include <cstring>
#include <algorithm>
#include <span>
#include <type_traits>
#include <stdexcept>

export module Compute.CpuTensorOps:Fill;

import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeMap;
import Dnn.TensorDataTypeTraits;
import Compute.DeviceType;
import Compute.DeviceTraits;
import Compute.CpuMemoryResource;
import Compute.CpuTensorDataTypeTraits;
import Compute.ExecutionContext;

namespace Mila::Dnn::Compute::Cpu
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
     * - Accepts ExecutionContext for API consistency (unused on CPU)
     */

    export struct FillOps
    {
        template<TensorDataType TDataType>
        using host_value_t = std::conditional_t<TensorDataTypeTraits<TDataType>::is_integer_type, int32_t, float>;

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
         * @param exec_context Optional execution context (unused for CPU, accepted for API consistency)
         *
         * @note Handles size mismatches gracefully (uses minimum size)
         * @note Type conversion handled automatically via static_cast
         * @note No synchronization needed - operations are synchronous on CPU
         * @note ExecutionContext parameter ignored but present for uniform API across devices
         */
        template<TensorDataType TDataType, typename TMemoryResource>
			requires isValidTensor<TDataType, TMemoryResource> /*&& std::is_same_v<TMemoryResource, CpuMemoryResource> */
        static void fill(
            Tensor<TDataType, TMemoryResource>& tensor,
            std::span<const host_value_t<TDataType>> host_values,
            [[maybe_unused]] ExecutionContext<DeviceType::Cpu>* exec_context = nullptr )
        {
            if (tensor.size() == 0 || host_values.empty())
                return;

            using HostValueType = host_value_t<TDataType>;
            using NativeType = typename CpuTensorDataTypeTraits::template native_type<TDataType>;

            const size_t count = std::min( tensor.size(), host_values.size() );
            NativeType* typed_dst = static_cast<NativeType*>(tensor.data());

            // Optimization: Direct copy when types match
            if constexpr (std::is_same_v<NativeType, HostValueType>)
            {
                std::copy_n( host_values.data(), count, typed_dst );
            }
            else
            {
                // Element-wise conversion when types differ
                std::transform( host_values.begin(),
                    host_values.begin() + count,
                    typed_dst,
                    []( HostValueType val ) {
                        return static_cast<NativeType>(val);
                    } );
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
         * @param exec_context Optional execution context (unused for CPU, accepted for API consistency)
         *
         * @note Conversion happens once before fill operation
         * @note No synchronization needed - operations are synchronous on CPU
         * @note ExecutionContext parameter ignored but present for uniform API across devices
         */
        template<TensorDataType TDataType, typename TMemoryResource>
			requires isValidTensor<TDataType, TMemoryResource> /* && std::is_same_v<TMemoryResource, CpuMemoryResource> */
        static void fill(
            Tensor<TDataType, TMemoryResource>& tensor,
            host_value_t<TDataType> host_value,
            [[maybe_unused]] ExecutionContext<DeviceType::Cpu>* exec_context = nullptr )
        {
            if (tensor.size() == 0)
                return;

            using NativeType = typename CpuTensorDataTypeTraits::template native_type<TDataType>;

            NativeType* typed_dst = static_cast<NativeType*>(tensor.data());
            NativeType native_value = static_cast<NativeType>(host_value);

            std::fill_n( typed_dst, tensor.size(), native_value );
        }
    };
}