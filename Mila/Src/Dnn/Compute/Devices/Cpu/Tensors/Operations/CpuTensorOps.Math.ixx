/**
 * @file CpuTensorOps.Math.ixx
 * @brief CPU tensor mathematical operations partition
 *
 * Provides CPU implementations of element-wise mathematical operations.
 * CPU operations are synchronous and don't require ExecutionContext for
 * stream management, but accept it as a parameter for API consistency with
 * device implementations like CUDA.
 *
 * ExecutionContext handling:
 * - Accepts ExecutionContext parameter for API consistency with device implementations
 * - Parameter is unused for CPU operations (all operations are synchronous)
 * - No stream management needed on CPU
 */

module;
#include <memory>
#include <stdexcept>
#include <source_location>
#include <cmath>
#include <algorithm>
#include <execution>

export module Dnn.TensorOps:Math.Cpu;

import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeMap;
import Dnn.TensorHostTypeMap;
import Compute.DeviceType;
import Compute.DeviceTraits;
import Compute.CpuMemoryResource;
import Compute.ExecutionContext;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    template<DeviceType TDevice> struct TensorOps;

    /**
     * @brief CPU specialization of TensorOps for mathematical operations.
     *
     * Implements element-wise operations for CPU tensors using standard library
     * algorithms with optional parallel execution for large tensors.
     *
     * Key features:
     * - Synchronous execution (no stream management needed)
     * - Parallel execution for large tensors (>10000 elements)
     * - Accepts ExecutionContext for API consistency (unused on CPU)
     * - Automatic type conversion via TensorHostTypeMap
     * - Zero-copy direct memory access
     */
    export template<>
        struct TensorOps<DeviceType::Cpu>
    {
        /**
         * @brief Element-wise addition of two tensors (CPU implementation)
         *
         * Performs element-wise addition a[i] + b[i] for all elements and stores
         * the result in a pre-allocated result tensor. Both input tensors must have
         * identical shapes matching the result tensor shape.
         *
         * @tparam TDataType Abstract tensor data type
         * @tparam TMemoryResource Memory resource (must be CPU-accessible)
         * @param a First operand tensor
         * @param b Second operand tensor
         * @param result Pre-allocated result tensor (must have matching shape)
         * @param exec_context Optional execution context (unused for CPU, accepted for API consistency)
         * @throws std::invalid_argument If tensor shapes don't match or tensors are empty
         *
         * @note ExecutionContext parameter ignored but present for uniform API across devices
         * @note Uses parallel execution for tensors with >10000 elements
         */
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static void add(
            const Tensor<TDataType, TMemoryResource>& a,
            const Tensor<TDataType, TMemoryResource>& b,
            Tensor<TDataType, TMemoryResource>& result,
            [[maybe_unused]] ExecutionContext<DeviceType::Cpu>* exec_context = nullptr )
        {
            validateShapeCompatibility( a, b, result, "add" );

            performElementwiseOperation( a, b, result, []( auto x, auto y ) {
                return x + y;
                } );
        }

        /**
         * @brief Element-wise subtraction of two tensors (CPU implementation)
         *
         * Performs element-wise subtraction a[i] - b[i] for all elements and stores
         * the result in a pre-allocated result tensor.
         *
         * @tparam TDataType Abstract tensor data type
         * @tparam TMemoryResource Memory resource (must be CPU-accessible)
         * @param a First operand tensor (minuend)
         * @param b Second operand tensor (subtrahend)
         * @param result Pre-allocated result tensor (must have matching shape)
         * @param exec_context Optional execution context (unused for CPU, accepted for API consistency)
         * @throws std::invalid_argument If tensor shapes don't match or tensors are empty
         *
         * @note ExecutionContext parameter ignored but present for uniform API across devices
         * @note Uses parallel execution for tensors with >10000 elements
         */
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static void subtract(
            const Tensor<TDataType, TMemoryResource>& a,
            const Tensor<TDataType, TMemoryResource>& b,
            Tensor<TDataType, TMemoryResource>& result,
            [[maybe_unused]] ExecutionContext<DeviceType::Cpu>* exec_context = nullptr )
        {
            validateShapeCompatibility( a, b, result, "subtract" );

            performElementwiseOperation( a, b, result, []( auto x, auto y ) {
                return x - y;
                } );
        }

        /**
         * @brief Element-wise multiplication of two tensors (CPU implementation)
         *
         * Performs element-wise multiplication a[i] * b[i] for all elements and stores
         * the result in a pre-allocated result tensor.
         *
         * @tparam TDataType Abstract tensor data type
         * @tparam TMemoryResource Memory resource (must be CPU-accessible)
         * @param a First operand tensor
         * @param b Second operand tensor
         * @param result Pre-allocated result tensor (must have matching shape)
         * @param exec_context Optional execution context (unused for CPU, accepted for API consistency)
         * @throws std::invalid_argument If tensor shapes don't match or tensors are empty
         *
         * @note ExecutionContext parameter ignored but present for uniform API across devices
         * @note Uses parallel execution for tensors with >10000 elements
         */
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static void multiply(
            const Tensor<TDataType, TMemoryResource>& a,
            const Tensor<TDataType, TMemoryResource>& b,
            Tensor<TDataType, TMemoryResource>& result,
            [[maybe_unused]] ExecutionContext<DeviceType::Cpu>* exec_context = nullptr )
        {
            validateShapeCompatibility( a, b, result, "multiply" );

            performElementwiseOperation( a, b, result, []( auto x, auto y ) {
                return x * y;
                } );
        }

        /**
         * @brief Element-wise division of two tensors (CPU implementation)
         *
         * Performs element-wise division a[i] / b[i] for all elements and stores
         * the result in a pre-allocated result tensor.
         *
         * For floating-point types, follows IEEE 754 standards:
         * - Division by zero produces infinity or NaN
         * For integer types:
         * - Division by zero throws std::runtime_error
         *
         * @tparam TDataType Abstract tensor data type
         * @tparam TMemoryResource Memory resource (must be CPU-accessible)
         * @param a First operand tensor (dividend)
         * @param b Second operand tensor (divisor)
         * @param result Pre-allocated result tensor (must have matching shape)
         * @param exec_context Optional execution context (unused for CPU, accepted for API consistency)
         * @throws std::invalid_argument If tensor shapes don't match or tensors are empty
         * @throws std::runtime_error If division by zero in integer division
         *
         * @note ExecutionContext parameter ignored but present for uniform API across devices
         * @note Uses parallel execution for tensors with >10000 elements
         */
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static void divide(
            const Tensor<TDataType, TMemoryResource>& a,
            const Tensor<TDataType, TMemoryResource>& b,
            Tensor<TDataType, TMemoryResource>& result,
            [[maybe_unused]] ExecutionContext<DeviceType::Cpu>* exec_context = nullptr )
        {
            validateShapeCompatibility( a, b, result, "divide" );

            performElementwiseOperation( a, b, result, []( auto x, auto y ) {
                using HostType = typename TensorHostTypeMap<TDataType>::host_type;

                if constexpr (std::is_floating_point_v<HostType>)
                {
                    // IEEE 754 handles division by zero (produces inf or nan)
                    return x / y;
                }
                else
                {
                    // Integer division by zero is undefined behavior - throw
                    if (y == static_cast<HostType>(0))
                    {
                        throw std::runtime_error(
                            "Division by zero detected in integer division"
                        );
                    }
                    return x / y;
                }
                } );
        }

        /**
         * @brief Computes sum of all tensor elements (CPU implementation)
         *
         * Reduces tensor to a single scalar value representing the sum of all elements.
         * Uses parallel reduction for large tensors.
         *
         * @tparam TDataType Abstract tensor data type
         * @tparam TMemoryResource Memory resource (must be CPU-accessible)
         * @param tensor Input tensor
         * @param exec_context Optional execution context (unused for CPU, accepted for API consistency)
         * @return Sum of all elements as float
         *
         * @note ExecutionContext parameter ignored but present for uniform API across devices
         * @note Uses parallel reduction for tensors with >10000 elements
         * @note Always returns after computation (synchronous operation)
         */
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static float sum(
            const Tensor<TDataType, TMemoryResource>& tensor,
            [[maybe_unused]] ExecutionContext<DeviceType::Cpu>* exec_context = nullptr )
        {
            if (tensor.size() == 0)
            {
                return 0.0f;
            }

            using HostType = typename TensorHostTypeMap<TDataType>::host_type;
            const auto* data = static_cast<const HostType*>(tensor.data());
            const size_t num_elements = tensor.size();

            // Use parallel reduction for large tensors
            if (num_elements > 10000)
            {
                return std::reduce(
                    std::execution::par_unseq,
                    data, data + num_elements,
                    static_cast<float>(0),
                    []( float acc, HostType val ) {
                        return acc + static_cast<float>(val);
                    }
                );
            }
            else
            {
                float result = 0.0f;
                for (size_t i = 0; i < num_elements; ++i)
                {
                    result += static_cast<float>( data[i] );
                }
                return result;
            }
        }

    private:
        /**
         * @brief Validates that three tensors have compatible shapes for element-wise operations
         *
         * @tparam TDataType Abstract tensor data type
         * @tparam TMemoryResource Memory resource type
         * @param a First input tensor
         * @param b Second input tensor
         * @param result Result tensor
         * @param operation_name Name of the operation for error reporting
         * @throws std::invalid_argument If shapes don't match or tensors are empty
         */
        template<TensorDataType TDataType, typename TMemoryResource>
        static void validateShapeCompatibility(
            const Tensor<TDataType, TMemoryResource>& a,
            const Tensor<TDataType, TMemoryResource>& b,
            const Tensor<TDataType, TMemoryResource>& result,
            const std::string& operation_name )
        {
            if (a.shape() != b.shape() || a.shape() != result.shape())
            {
                throw std::invalid_argument(
                    operation_name + ": All tensor shapes must match for element-wise operations"
                );
            }

            if (a.empty() || b.empty() || result.empty())
            {
                throw std::invalid_argument(
                    operation_name + ": Cannot perform operations on empty tensors"
                );
            }
        }

        /**
         * @brief Performs element-wise operation using the provided binary function
         *
         * Applies a binary operation to corresponding elements of two input tensors
         * and stores the result in the output tensor. Uses parallel execution for
         * improved performance on large tensors (>10000 elements).
         *
         * @tparam TDataType Abstract tensor data type
         * @tparam TMemoryResource Memory resource type
         * @tparam TBinaryOp Binary operation function type
         * @param a First input tensor
         * @param b Second input tensor
         * @param result Output tensor (must be pre-allocated with correct shape)
         * @param op Binary operation to apply (e.g., std::plus, std::minus)
         */
        template<TensorDataType TDataType, typename TMemoryResource, typename TBinaryOp>
            requires isValidTensor<TDataType, TMemoryResource>
        static void performElementwiseOperation(
            const Tensor<TDataType, TMemoryResource>& a,
            const Tensor<TDataType, TMemoryResource>& b,
            Tensor<TDataType, TMemoryResource>& result,
            TBinaryOp op )
        {
            // Get raw data pointers
            const void* a_data = a.data();
            const void* b_data = b.data();
            void* result_data = result.data();

            // Cast to appropriate host type for computation
            using HostType = typename TensorHostTypeMap<TDataType>::host_type;
            const auto* a_typed = static_cast<const HostType*>(a_data);
            const auto* b_typed = static_cast<const HostType*>(b_data);
            auto* result_typed = static_cast<HostType*>(result_data);

            const size_t num_elements = a.size();

            // Use parallel execution for better performance on large tensors
            // Threshold of 10000 elements balances parallelization overhead
            // with performance gains from multi-core execution
            if (num_elements > 10000)
            {
                std::transform(
                    std::execution::par_unseq,
                    a_typed, a_typed + num_elements,
                    b_typed,
                    result_typed,
                    op
                );
            }
            else
            {
                std::transform(
                    a_typed, a_typed + num_elements,
                    b_typed,
                    result_typed,
                    op
                );
            }
        }
    };
}