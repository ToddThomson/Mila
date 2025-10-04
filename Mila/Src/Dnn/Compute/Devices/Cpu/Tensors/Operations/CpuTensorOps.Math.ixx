/**
 * @file CpuTensorOps.Math.ixx
 * @brief CPU tensor mathematical operations partition
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
import Dnn.TensorTypeMap;
import Dnn.TensorHostTypeMap;
import Compute.DeviceTraits;
import Compute.CpuMemoryResource;

namespace Mila::Dnn
{
    template<typename TComputeDeviceTag> struct TensorOps;

    export template<>
    struct TensorOps<Compute::CpuComputeDeviceTag>
    {
        /**
         * @brief Element-wise addition of two tensors (CPU implementation)
         *
         * Performs element-wise addition a[i] + b[i] for all elements.
         * Both tensors must have identical shapes.
         *
         * @tparam TDataType Abstract tensor data type
         * @tparam TMemoryResource Memory resource (must be CPU-accessible)
         * @param a First operand tensor
         * @param b Second operand tensor
         * @return New tensor containing the element-wise sum
         * @throws std::invalid_argument If tensor shapes don't match
         */
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static Tensor<TDataType, TMemoryResource> add( const Tensor<TDataType, TMemoryResource>& a, const Tensor<TDataType, TMemoryResource>& b ) {
            validateShapeCompatibility( a, b, "add" );

            Tensor<TDataType, TMemoryResource> result( a.getDeviceContext(), a.shape() );

            performElementwiseOperation( a, b, result, []( auto x, auto y ) { return x + y; } );

            return result;
        }

        /**
         * @brief Element-wise subtraction of two tensors (CPU implementation)
         *
         * Performs element-wise subtraction a[i] - b[i] for all elements.
         * Both tensors must have identical shapes.
         *
         * @tparam TDataType Abstract tensor data type
         * @tparam TMemoryResource Memory resource (must be CPU-accessible)
         * @param a First operand tensor (minuend)
         * @param b Second operand tensor (subtrahend)
         * @return New tensor containing the element-wise difference
         * @throws std::invalid_argument If tensor shapes don't match
         */
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static Tensor<TDataType, TMemoryResource> subtract( const Tensor<TDataType, TMemoryResource>& a, const Tensor<TDataType, TMemoryResource>& b ) {
            validateShapeCompatibility( a, b, "subtract" );

            Tensor<TDataType, TMemoryResource> result( a.getDeviceContext(), a.shape() );

            performElementwiseOperation( a, b, result, []( auto x, auto y ) { return x - y; } );

            return result;
        }

        /**
         * @brief Element-wise multiplication of two tensors (CPU implementation)
         *
         * Performs element-wise multiplication a[i] * b[i] for all elements.
         * Both tensors must have identical shapes.
         *
         * @tparam TDataType Abstract tensor data type
         * @tparam TMemoryResource Memory resource (must be CPU-accessible)
         * @param a First operand tensor
         * @param b Second operand tensor
         * @return New tensor containing the element-wise product
         * @throws std::invalid_argument If tensor shapes don't match
         */
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static Tensor<TDataType, TMemoryResource> multiply( const Tensor<TDataType, TMemoryResource>& a, const Tensor<TDataType, TMemoryResource>& b ) {
            validateShapeCompatibility( a, b, "multiply" );

            Tensor<TDataType, TMemoryResource> result( a.getDeviceContext(), a.shape() );

            performElementwiseOperation( a, b, result, []( auto x, auto y ) { return x * y; } );

            return result;
        }

        /**
         * @brief Element-wise division of two tensors (CPU implementation)
         *
         * Performs element-wise division a[i] / b[i] for all elements.
         * Both tensors must have identical shapes.
         *
         * @tparam TDataType Abstract tensor data type
         * @tparam TMemoryResource Memory resource (must be CPU-accessible)
         * @param a First operand tensor (dividend)
         * @param b Second operand tensor (divisor)
         * @return New tensor containing the element-wise quotient
         * @throws std::invalid_argument If tensor shapes don't match
         * @throws std::runtime_error If division by zero is detected
         */
        template<TensorDataType TDataType, typename TMemoryResource>
            requires isValidTensor<TDataType, TMemoryResource>
        static Tensor<TDataType, TMemoryResource> divide( const Tensor<TDataType, TMemoryResource>& a, const Tensor<TDataType, TMemoryResource>& b ) {
            validateShapeCompatibility( a, b, "divide" );

            Tensor<TDataType, TMemoryResource> result( a.getDeviceContext(), a.shape() );

            performElementwiseOperation( a, b, result, []( auto x, auto y ) {
                using HostType = typename TensorDataTypeMap<TDataType>::host_type;
                if constexpr (std::is_floating_point_v<HostType>) {
                    return x / y; // IEEE 754 handles division by zero
                }
                else {
                    if (y == static_cast<HostType>(0)) {
                        throw std::runtime_error( "Division by zero detected in integer division" );
                    }
                    return x / y;
                }
                } );

            return result;
        }

    private:
        /**
         * @brief Validates that two tensors have compatible shapes for element-wise operations
         *
         * @tparam TDataType Abstract tensor data type
         * @tparam TMemoryResource Memory resource type
         * @param a First tensor
         * @param b Second tensor
         * @param operation_name Name of the operation for error reporting
         * @throws std::invalid_argument If shapes don't match
         */
        template<TensorDataType TDataType, typename TMemoryResource>
        static void validateShapeCompatibility( const Tensor<TDataType, TMemoryResource>& a,
            const Tensor<TDataType, TMemoryResource>& b,
            const std::string& operation_name ) {
            if (a.shape() != b.shape()) {
                throw std::invalid_argument( operation_name + ": Tensor shapes must match for element-wise operations" );
            }

            if (a.empty() || b.empty()) {
                throw std::invalid_argument( operation_name + ": Cannot perform operations on empty tensors" );
            }
        }

        /**
         * @brief Performs element-wise operation using the provided binary function
         *
         * This helper function applies a binary operation to corresponding elements
         * of two input tensors and stores the result in the output tensor.
         * Uses parallel execution for improved performance on multi-core systems.
         *
         * @tparam TDataType Abstract tensor data type
         * @tparam TMemoryResource Memory resource type
         * @tparam TBinaryOp Binary operation function type
         * @param a First input tensor
         * @param b Second input tensor
         * @param result Output tensor (must be pre-allocated with correct shape)
         * @param op Binary operation to apply
         */
        template<TensorDataType TDataType, typename TMemoryResource, typename TBinaryOp>
            requires isValidTensor<TDataType, TMemoryResource>
        static void performElementwiseOperation( 
            const Tensor<TDataType, TMemoryResource>& a,
            const Tensor<TDataType, TMemoryResource>& b,
            Tensor<TDataType, TMemoryResource>& result,
            TBinaryOp op ) {

            // Get raw data pointers
            const void* a_data = a.rawData();
            const void* b_data = b.rawData();
            void* result_data = result.rawData();

            // Cast to appropriate host type for computation
            using HostType = typename TensorHostTypeMap<TDataType>::host_type;
            const auto* a_typed = static_cast<const HostType*>(a_data);
            const auto* b_typed = static_cast<const HostType*>(b_data);
            auto* result_typed = static_cast<HostType*>(result_data);

            const size_t num_elements = a.size();

            // Use parallel execution for better performance on large tensors
            // For small tensors, sequential execution might be faster due to overhead
            if (num_elements > 10000) {
                std::transform( std::execution::par_unseq,
                    a_typed, a_typed + num_elements,
                    b_typed,
                    result_typed,
                    op );
            }
            else {
                std::transform( a_typed, a_typed + num_elements,
                    b_typed,
                    result_typed,
                    op );
            }
        }
    };
}