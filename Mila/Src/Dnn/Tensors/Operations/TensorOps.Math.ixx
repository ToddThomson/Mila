/**
 * @file TensorOps.Math.ixx
 * @brief Device-dispatching math helpers for tensor arithmetic operations
 *
 * This partition provides the high-level, device-agnostic entry points for
 * tensor math operations (e.g., element-wise addition). Each helper forwards
 * to the device-specific `TensorOps<ComputeDeviceTag>::...` implementation
 * (see CPU and CUDA specializations).
 *
 * The templates are constrained with `isValidTensor<TDataType, TMemoryResource>`
 * to ensure the tensor configuration is valid (memory resource compatibility,
 * type traits available, and device accessibility).
 *
 * ExecutionContext handling:
 *   - Optional ExecutionContext parameter for stream control (borrowed, not owned)
 *   - When provided, operations use the context's stream (caller controls sync)
 *   - When null, operations use default stream and synchronize before returning
 *   - Raw pointer semantics ensure zero overhead
 *
 * Usage:
 *   - Call `add(a, b, result)` for element-wise addition of two tensors with the same
 *     abstract data type and memory resource. The call is automatically
 *     dispatched to the appropriate device implementation.
 *   - Optionally provide ExecutionContext for explicit stream control:
 *     `add(a, b, result, ctx.get())`
 *
 * Preconditions:
 *   - All operands must satisfy `isValidTensor` and have matching shapes.
 *   - Result tensor must be pre-allocated with matching shape.
 *   - Device-specific implementations validate shapes and perform operations efficiently.
 *   - ExecutionContext (if provided) must outlive the function call.
 */

module;
#include <concepts>

export module Dnn.TensorOps:Math;
export import :Math.Cpu;
//export import :Math.Cuda;

import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeMap;
import Compute.DeviceTraits;
import Compute.ExecutionContext;
import Compute.DeviceType;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Element-wise addition with optional ExecutionContext (device-dispatched).
     *
     * Computes result[i] = a[i] + b[i] for all elements. Automatically dispatches
     * to the appropriate device implementation based on memory resource type.
     *
     * @tparam TDataType Abstract tensor data type
     * @tparam TMemoryResource Memory resource type determining device
     * @param a First input tensor
     * @param b Second input tensor
     * @param result Output tensor (must be pre-allocated with matching shape)
     * @param exec_context Optional execution context for stream control (borrowed, not owned)
     *
     * @note For CUDA tensors, use CudaExecutionContext; for CPU, parameter is ignored
     * @note exec_context must outlive this function call
     * @note When exec_context provided, caller controls synchronization
     * @note When null, uses default stream/execution and synchronizes before returning
     *
     * Example:
     * @code
     * // With explicit context (async)
     * auto ctx = std::make_unique<CudaExecutionContext>(0);
     * add(tensor_a, tensor_b, result, ctx.get());
     * ctx->synchronize();
     *
     * // Without context (sync)
     * add(tensor_a, tensor_b, result);  // Returns after completion
     * @endcode
     */
    export template<TensorDataType TDataType, typename TMemoryResource>
        requires isValidTensor<TDataType, TMemoryResource>
    void add(
        const Tensor<TDataType, TMemoryResource>& a,
        const Tensor<TDataType, TMemoryResource>& b,
        Tensor<TDataType, TMemoryResource>& result,
        ExecutionContext<TMemoryResource::device_type>* exec_context = nullptr )
    {
        constexpr DeviceType device = TMemoryResource::device_type;
        TensorOps<device>::add( a, b, result, exec_context );
    }

    /**
     * @brief Element-wise subtraction with optional ExecutionContext (device-dispatched).
     *
     * Computes result[i] = a[i] - b[i] for all elements.
     *
     * @tparam TDataType Abstract tensor data type
     * @tparam TMemoryResource Memory resource type determining device
     * @param a First input tensor (minuend)
     * @param b Second input tensor (subtrahend)
     * @param result Output tensor (must be pre-allocated with matching shape)
     * @param exec_context Optional execution context for stream control (borrowed, not owned)
     */
    export template<TensorDataType TDataType, typename TMemoryResource>
        requires isValidTensor<TDataType, TMemoryResource>
    void subtract(
        const Tensor<TDataType, TMemoryResource>& a,
        const Tensor<TDataType, TMemoryResource>& b,
        Tensor<TDataType, TMemoryResource>& result,
        ExecutionContext<TMemoryResource::device_type>* exec_context = nullptr )
    {
        constexpr DeviceType device = TMemoryResource::device_type;
        TensorOps<device>::subtract( a, b, result, exec_context );
    }

    /**
     * @brief Element-wise multiplication with optional ExecutionContext (device-dispatched).
     *
     * Computes result[i] = a[i] * b[i] for all elements.
     *
     * @tparam TDataType Abstract tensor data type
     * @tparam TMemoryResource Memory resource type determining device
     * @param a First input tensor
     * @param b Second input tensor
     * @param result Output tensor (must be pre-allocated with matching shape)
     * @param exec_context Optional execution context for stream control (borrowed, not owned)
     */
    export template<TensorDataType TDataType, typename TMemoryResource>
        requires isValidTensor<TDataType, TMemoryResource>
    void multiply(
        const Tensor<TDataType, TMemoryResource>& a,
        const Tensor<TDataType, TMemoryResource>& b,
        Tensor<TDataType, TMemoryResource>& result,
        ExecutionContext<TMemoryResource::device_type>* exec_context = nullptr )
    {
        constexpr DeviceType device = TMemoryResource::device_type;
        TensorOps<device>::multiply( a, b, result, exec_context );
    }

    /**
     * @brief Element-wise division with optional ExecutionContext (device-dispatched).
     *
     * Computes result[i] = a[i] / b[i] for all elements.
     *
     * @tparam TDataType Abstract tensor data type
     * @tparam TMemoryResource Memory resource type determining device
     * @param a First input tensor (dividend)
     * @param b Second input tensor (divisor)
     * @param result Output tensor (must be pre-allocated with matching shape)
     * @param exec_context Optional execution context for stream control (borrowed, not owned)
     */
    export template<TensorDataType TDataType, typename TMemoryResource>
        requires isValidTensor<TDataType, TMemoryResource>
    void divide(
        const Tensor<TDataType, TMemoryResource>& a,
        const Tensor<TDataType, TMemoryResource>& b,
        Tensor<TDataType, TMemoryResource>& result,
        ExecutionContext<TMemoryResource::device_type>* exec_context = nullptr )
    {
        constexpr DeviceType device = TMemoryResource::device_type;
        TensorOps<device>::divide( a, b, result, exec_context );
    }

    /**
     * @brief Sum reduction with optional ExecutionContext (device-dispatched).
     *
     * Computes the sum of all elements in the tensor. Always synchronizes before
     * returning the result (even when exec_context is provided).
     *
     * @tparam TDataType Abstract tensor data type
     * @tparam TMemoryResource Memory resource type determining device
     * @param tensor Input tensor
     * @param exec_context Optional execution context for stream control (borrowed, not owned)
     * @return Sum of all elements as float
     *
     * @note Always returns after synchronization to ensure result validity
     */
    export template<TensorDataType TDataType, typename TMemoryResource>
        requires isValidTensor<TDataType, TMemoryResource>
    float sum(
        const Tensor<TDataType, TMemoryResource>& tensor,
        ExecutionContext<TMemoryResource::device_type>* exec_context = nullptr )
    {
        constexpr DeviceType device = TMemoryResource::device_type;
        return TensorOps<device>::sum( tensor, exec_context );
    }

    // --------------------------------------------------------------------
    // Operator Overloads (Syntactic Sugar)
    // --------------------------------------------------------------------
    // Note: Operators cannot accept ExecutionContext, so they always use
    // default execution (synchronous behavior). For async operations with
    // explicit stream control, use the function forms above.
    // --------------------------------------------------------------------

    /**
     * @brief Element-wise addition operator (always synchronous).
     *
     * @note This operator always uses default execution and synchronizes.
     * @note For async operations with stream control, use add(a, b, result, ctx).
     */
    export template<TensorDataType TDataType, typename TMemoryResource>
        requires isValidTensor<TDataType, TMemoryResource>
    Tensor<TDataType, TMemoryResource> operator+(
        const Tensor<TDataType, TMemoryResource>& a,
        const Tensor<TDataType, TMemoryResource>& b )
    {
        Tensor<TDataType, TMemoryResource> result( a.getDevice(), a.shape() );
        add( a, b, result, nullptr );  // Synchronous default execution
        return result;
    }

    /**
     * @brief Element-wise subtraction operator (always synchronous).
     *
     * @note This operator always uses default execution and synchronizes.
     * @note For async operations with stream control, use subtract(a, b, result, ctx).
     */
    export template<TensorDataType TDataType, typename TMemoryResource>
        requires isValidTensor<TDataType, TMemoryResource>
    Tensor<TDataType, TMemoryResource> operator-(
        const Tensor<TDataType, TMemoryResource>& a,
        const Tensor<TDataType, TMemoryResource>& b )
    {
        Tensor<TDataType, TMemoryResource> result( a.getDevice(), a.shape() );
        subtract( a, b, result, nullptr );  // Synchronous default execution
        return result;
    }

    /**
     * @brief Element-wise multiplication operator (always synchronous).
     *
     * @note This operator always uses default execution and synchronizes.
     * @note For async operations with stream control, use multiply(a, b, result, ctx).
     */
    export template<TensorDataType TDataType, typename TMemoryResource>
        requires isValidTensor<TDataType, TMemoryResource>
    Tensor<TDataType, TMemoryResource> operator*(
        const Tensor<TDataType, TMemoryResource>& a,
        const Tensor<TDataType, TMemoryResource>& b )
    {
        Tensor<TDataType, TMemoryResource> result( a.getDevice(), a.shape() );
        multiply( a, b, result, nullptr );  // Synchronous default execution
        return result;
    }

    /**
     * @brief Element-wise division operator (always synchronous).
     *
     * @note This operator always uses default execution and synchronizes.
     * @note For async operations with stream control, use divide(a, b, result, ctx).
     */
    export template<TensorDataType TDataType, typename TMemoryResource>
        requires isValidTensor<TDataType, TMemoryResource>
    Tensor<TDataType, TMemoryResource> operator/(
        const Tensor<TDataType, TMemoryResource>& a,
        const Tensor<TDataType, TMemoryResource>& b )
    {
        Tensor<TDataType, TMemoryResource> result( a.getDevice(), a.shape() );
        divide( a, b, result, nullptr );  // Synchronous default execution
        return result;
    }
}