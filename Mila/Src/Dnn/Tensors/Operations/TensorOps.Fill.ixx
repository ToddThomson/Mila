/**
 * @file TensorOps.Fill.ixx
 * @brief High-level initializer helpers (device-dispatching) for tensors
 *
 * This partition provides the generic, device-agnostic entry points for tensor
 * initialization operations (copying host-provided values into tensors and
 * filling tensors with scalar values). The implementation forwards to the
 * device-specific `TensorOps<...>` partitions (for example
 * `TensorOps<Compute::CpuComputeDeviceTag>::fill(...)`).
 *
 * The host representation for a logical tensor element is defined by
 * `TensorDataTypeTraits<TDataType>::host_type` and is exposed here via the
 * alias `host_value_t<TDataType>` so callers and implementations use a single,
 * authoritative host-side type for conversions.
 *
 * ExecutionContext handling:
 *   - Optional ExecutionContext parameter for stream control (borrowed, not owned)
 *   - When provided, operations use the context's stream (caller controls sync)
 *   - When null, operations use default stream and synchronize before returning
 *   - Raw pointer semantics ensure zero overhead
 *
 * Usage:
 * - Call `fill(tensor, values)` or `fill(tensor, scalar)` from user code.
 * - Optionally provide ExecutionContext for explicit stream control:
 *   `fill(tensor, values, ctx.get())`
 * - The host value type is selected automatically from the tensor data type
 *   (float for floating-point tensors, int32_t for integer tensors).
 *
 * Preconditions:
 * - The `isValidTensor<TDataType,TMemoryResource>` concept must hold for the
 *   tensor types used here (ensures memory resource compatibility and trait
 *   availability).
 * - ExecutionContext (if provided) must outlive the function call.
 */

module;
#include <concepts>
#include <span>
#include <type_traits>
#include <cstdint>

export module Dnn.TensorOps:Fill;

import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeMap;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorOps.Base;
import Compute.DeviceTraits;
import Compute.ExecutionContext;
import Compute.DeviceType;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Copy host values into a tensor with device dispatch and optional ExecutionContext.
     *
     * Forwards the host->tensor copy operation (span form) to the device-specific
     * implementation `TensorOps<Tag>::fill`. Borrows execution context for stream
     * control with zero overhead. Falls back to default stream when no context provided.
     *
     * The host element type is selected by `host_value_t<TDataType>` so callers
     * must provide values in the expected host representation (float for
     * floating-point tensor types, int32_t for integer tensor types). The
     * device implementation performs any necessary conversion/quantization.
     *
     * @tparam TDataType Abstract tensor data type.
     * @tparam TMemoryResource Memory resource type backing the tensor.
     * @param tensor Destination tensor to be filled. Must satisfy `isValidTensor`.
     * @param host_values Span of host values in host representation (see host_value_t).
     * @param exec_context Optional execution context for stream control (borrowed, not owned)
     *
     * @note exec_context must outlive this function call
     * @note When exec_context provided, caller controls synchronization
     * @note When exec_context is null, uses default stream and synchronizes before returning
     * @note For CUDA tensors, use CudaExecutionContext; for CPU, parameter is ignored
     *
     * Example:
     * @code
     * // With explicit context (async)
     * auto ctx = std::make_unique<CudaExecutionContext>(0);
     * std::vector<float> values = {1.0f, 2.0f, 3.0f};
     * fill(tensor, std::span{values}, ctx.get());
     * ctx->synchronize();
     *
     * // Without context (sync)
     * fill(tensor, std::span{values});  // Returns after completion
     * @endcode
     */
    export template<TensorDataType TDataType, typename TMemoryResource>
        requires isValidTensor<TDataType, TMemoryResource>
    void fill(
        Tensor<TDataType, TMemoryResource>& tensor,
        std::span<const host_value_t<TDataType>> host_values,
        ExecutionContext<TMemoryResource::device_type>* exec_context = nullptr )
    {
        constexpr DeviceType device = TMemoryResource::device_type;
        TensorOps<device>::fill( tensor, host_values, exec_context );
    }

    /**
     * @brief Fill a tensor with a scalar host value (device-dispatched) with optional ExecutionContext.
     *
     * Forwards scalar fills to the device-specific `TensorOps<Tag>::fill`.
     * Borrows execution context for stream control with zero overhead.
     * The function signature enforces the expected host scalar representation
     * for each abstract tensor data type via `host_value_t<TDataType>`.
     *
     * @tparam TDataType Abstract tensor data type.
     * @tparam TMemoryResource Memory resource type backing the tensor.
     * @param tensor Destination tensor to be filled. Must satisfy `isValidTensor`.
     * @param host_value Scalar value in host representation to broadcast to the tensor.
     * @param exec_context Optional execution context for stream control (borrowed, not owned)
     *
     * @note exec_context must outlive this function call
     * @note When exec_context provided, caller controls synchronization
     * @note When exec_context is null, uses default stream and synchronizes before returning
     * @note For CUDA tensors, use CudaExecutionContext; for CPU, parameter is ignored
     *
     * Example:
     * @code
     * // With explicit context (async)
     * auto ctx = std::make_unique<CudaExecutionContext>(0);
     * fill(float_tensor, 3.14f, ctx.get());
     * fill(int_tensor, 42, ctx.get());
     * ctx->synchronize();
     *
     * // Without context (sync)
     * fill(float_tensor, 3.14f);  // Returns after completion
     * fill(int_tensor, 42);       // Returns after completion
     * @endcode
     */
    export template<TensorDataType TDataType, typename TMemoryResource>
        requires isValidTensor<TDataType, TMemoryResource>
    void fill(
        Tensor<TDataType, TMemoryResource>& tensor,
        host_value_t<TDataType> host_value,
        ExecutionContext<TMemoryResource::device_type>* exec_context = nullptr )
    {
        constexpr DeviceType device = TMemoryResource::device_type;
        TensorOps<device>::fill( tensor, host_value, exec_context );
    }
}