module;
#include <concepts>
#include <memory>
#include <span>
#include <type_traits>
#include <stdexcept>

export module Dnn.TensorOps:Transfer;

import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeMap;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorOps.Base;
import Compute.DeviceTraits;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.DeviceType;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Copies tensor data from source to destination tensor with optional ExecutionContext
     *
     * Transfers data from source tensor to pre-allocated destination tensor.
     * Both tensors must have compatible shapes (same dimensions).
     * Supports type conversion and cross-device transfers with explicit stream control.
     *
     * Device compatibility rules:
     * - Host-accessible to host-accessible: Always allowed
     * - Host-accessible to device-only: Uses destination device
     * - Device-only to host-accessible: Uses source device
     * - Device-only to device-only: Must be same device type (e.g., both CUDA)
     *
     * ExecutionContext handling:
     *   - Optional ExecutionContext parameter for stream control (borrowed, not owned)
     *   - When provided, operations use the context's stream (caller controls sync)
     *   - When null, operations use default stream and synchronize before returning
     *   - Raw pointer semantics ensure zero overhead
     *
     * @tparam TSrcDataType Source tensor data type
     * @tparam TSrcMemoryResource Source memory resource type
     * @tparam TDstDataType Destination tensor data type
     * @tparam TDstMemoryResource Destination memory resource type
     * @param src Source tensor to copy from
     * @param dst Destination tensor to copy to (must be pre-allocated)
     * @param exec_context Optional execution context for stream control (borrowed, not owned)
     *
     * @throws std::runtime_error If device-only tensors are on incompatible device types
     *
     * @note exec_context must outlive this function call
     * @note When exec_context provided, caller controls synchronization
     * @note When exec_context is null, uses default stream and synchronizes before returning
     * @note For CPU-only operations, exec_context parameter is ignored
     *
     * Example:
     * @code
     * // With explicit context (async)
     * auto ctx = std::make_unique<CudaExecutionContext>(0);
     * copy(src_tensor, dst_tensor, ctx.get());
     * ctx->synchronize();
     *
     * // Without context (sync)
     * copy(src_tensor, dst_tensor);  // Returns after completion
     * @endcode
     */
    export template<TensorDataType TSrcDataType, typename TSrcMemoryResource,
        TensorDataType TDstDataType, typename TDstMemoryResource>
        requires isValidTensor<TSrcDataType, TSrcMemoryResource> && isValidTensor<TDstDataType, TDstMemoryResource>
    void copy(
        const Tensor<TSrcDataType, TSrcMemoryResource>& src,
        Tensor<TDstDataType, TDstMemoryResource>& dst,
        IExecutionContext* exec_context = nullptr ) {

        if (src.shape() != dst.shape()) {
            throw std::invalid_argument("Source and destination tensors must have the same shape for copy operation.");
		}

        if constexpr (!TSrcMemoryResource::is_host_accessible && !TDstMemoryResource::is_host_accessible)
        {
            // Both tensors are device-only, must be same device type
            static_assert(TSrcMemoryResource::device_type == TDstMemoryResource::device_type,
                "Cannot copy between different device types (e.g., CUDA to Metal). "
                "Use host-accessible memory as intermediate step.");

            throw std::runtime_error( "Device to Device Copy() is not implemented." );
            
            // TODO:
            //constexpr DeviceType device = TSrcMemoryResource::device_type;
            //TensorOps<device>::copy( src, dst, exec_context );
        }

        // Determine which DeviceType to use based on memory accessibility
        if constexpr (!TSrcMemoryResource::is_host_accessible || !TDstMemoryResource::is_host_accessible)
        {
            // At least one tensor is device-only, must use device operations
            if constexpr (!TSrcMemoryResource::is_host_accessible)
            {
                // Source is device-only, use source device
                constexpr DeviceType device = TSrcMemoryResource::device_type;
                TensorOps<device>::copy( src, dst, exec_context );
            }
            else
            {
                // Destination is device-only, use destination device
                constexpr DeviceType device = TDstMemoryResource::device_type;
                TensorOps<device>::copy( src, dst, exec_context );
            }
        }
        else
        {
            // Both are host-accessible, use destination device (matches exec_context type)
            constexpr DeviceType device = TDstMemoryResource::device_type;
            TensorOps<device>::copy( src, dst, exec_context );
        }
    }
}