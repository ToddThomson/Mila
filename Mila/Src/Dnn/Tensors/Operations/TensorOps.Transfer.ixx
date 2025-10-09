module;
#include <concepts>
#include <memory>
#include <span>

export module Dnn.TensorOps:Transfer;

export import :Transfer.Cpu;
export import :Transfer.Cuda;

import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeMap;
import Dnn.TensorDataTypeTraits;
import Compute.DeviceTraits;
import Compute.CpuMemoryResource;
import Compute.CudaDeviceMemoryResource;

namespace Mila::Dnn
{
    /**
     * @brief Copies tensor data from source to destination tensor
     *
     * Transfers data from source tensor to pre-allocated destination tensor.
     * Both tensors must have compatible shapes (same dimensions).
     * Supports type conversion and cross-device transfers.
     *
     * Device compatibility rules:
     * - Host-accessible to host-accessible: Always allowed
     * - Host-accessible to device-only: Uses destination device
     * - Device-only to host-accessible: Uses source device
     * - Device-only to device-only: Must be same device type (e.g., both CUDA)
     *
     * @tparam TSrcDataType Source tensor data type
     * @tparam TSrcMemoryResource Source memory resource type
     * @tparam TDstDataType Destination tensor data type
     * @tparam TDstMemoryResource Destination memory resource type
     * @param src Source tensor to copy from
     * @param dst Destination tensor to copy to (must be pre-allocated)
     *
     * @throws std::runtime_error If device-only tensors are on incompatible device types
     */
    export template<TensorDataType TSrcDataType, typename TSrcMemoryResource,
                    TensorDataType TDstDataType, typename TDstMemoryResource>
        requires isValidTensor<TSrcDataType, TSrcMemoryResource> && isValidTensor<TDstDataType, TDstMemoryResource>
    void copy( const Tensor<TSrcDataType, TSrcMemoryResource>& src, Tensor<TDstDataType, TDstMemoryResource>& dst ) {

        if constexpr (!TSrcMemoryResource::is_host_accessible && !TDstMemoryResource::is_host_accessible)
        {
            // Both are device-only, must be same device type
            static_assert(TSrcMemoryResource::device_type == TDstMemoryResource::device_type,
                "Cannot copy between different device types (e.g., CUDA to Metal). "
                "Use host-accessible memory as intermediate step.");
        }

        // Determine which device context to use based on memory accessibility
        if constexpr (!TSrcMemoryResource::is_host_accessible || !TDstMemoryResource::is_host_accessible) {
            // At least one tensor is device-only, must use device operations
            if constexpr (!TSrcMemoryResource::is_host_accessible) {
                // Source is device-only, use source device
                constexpr DeviceType device = TSrcMemoryResource::device_type;
                TensorOps<device>::template copy<TSrcDataType, TSrcMemoryResource, TDstDataType, TDstMemoryResource>( src, dst );
            }
            else {
                // Destination is device-only, use destination device
                constexpr DeviceType device = TDstMemoryResource::device_type;
                TensorOps<device>::template copy<TSrcDataType, TSrcMemoryResource, TDstDataType, TDstMemoryResource>( src, dst );
            }
        }
        else {
			// Both are host-accessible, use Source device (arbitrary choice)
            constexpr DeviceType device = TSrcMemoryResource::device_type;
            //TensorOps<device>::template copy<TSrcDataType, TSrcMemoryResource, TDstDataType, TDstMemoryResource>( src, dst );
        }
    }
}