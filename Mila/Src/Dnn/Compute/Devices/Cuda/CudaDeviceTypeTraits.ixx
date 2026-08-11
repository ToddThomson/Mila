/**
 * @file CudaDeviceTypeTraits.ixx
 * @brief DeviceTypeTraits specialization for CUDA devices.
 *
 * Provides the canonical mapping from ::Mila::Dnn::Compute::DeviceType::Cuda
 * to the preferred memory resource type used for CUDA-backed tensors.
 */

module;
#include <type_traits>

export module Compute.DeviceTypeTraits.Cuda;

import Compute.DeviceTypeTraits;
import Compute.DeviceType;
import Compute.CudaDeviceMemoryResource;
import Compute.CudaPinnedMemoryResource;

namespace Mila::Dnn::Compute
{
    /**
     * @brief DeviceTypeTraits specialization for the CUDA device.
     *
     * Provides the canonical memory resource type for DeviceType::Cuda.
     *
     * Example:
     * @code
     * using MR = DeviceTypeTraits<DeviceType::Cuda>::memory_resource;
     * @endcode
     */
    template <>
    struct DeviceTypeTraits<DeviceType::Cuda> {
        using memory_resource = CudaDeviceMemoryResource;

        // Pinned, not pageable: it is both host- and device-accessible, so a transfer
        // through it is a direct DMA with no driver staging copy, and it stays readable
        // host-side for the serialization paths that need to look at the bytes.
        using host_staging_memory_resource = CudaPinnedMemoryResource;
    };
}