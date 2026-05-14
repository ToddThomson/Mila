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
    };
}