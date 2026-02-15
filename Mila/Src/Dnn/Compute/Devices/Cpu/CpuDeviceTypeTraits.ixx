/**
 * @file CpuDeviceTypeTraits.ixx
 * @brief DeviceTypeTraits specialization for the CPU device.
 *
 * Provides the canonical mapping from ::Mila::Dnn::Compute::DeviceType::Cpu
 * to the preferred memory resource type used for CPU-backed tensors.
 */

module;
#include <type_traits>

export module Compute.DeviceTypeTraits.Cpu;

import Compute.DeviceTypeTraits;
import Compute.DeviceType;
import Compute.CpuMemoryResource;

namespace Mila::Dnn::Compute
{
    /**
     * @brief DeviceTypeTraits specialization for the CPU device.
     *
     * Provides the canonical memory resource type for DeviceType::Cpu.
     *
     * Example:
     * @code
     * using MR = DeviceTypeTraits<DeviceType::Cpu>::memory_resource;
     * @endcode
     */
    export template <>
    struct DeviceTypeTraits<DeviceType::Cpu> {
        using memory_resource = CpuMemoryResource;
    };
}