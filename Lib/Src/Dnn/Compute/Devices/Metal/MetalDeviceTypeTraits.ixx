/**
 * @file MetalDeviceTypeTraits.ixx
 * @brief DeviceTypeTraits specialization for the Metal device.
 */

export module Compute.DeviceTypeTraits.Metal;

import std;
import Compute.CpuMemoryResource;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.MetalMemoryResource;

namespace Mila::Dnn::Compute
{
    template <>
    struct DeviceTypeTraits<DeviceType::Metal>
    {
        using memory_resource = MetalMemoryResource;

        // Unified memory has no staging step to speed up, so a host tensor is what a transfer goes through.
        using host_staging_memory_resource = CpuMemoryResource;
    };
}
