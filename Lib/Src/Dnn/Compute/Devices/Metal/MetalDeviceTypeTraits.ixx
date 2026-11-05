/**
 * @file MetalDeviceTypeTraits.ixx
 * @brief DeviceTypeTraits specialization for the Metal device.
 */

export module Compute.DeviceTypeTraits.Metal;

import std;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.MetalMemoryResource;

namespace Mila::Dnn::Compute
{
    template <>
    struct DeviceTypeTraits<DeviceType::Metal>
    {
        using memory_resource = MetalMemoryResource;

        // Unified memory has no staging step to speed up, and FP16 and BF16 tensors need device-accessible memory, which
        // this is. The host reaches it after a wait, as `copy` already does.
        using host_staging_memory_resource = MetalMemoryResource;
    };
}
