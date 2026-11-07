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

        // Metal runs inference only, so its operations declare no backward pass (MetalBackend.md section 1).
        static constexpr bool supports_training = false;

        // Unified memory: the GPU and the host share one pool, so keeping a table "in host memory" saves nothing.
        static constexpr bool memory_separate_from_host = false;
    };
}
