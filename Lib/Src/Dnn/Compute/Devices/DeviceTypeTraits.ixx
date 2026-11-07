/**
 * @file DeviceTypeTraits.ixx
 * @brief What a device type uses for memory, staging and training, specialized per backend.
 */

export module Compute.DeviceTypeTraits;

import std;
import Compute.DeviceType;
import Compute.CpuMemoryResource;

namespace Mila::Dnn::Compute
{
    // Primary unspecialized template -- will fail to instantiate for unsupported devices.
    export template <DeviceType TDevice>
    struct DeviceTypeTraits;

    /**
     * @brief Staging memory the host reads through a tensor: the device's staging memory where the host can, the host's
     *        own where it cannot.
     *
     * Two kinds of staging differ here. Bytes on their way to the device take host_staging_memory_resource whatever it
     * is. Bytes the host reads back -- a value copied off the device -- need is_host_accessible, which Metal's shared
     * memory deliberately is not (Specifications/Metal/MetalBackend.md section 5), so on Metal they stage in CPU memory.
     */
    export template <DeviceType TDevice>
    using HostReadableStagingMemoryResource = std::conditional_t<
        DeviceTypeTraits<TDevice>::host_staging_memory_resource::is_host_accessible,
        typename DeviceTypeTraits<TDevice>::host_staging_memory_resource,
        CpuMemoryResource>;
}
