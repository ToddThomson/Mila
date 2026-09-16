/**
 * @file DeviceAllocation.ixx
 * @brief What one device allocation occupies once the driver has rounded it, and the device's free memory.
 *
 * The two readings a footprint prediction takes from the device. See Specifications/MemoryFootprint.md 11.3 and 11.8.
 */

module;
#include <cstddef>
#include <exception>
#include <memory>

export module Compute.DeviceAllocation;

import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceRegistry;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Requests up to this size share blocks the driver packs; larger ones are rounded to the granularity.
     */
    export inline constexpr std::size_t kPackedAllocationLimitBytes = std::size_t{ 1024 } * 1024;

    /**
     * @brief Bytes one allocation of `bytes` occupies on a device with the given granularity.
     *
     * An allocation of at most kPackedAllocationLimitBytes counts as its size: its share of a packed block is not
     * predicted. A granularity of zero rounds nothing. Rounding the sum of two allocations is wrong; every
     * prediction and every report of a device allocation goes through this one allocation at a time, so the two
     * round identically.
     */
    export constexpr std::size_t occupiedDeviceBytes( std::size_t bytes, std::size_t granularity ) noexcept
    {
        if ( granularity == 0 || bytes <= kPackedAllocationLimitBytes )
        {
            return bytes;
        }

        return ( bytes + granularity - 1 ) / granularity * granularity;
    }

    /**
     * @brief The allocation granularity of `device`, or zero when it has none or cannot be asked.
     */
    export std::size_t allocationGranularity( DeviceId device ) noexcept
    {
        try
        {
            const std::shared_ptr<Device> instance = DeviceRegistry::instance().getDevice( device );

            return instance ? instance->getAllocationGranularity() : 0;
        } catch ( const std::exception& )
        {
            return 0;
        }
    }

    /**
     * @brief Bytes one tensor's allocation occupies on its own device.
     */
    export std::size_t occupiedTensorBytes( const auto& tensor )
    {
        return occupiedDeviceBytes( tensor.getStorageSize(), allocationGranularity( tensor.getDeviceId() ) );
    }

    /**
     * @brief The device's free memory, or zero when the device cannot say.
     *
     * Zero is the answer for a device with no distinct memory pool (the CPU) and for a failed query; a caller
     * treats it as unknown rather than as a full device.
     */
    export std::size_t readFreeDeviceBytes( DeviceId device ) noexcept
    {
        try
        {
            const std::shared_ptr<Device> instance = DeviceRegistry::instance().getDevice( device );

            if ( !instance )
            {
                return 0;
            }

            const DeviceMemoryInfo memory = instance->getMemoryInfo();

            return memory.total_bytes == 0 ? 0 : memory.free_bytes;
        } catch ( const std::exception& )
        {
            return 0;
        }
    }
}
