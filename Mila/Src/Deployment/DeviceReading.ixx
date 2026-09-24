/**
 * @file DeviceReading.ixx
 * @brief One reading of a device, taken once by the deployment planner.
 *
 * Every decision a plan makes about a device is made against its reading, and nothing below the
 * planner reads the device again. See Specifications/Deployment.md sections 3.3 and 9.
 */

module;
#include <cstddef>
#include <exception>
#include <memory>

export module Deployment.DeviceReading;

import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceRegistry;

namespace Mila::Deployment
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    /**
     * @brief A device's identity, its free and total memory, and its allocation granularity, at one moment.
     *
     * A free reading of zero means the device could not say (the CPU, or a failed query) and is not a full
     * device.
     */
    export struct DeviceReading
    {
        DeviceId device{};

        std::size_t free_bytes{ 0 };

        std::size_t total_bytes{ 0 };

        /// What a prediction rounds each allocation over 1 MiB up to (MemoryFootprint.md 11.8).
        std::size_t allocation_granularity{ 0 };

        /// True when the device reported its memory, so a plan can be priced against it.
        [[nodiscard]] bool reportsMemory() const noexcept
        {
            return total_bytes != 0;
        }

        /**
         * @brief Read the device now.
         *
         * Take it after the graph is constructed: construction creates the execution context, which holds
         * device memory the build will not find free (Deployment.md section 9).
         */
        static DeviceReading take( DeviceId device ) noexcept
        {
            DeviceReading reading;
            reading.device = device;

            try
            {
                const std::shared_ptr<Device> instance = DeviceRegistry::instance().getDevice( device );

                if ( instance )
                {
                    const DeviceMemoryInfo memory = instance->getMemoryInfo();

                    reading.total_bytes = memory.total_bytes;
                    reading.free_bytes = memory.total_bytes == 0 ? 0 : memory.free_bytes;
                    reading.allocation_granularity = instance->getAllocationGranularity();
                }
            } catch ( const std::exception& )
            {
                reading.free_bytes = 0;
                reading.total_bytes = 0;
            }

            return reading;
        }
    };
}
