/**
 * @file DevicePerformance.ixx
 * @brief What a device reports about its speed, as it reports it.
 */

export module Compute.DevicePerformance;

import std;

namespace Mila::Dnn::Compute
{
    /**
     * @brief A device's speed facts, raw: nothing here is a score, and nothing ranks devices by them yet.
     *
     * The deployment planner carries them in each reading so a plan records them; ordering devices by speed waits on a
     * rule that has been measured (Specifications/Deployment.md 12.9). Zero is "not reported", never a measured zero.
     */
    export struct DevicePerformance
    {
        /// CUDA's streaming multiprocessors. Metal reports no core count through its API.
        int compute_unit_count{ 0 };

        /// The compute clock, in kHz.
        int clock_khz{ 0 };

        /// The memory clock, in kHz.
        int memory_clock_khz{ 0 };

        /// The width of the memory bus, in bits.
        int memory_bus_width_bits{ 0 };

        /// True when the device reported none of these.
        [[nodiscard]] bool isEmpty() const noexcept
        {
            return compute_unit_count == 0 && clock_khz == 0 && memory_clock_khz == 0 && memory_bus_width_bits == 0;
        }
    };
}
