/**
 * @file DeviceMemoryInfo.ixx
 * @brief What a device has free, and what it has in total.
 */

export module Compute.DeviceMemoryInfo;

import std;

namespace Mila::Dnn::Compute
{
    /**
     * @brief What a device currently has free, and what it has in total.
     *
     * Free memory is read, never modeled: it already accounts for the display compositor,
     * other processes, and anything else resident, none of which a library can predict.
     * See Specifications/MemoryFootprint.md section 2.
     */
    export struct DeviceMemoryInfo
    {
        std::size_t free_bytes{ 0 };
        std::size_t total_bytes{ 0 };
    };
}
