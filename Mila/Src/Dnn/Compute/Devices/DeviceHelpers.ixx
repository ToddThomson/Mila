module;
#include <vector>
#include <string>

export module Compute.DeviceHelpers;

import Compute.DeviceContext;
import Compute.DeviceRegistry;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Lists all available compute devices.
     *
     * This function returns a list of all available compute devices
     * that can be used with DeviceContext.
     *
     * @return std::vector<std::string> A list of device identifiers (e.g., "CPU", "CUDA:0", "CUDA:1").
     */
    export std::vector<std::string> list_devices() {
        auto& registry = DeviceRegistry::instance();

        return registry.listDevices();
    }

    /**
    * @brief Checks if a specific device is available.
    *
    * @param device_name The name of the device to check (e.g., "CPU", "CUDA:0").
    * @return bool True if the device is available, false otherwise.
    */
    export bool is_device_available( const std::string& device_name ) {
        try {
            // Try to create a DeviceContext with the specified device
            DeviceContext context( device_name );
            return true;
        }
        catch ( const std::exception& ) {
            // If an exception is thrown, the device is not available
            return false;
        }
    }

}
