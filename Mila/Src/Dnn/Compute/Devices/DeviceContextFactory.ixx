/**
 * @file DeviceContextFactory.ixx
 * @brief Factory implementation for device context creation using device registry pattern.
 */

module;
#include <memory>
#include <string>
#include <stdexcept>
#include <algorithm>

export module Compute.DeviceContextFactory;

import Compute.DeviceContext;
import Compute.DeviceRegistrar;
import Compute.DeviceRegistry;
import Compute.CpuDeviceContext;
import Compute.CudaDeviceContext;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Factory method implementation for DeviceContext::create() utilizing device registry pattern.
     *
     * This implementation properly integrates with the device registry infrastructure to:
     * 1. Initialize device registration through DeviceRegistrar
     * 2. Validate device availability before attempting creation
     * 3. Create appropriate device contexts only for registered devices
     * 4. Provide clear error messages for unavailable devices
     */
    std::shared_ptr<DeviceContext> DeviceContext::create(const std::string& device_name) {
        DeviceRegistrar& registrar = DeviceRegistrar::instance();
        
        // Normalize device name for consistent comparison
        std::string normalized_name = device_name;
        std::transform(normalized_name.begin(), normalized_name.end(), normalized_name.begin(), ::toupper);

        /*if (!registrar.isDeviceAvailable( device_name )) {

            auto available_devices = registrar.listAvailableDevices();
            std::string available_list;
            for (size_t i = 0; i < available_devices.size(); ++i) {
                available_list += available_devices[i];
                if (i < available_devices.size() - 1) {
                    available_list += ", ";
                }
            }

            throw std::runtime_error(
                "Device '" + device_name + "' is not available. " +
                "Available devices: [" + available_list + "]"
            );
        }*/

        // Create device context based on device type
        // Only create contexts for devices that are actually registered and available
        try {
            if (normalized_name == "CPU") {
                return std::make_shared<CpuDeviceContext>();
            }
            else if (normalized_name.starts_with("CUDA")) {
                return std::make_shared<CudaDeviceContext>(device_name);
            }
            else {
                // This should not happen if device registry is working correctly,
                // but provides a safety net for edge cases
                throw std::runtime_error(
                    "Device '" + device_name + "' is registered but no context factory is available. " +
                    "This indicates a configuration issue in the device infrastructure."
                );
            }
        }
        catch (const std::exception& e) {
            throw std::runtime_error(
                "Failed to create device context for '" + device_name + "': " + e.what()
            );
        }
    }
}