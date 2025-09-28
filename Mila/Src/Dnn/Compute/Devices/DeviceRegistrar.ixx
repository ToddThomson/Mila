/**
 * @file DeviceRegistrar.ixx
 * @brief Device-agnostic registrar for managing compute device discovery and registration.
 */

module;
#include <string>
#include <memory>
#include <vector>

export module Compute.DeviceRegistrar;

import Compute.DeviceRegistry;
import Compute.CpuDevicePlugin;
import Compute.CudaDevicePlugin;
// FUTURE: import Compute.MetalDevicePlugin;
// FUTURE: import Compute.OpenCLDevicePlugin;
// FUTURE: import Compute.VulkanDevicePlugin;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Device-agnostic registrar for managing compute device discovery and registration.
     *
     * This class provides a centralized, device-agnostic mechanism for coordinating the
     * registration of all available compute devices within the Mila framework. It delegates
     * device-specific discovery and registration to specialized device plugins, maintaining
     * complete separation from device-specific APIs and code.
     *
     * Key architectural principles:
     * - Device-agnostic: Contains no device-specific code or API dependencies
     * - Single responsibility: Focuses solely on device plugin coordination
     * - Extensible: New device types can be added by implementing device plugins
     * - Plugin-based: Device discovery and registration handled by specialized plugins
     * - Singleton pattern with lazy initialization ensures devices are registered once
     *
     * @note Device context creation is handled by DeviceContext::create() factory method
     * @note This class does not manage device contexts - only device registration
     */
    export class DeviceRegistrar {
    public:
        /**
         * @brief Gets the singleton instance of DeviceRegistrar.
         *
         * Performs lazy initialization of all available compute devices through
         * device plugins on first access. Thread-safe singleton implementation.
         *
         * @return DeviceRegistrar& Reference to the singleton instance.
         */
        static DeviceRegistrar& instance() {
            static DeviceRegistrar instance;

            // Lazy initialization of devices through plugins
            if (!is_initialized_) {
                registerAllDevices();
                is_initialized_ = true;
            }

            return instance;
        }

        /**
         * @brief Lists all currently available devices.
         *
         * Delegates to the DeviceRegistry to provide a list of all devices
         * that have been successfully registered by device plugins.
         *
         * @return std::vector<std::string> Vector of available device identifiers
         */
        std::vector<std::string> listAvailableDevices() const {
            return DeviceRegistry::instance().listDevices();
        }

        /**
         * @brief Checks if a specific device is available.
         *
         * Queries the DeviceRegistry to determine if the specified device
         * has been registered and is available for use.
         *
         * @param device_name Device identifier to check
         * @return true if device is available, false otherwise
         */
        bool isDeviceAvailable(const std::string& device_name) const {
            return DeviceRegistry::instance().hasDevice(device_name);
        }

        // Delete copy constructor and copy assignment operator
        DeviceRegistrar(const DeviceRegistrar&) = delete;
        DeviceRegistrar& operator=(const DeviceRegistrar&) = delete;

    private:
        DeviceRegistrar() = default;

        /**
         * @brief Registers all available compute devices through device plugins.
         *
         * Delegates to device-specific plugins that handle device discovery,
         * availability checking, and registration. Each plugin is responsible
         * for its own device detection logic and API interactions.
         *
         * This method coordinates the plugin initialization sequence, ensuring
         * all supported device types are given the opportunity to register
         * their available devices with the DeviceRegistry.
         */
        static void registerAllDevices() {
            // Register devices through device plugins
            CpuDevicePlugin::registerDevices();
            CudaDevicePlugin::registerDevices();

            // FUTURE: Enable when device plugins are implemented
            // MetalDevicePlugin::registerDevices();
            // OpenCLDevicePlugin::registerDevices();
            // VulkanDevicePlugin::registerDevices();
        }

        /**
         * @brief Flag indicating whether devices have been initialized.
         */
        static inline bool is_initialized_ = false;
    };
}