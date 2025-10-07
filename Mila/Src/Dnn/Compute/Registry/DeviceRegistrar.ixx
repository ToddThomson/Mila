/**
 * @file DeviceRegistrar.ixx
 * @brief Device-agnostic registrar for automatic device discovery and registration.
 * Architecture:
 * - DeviceRegistrar: Owns registration process, calls plugins with callback
 * - Plugins: Accept registration callback, register devices through it
 * - DeviceRegistry: Stores registered devices, provides query API
 */

module;
#include <string>
#include <memory>
#include <functional>

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
     * @brief Device-agnostic registrar for automatic device discovery and registration.
     *
     * Coordinates automatic registration of compute devices through plugins.
     * Plugins receive a registration callback function to register their devices.
     */
    export class DeviceRegistrar {
    public:
        /**
         * @brief Type alias for device registration callback.
         *
         * Function signature that plugins use to register devices.
         */
        using RegistrationCallback = std::function<void(const std::string&, DeviceRegistry::DeviceFactory)>;

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
            return instance;
        }

        DeviceRegistrar(const DeviceRegistrar&) = delete;
        DeviceRegistrar& operator=(const DeviceRegistrar&) = delete;

    private:
        /**
         * @brief Private constructor performs automatic device registration.
         *
         * Triggers device plugin initialization by passing them a registration callback.
         */
        DeviceRegistrar() {
            registerAllDevices();
        }

        /**
         * @brief Registers all available compute devices through device plugins.
         *
         * Creates a registration callback and passes it to each plugin.
         * Plugins use the callback to register their devices.
         */
        static void registerAllDevices() {
            RegistrationCallback registerCallback = 
                [](const std::string& name, DeviceRegistry::DeviceFactory factory) {
                    DeviceRegistry::instance().registerDevice(name, std::move(factory));
                };

            CpuDevicePlugin::registerDevices(registerCallback);
            CudaDevicePlugin::registerDevices(registerCallback);

            // FUTURE: Enable when device plugins are implemented
            // MetalDevicePlugin::registerDevices(registerCallback);
            // OpenCLDevicePlugin::registerDevices(registerCallback);
            // VulkanDevicePlugin::registerDevices(registerCallback);
        }
    };
}