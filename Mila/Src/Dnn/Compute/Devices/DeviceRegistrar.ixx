/**
 * @file DeviceRegistrar.ixx
 * @brief Implementation of a class to manage device registration.
 */

module;
#include <string>
#include <memory>

export module Compute.DeviceRegistrar;

import Compute.DeviceRegistry;
import Compute.ComputeDevice;
import Compute.CpuDevice;
import Compute.CudaDevice;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Class to manage compute device initialization.
     *
     * This class provides a centralized mechanism for registering all available
     * compute devices within the Mila framework. It follows a singleton pattern
     * with lazy initialization to ensure devices are registered only once when needed.
     */
    export class DeviceRegistrar {
    public:
        /**
         * @brief Get the singleton instance of DeviceRegistrar.
         *
         * @return DeviceRegistrar& Reference to the singleton instance.
         */
        static DeviceRegistrar& instance() {
            static DeviceRegistrar instance;

            // Lazy initialization of devices
            if ( !is_initialized_ ) {
                registerDevices();
                is_initialized_ = true;
            }

            return instance;
        }

        // Delete copy constructor and copy assignment operator
        DeviceRegistrar( const DeviceRegistrar& ) = delete;
        DeviceRegistrar& operator=( const DeviceRegistrar& ) = delete;

    private:
        DeviceRegistrar() = default;

        /**
         * @brief Register all available compute devices.
         */
        static void registerDevices() {
            CpuDevice::registerDevice();
            CudaDevice::registerDevices();
        }

        static inline bool is_initialized_ = false;
    };
}