/**
 * @file CpuDevicePlugin.ixx
 * @brief CPU device plugin for device-agnostic registration and discovery.
 *
 * This plugin handles CPU device discovery and registration logic, providing
 * a consistent interface with other device plugins. Unlike GPU devices, CPU
 * is always available and represents a single logical compute unit for the
 * host system.
 */

module;
#include <string>
#include <memory>
#include <thread>
#include <functional>

export module Compute.CpuDevicePlugin;

import Compute.DeviceRegistry;
import Compute.CpuDevice;

namespace Mila::Dnn::Compute
{
    /**
     * @brief CPU device plugin for device-agnostic registration.
     *
     * This plugin encapsulates CPU device registration logic, providing a
     * consistent static interface with other device plugins while handling CPU-specific
     * characteristics such as guaranteed availability and single-device nature.
     */
    export class CpuDevicePlugin {
    public:
        /**
         * @brief Type alias for device registration callback.
         */
        using RegistrationCallback = std::function<void( const std::string&, DeviceRegistry::DeviceFactory )>;

        /**
         * @brief Registers the CPU device using the provided callback.
         *
         * CPU is always available as a fallback compute device. Uses the registration
         * callback to register the device.
         *
         * @param registerCallback Function to call for registering devices
         */
        static void registerDevices( RegistrationCallback registerCallback ) {
            try {
                registerCallback( "CPU", []() {
                    return std::make_shared<CpuDevice>();
                    } );
            }
            catch (...) {
                // CPU registration should never fail
            }
        }

        /**
         * @brief Checks if CPU is available for computation.
         *
         * @return Always returns true
         */
        static bool isAvailable() {
            return true;
        }

        /**
         * @brief Gets the count of CPU devices.
         *
         * @return Always returns 1
         */
        static int getDeviceCount() {
            return 1;
        }

        /**
         * @brief Gets the plugin name identifying the CPU device type.
         *
         * @return String "CPU" identifying this as the CPU device plugin
         */
        static std::string getPluginName() {
            return "CPU";
        }

        /**
         * @brief Gets the number of logical CPU cores available.
         *
         * Returns the number of hardware threads available on the system.
         *
         * @return Number of logical CPU cores/threads available
         */
        static unsigned int getLogicalCoreCount() {
            unsigned int coreCount = std::thread::hardware_concurrency();
            return (coreCount > 0) ? coreCount : 1;
        }

        /**
         * @brief Gets system memory information for CPU operations.
         *
         * @return Available system memory in bytes, or 0 if detection not implemented
         */
        static size_t getAvailableMemory() {
            return 0;
        }

    private:
        /**
         * @brief Checks if the CPU meets minimum requirements for neural network operations.
         *
         * @return true if CPU meets requirements
         */
        static bool isCpuCapable() {
            return true;
        }
    };
}