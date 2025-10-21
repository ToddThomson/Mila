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
import Compute.ComputeDevice;
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
         * @brief Returns an index-aware device factory for CPU.
         *
         * The DeviceRegistry expects factories of the form
         * std::function<std::shared_ptr<ComputeDevice>(int)>. CPU ignores the
         * index parameter and always returns the same type of device for any index.
         *
         * @return Factory callable taking an int index and returning a ComputeDevice instance.
         */
        static std::function<std::shared_ptr<ComputeDevice>(int)> getDeviceFactory() {
            return [] ( int /*index*/ ) {
                return std::make_shared<CpuDevice>();
            };
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