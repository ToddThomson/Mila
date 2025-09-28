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

export module Compute.CpuDevicePlugin;

// FUTURE: import Compute.DevicePlugin;
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
     *
     * Key responsibilities:
     * - CPU device registration with the DeviceRegistry
     * - System capability detection (thread count, SIMD support)
     * - Static plugin interface for the DeviceRegistrar
     * - Error handling for edge cases (though CPU is always available)
     */
    export class CpuDevicePlugin {
        // FUTURE: public DevicePlugin {
    public:
        /**
         * @brief Registers the CPU device with the DeviceRegistry.
         *
         * CPU is always available as a fallback compute device, so this method
         * unconditionally registers a single CPU device. The registration includes
         * a factory function that creates CpuDevice instances.
         *
         * Device naming convention: "CPU" (no index since there's only one logical CPU device)
         *
         * @note This method is safe to call multiple times (registry handles duplicates)
         * @note CPU registration never fails - it's the ultimate fallback device
         * @note Unlike GPU plugins, CPU doesn't require runtime availability checking
         */
        static void registerDevices() {
            try {
                auto& registry = DeviceRegistry::instance();

                // Register the single CPU device with factory function
                registry.registerDevice("CPU", []() {
                    return std::make_shared<CpuDevice>();
                    });
            }
            catch (...) {
                // CPU registration should never fail, but suppress any unexpected exceptions
                // to maintain consistency with other device plugins
            }
        }

        /**
         * @brief Checks if CPU is available for computation.
         *
         * CPU is always available as the host processor, so this method always
         * returns true. Provided for consistency with other device plugins that
         * may have availability constraints.
         *
         * @return Always returns true - CPU is always available
         */
        static bool isAvailable() {
            return true;
        }

        /**
         * @brief Gets the count of CPU devices (always 1).
         *
         * Returns 1 since there is always exactly one logical CPU device in the system,
         * regardless of the number of physical cores or hardware threads.
         *
         * @return Always returns 1 - there is one logical CPU device
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
         * Returns the number of hardware threads available on the system,
         * which is useful for determining optimal parallelization strategies
         * for CPU-based operations.
         *
         * @return Number of logical CPU cores/threads available
         *
         * @note This represents hardware threads, not physical cores
         * @note Used for optimizing CPU-based parallel operations
         * @note Returns at least 1 even on single-core systems
         */
        static unsigned int getLogicalCoreCount() {
            unsigned int coreCount = std::thread::hardware_concurrency();
            return (coreCount > 0) ? coreCount : 1; // Fallback to 1 if detection fails
        }

        /**
         * @brief Gets system memory information for CPU operations.
         *
         * Provides information about available system memory that can be used
         * for CPU tensor operations and memory resource planning.
         *
         * @return Available system memory in bytes, or 0 if detection fails
         *
         * @note This is an estimate and actual available memory may vary
         * @note Platform-specific implementation may be needed for accuracy
         */
        static size_t getAvailableMemory() {
            // Basic implementation - could be enhanced with platform-specific code
            // For now, return 0 to indicate memory detection not implemented
            // Future enhancement: Use platform-specific APIs to get actual memory info
            return 0;
        }

    private:
        /**
         * @brief Checks if the CPU meets minimum requirements for neural network operations.
         *
         * Validates that the CPU has sufficient capabilities for efficient neural
         * network computations. Currently performs basic validation but could be
         * extended to check for specific instruction sets (AVX, AVX2, AVX-512).
         *
         * @return true if CPU meets requirements, false otherwise
         *
         * @note Currently always returns true - all modern CPUs are suitable
         * @note Future enhancement: Check for SIMD instruction set availability
         * @note Future enhancement: Validate minimum core count or memory
         */
        static bool isCpuCapable() {
            // Basic capability check - could be enhanced with:
            // - SIMD instruction set detection (AVX, AVX2, AVX-512)
            // - Minimum core count validation
            // - Architecture-specific optimizations (ARM vs x86)

            return true; // All modern CPUs are capable of neural network operations
        }
    };
}