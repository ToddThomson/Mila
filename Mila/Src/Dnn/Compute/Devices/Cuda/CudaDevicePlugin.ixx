/**
 * @file CudaDevicePlugin.ixx
 * @brief CUDA device plugin for device-agnostic registration and discovery.
 *
 * This plugin handles CUDA-specific device discovery and registration logic,
 * isolating all CUDA API dependencies from the main DeviceRegistrar. The plugin
 * performs CUDA availability checking, device enumeration, and registration with
 * the DeviceRegistry in a self-contained manner.
 */

module;
#include <cuda_runtime.h>
#include <string>
#include <memory>

export module Compute.CudaDevicePlugin;

// FUTURE: import Compute.DevicePlugin;
import Compute.DeviceRegistry;
import Compute.CudaDevice;

namespace Mila::Dnn::Compute
{
    /**
     * @brief CUDA device plugin for device-agnostic registration.
     *
     * This plugin encapsulates all CUDA-specific logic for device discovery and
     * registration, providing a clean static interface to the DeviceRegistrar while
     * handling CUDA runtime API interactions internally.
     *
     * Key responsibilities:
     * - CUDA runtime availability detection
     * - CUDA device enumeration and capability checking
     * - Device registration with appropriate factory functions
     * - Error handling for CUDA-related failures
     * - Graceful degradation when CUDA is not available
     */
    export class CudaDevicePlugin {
        // FUTURE: public DevicePlugin {
    public:
        /**
         * @brief Registers all available CUDA devices with the DeviceRegistry.
         *
         * Performs CUDA runtime initialization, enumerates available CUDA devices,
         * and registers each device with an appropriate factory function. Handles
         * CUDA unavailability gracefully without throwing exceptions.
         *
         * Device naming convention: "CUDA:N" where N is the device index (0-based)
         *
         * @note This method is safe to call even when CUDA is not available
         * @note Registers devices only if CUDA runtime is available and functional
         * @note Each device is registered with a factory that creates CudaDevice instances
         */
        static void registerDevices() {
            try {
                int deviceCount = 0;
                cudaError_t error = cudaGetDeviceCount(&deviceCount);

                // Handle CUDA unavailability gracefully
                if (error != cudaSuccess) {
                    // CUDA is not available - this is not an error condition
                    // Simply don't register any CUDA devices
                    return;
                }

                if (deviceCount == 0) {
                    return;
                }

                // Register each available CUDA device
                auto& registry = DeviceRegistry::instance();

                for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
                    if (isDeviceUsable(deviceId)) {
                        std::string deviceName = "CUDA:" + std::to_string(deviceId);

                        registry.registerDevice(deviceName, [deviceId]() {
                            return std::make_shared<CudaDevice>(deviceId);
                            });
                    }
                }
            }
            catch (...) {
                // Suppress all exceptions during device registration
                // The system should continue to function without CUDA devices
                // if registration fails for any reason
            }
        }

        /**
         * @brief Checks if the CUDA runtime is available and functional.
         *
         * Performs a lightweight check to determine if CUDA operations can be
         * performed. This is useful for early detection of CUDA availability
         * without attempting full device enumeration.
         *
         * @return true if CUDA runtime is available and functional, false otherwise
         *
         * @note This method does not throw exceptions - returns false on any error
         */
        static bool isAvailable() {
            try {
                int deviceCount = 0;
                cudaError_t error = cudaGetDeviceCount(&deviceCount);
                return (error == cudaSuccess && deviceCount > 0);
            }
            catch (...) {
                return false;
            }
        }

        /**
         * @brief Gets the number of available CUDA devices.
         *
         * @return Number of CUDA devices available, or 0 if CUDA is not available
         *
         * @note This method does not throw exceptions - returns 0 on any error
         */
        static int getDeviceCount() {
            try {
                int deviceCount = 0;
                cudaError_t error = cudaGetDeviceCount(&deviceCount);
                return (error == cudaSuccess) ? deviceCount : 0;
            }
            catch (...) {
                return 0;
            }
        }

        /**
         * @brief Gets the plugin name identifying the CUDA device type.
         *
         * @return String "CUDA" identifying this as the CUDA device plugin
         */
        static std::string getPluginName() {
            return "CUDA";
        }

        /**
         * @brief Legacy method name for CUDA availability checking.
         *
         * @deprecated Use isAvailable() instead for consistency with plugin interface
         * @return true if CUDA runtime is available and functional, false otherwise
         */
        static bool isCudaAvailable() {
            return isAvailable();
        }

    private:
        /**
         * @brief Checks if a specific CUDA device is usable for computation.
         *
         * Performs device-specific capability checking to ensure the device
         * meets minimum requirements for neural network operations. This includes
         * compute capability validation and memory availability checks.
         *
         * @param deviceId CUDA device index to check
         * @return true if device is usable, false otherwise
         *
         * @note This method performs actual device queries and may be slower than basic enumeration
         * @note Returns false for any device that cannot be properly queried
         */
        static bool isDeviceUsable(int deviceId) {
            try {
                cudaDeviceProp deviceProp;
                cudaError_t error = cudaGetDeviceProperties(&deviceProp, deviceId);

                if (error != cudaSuccess) {
                    return false;
                }

                // Check minimum compute capability (3.0 for modern neural network operations)
                if (deviceProp.major < 3) {
                    return false;
                }

                // Check minimum memory requirements (at least 1GB for practical use)
                if (deviceProp.totalGlobalMem < (1ULL << 30)) { // 1GB
                    return false;
                }

                int computeMode = -1;
                error = cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, deviceId);

                if (computeMode == cudaComputeModeProhibited) {
                    return false;
                }

                // Test basic device accessibility
                cudaError_t setError = cudaSetDevice(deviceId);
                if (setError != cudaSuccess) {
                    return false;
                }

                // Reset device to clean up any side effects
                cudaDeviceReset();

                return true;
            }
            catch (...) {
                return false;
            }
        }
    };
}