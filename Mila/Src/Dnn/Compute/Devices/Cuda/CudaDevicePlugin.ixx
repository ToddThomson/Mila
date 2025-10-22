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
#include <functional>
#include <optional>

export module Compute.CudaDevicePlugin;

import Compute.DeviceRegistry;
import Compute.ComputeDevice;
import Compute.CudaDevice;

namespace Mila::Dnn::Compute
{
    /**
     * @brief CUDA device plugin for device-agnostic registration.
     *
     * This plugin encapsulates all CUDA-specific logic for device discovery and
     * registration, providing a clean static interface while handling CUDA runtime
     * API interactions internally.
     */
    export class CudaDevicePlugin {
    public:
        /**
         * @brief Returns an optional index-aware factory for CUDA devices.
         *
         * If no usable CUDA devices are present, returns std::nullopt. Otherwise
         * returns a factory that constructs a `CudaDevice` for the requested index.
         *
         * @return Optional factory callable taking an int index and returning a ComputeDevice instance.
         */
        static std::optional<std::function<std::shared_ptr<ComputeDevice>(int)>> getDeviceFactory() {
            try {
                int deviceCount = 0;
                cudaError_t error = cudaGetDeviceCount( &deviceCount );

                if (error != cudaSuccess || deviceCount == 0) {
                    return std::nullopt;
                }

                bool foundUsable = false;
                for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
                    if (isDeviceUsable( deviceId )) {
                        foundUsable = true;
                        break;
                    }
                }

                if (!foundUsable) {
                    return std::nullopt;
                }

                // Use CudaDevice::create(...) factory instead of calling the constructor directly.
                // This ensures all device instances are created through the controlled factory,
                // matching the DeviceRegistry and preventing accidental direct construction.
                return std::function<std::shared_ptr<ComputeDevice>(int)>( []( int deviceIndex ) {
                    return CudaDevice::create( deviceIndex );
                } );
            }
            catch (...) {
                return std::nullopt;
            }
        }

        /**
         * @brief Checks if the CUDA runtime is available and functional.
         *
         * @return true if CUDA runtime is available and functional, false otherwise
         */
        static bool isAvailable() {
            try {
                int deviceCount = 0;
                cudaError_t error = cudaGetDeviceCount( &deviceCount );
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
         */
        static int getDeviceCount() {
            try {
                int deviceCount = 0;
                cudaError_t error = cudaGetDeviceCount( &deviceCount );
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
         * @deprecated Use isAvailable() instead
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
         * meets minimum requirements for neural network operations.
         *
         * @param deviceId CUDA device index to check
         * @return true if device is usable, false otherwise
         */
        static bool isDeviceUsable( int deviceId ) {
            try {
                cudaDeviceProp deviceProp;
                cudaError_t error = cudaGetDeviceProperties( &deviceProp, deviceId );

                if (error != cudaSuccess) {
                    return false;
                }

                if (deviceProp.major < 3) {
                    return false;
                }

                if (deviceProp.totalGlobalMem < (1ULL << 30)) {
                    return false;
                }

                int computeMode = -1;
                error = cudaDeviceGetAttribute( &computeMode, cudaDevAttrComputeMode, deviceId );

                if (computeMode == cudaComputeModeProhibited) {
                    return false;
                }

                cudaError_t setError = cudaSetDevice( deviceId );
                if (setError != cudaSuccess) {
                    return false;
                }

                cudaDeviceReset();

                return true;
            }
            catch (...) {
                return false;
            }
        }
    };
}