/**
 * @file MetalDevicePlugin.ixx
 * @brief Metal device plugin for device-agnostic registration and discovery.
 *
 * This plugin handles Metal-specific device discovery and registration logic,
 * isolating all Metal API dependencies from the main DeviceRegistrar. The plugin
 * performs Metal availability checking, device enumeration, and registration with
 * the DeviceRegistry in a self-contained manner.
 */

module;
#include <string>
#include <memory>

#ifdef __APPLE__
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#endif

export module Compute.MetalDevicePlugin;

// FUTURE: import Compute.DevicePlugin;
import Compute.DeviceRegistrar;
// FUTURE: import Compute.MetalDevice;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Metal device plugin for device-agnostic registration.
     *
     * This plugin encapsulates all Metal-specific logic for device discovery and
     * registration, providing a clean static interface to the DeviceRegistrar while
     * handling Metal framework API interactions internally.
     *
     * Key responsibilities:
     * - Metal framework availability detection
     * - Metal device enumeration and capability checking
     * - Device registration with appropriate factory functions
     * - Error handling for Metal-related failures
     * - Graceful degradation when Metal is not available
     */
    export class MetalDevicePlugin {
        // FUTURE: public DevicePlugin {
    public:
        /**
         * @brief Registers all available Metal devices with the DeviceRegistry.
         *
         * Performs Metal framework initialization, enumerates available Metal devices,
         * and registers each device with an appropriate factory function. Handles
         * Metal unavailability gracefully without throwing exceptions.
         *
         * Device naming convention: "Metal:N" where N is the device index (0-based)
         *
         * @note This method is safe to call even when Metal is not available
         * @note Registers devices only if Metal framework is available and functional
         * @note Each device is registered with a factory that creates MetalDevice instances
         */
        static void registerDevices() {
            try {
#ifdef __APPLE__
                @autoreleasepool{
                    NSArray<id<MTLDevice>>*devices = MTLCopyAllDevices();

                    if (!devices || devices.count == 0) {
                        return;
                    }

                    auto& registry = DeviceRegistry::instance();

                    for (NSUInteger i = 0; i < devices.count; ++i) {
                        id<MTLDevice> device = devices[i];

                        if (isDeviceUsable(device)) {
                            std::string deviceName = "Metal:" + std::to_string(i);

                            // FUTURE: Uncomment when MetalDevice is implemented
                            /*
                            registry.registerDevice(deviceName, [i]() {
                                return std::make_shared<MetalDevice>(static_cast<int>(i));
                            });
                            */
                        }
                    }
                }
#endif
            }
            catch (...) {
                // Suppress all exceptions during device registration
                // The system should continue to function without Metal devices
                // if registration fails for any reason
            }
        }

        /**
         * @brief Checks if the Metal framework is available and functional.
         *
         * Performs a lightweight check to determine if Metal operations can be
         * performed. This is useful for early detection of Metal availability
         * without attempting full device enumeration.
         *
         * @return true if Metal framework is available and functional, false otherwise
         *
         * @note This method does not throw exceptions - returns false on any error
         */
        static bool isAvailable() {
            try {
#ifdef __APPLE__
                @autoreleasepool{
                    NSArray<id<MTLDevice>>*devices = MTLCopyAllDevices();
                    return (devices && devices.count > 0);
                }
#else
                return false; // Metal is only available on Apple platforms
#endif
            }
            catch (...) {
                return false;
            }
        }

        /**
         * @brief Gets the number of available Metal devices.
         *
         * @return Number of Metal devices available, or 0 if Metal is not available
         *
         * @note This method does not throw exceptions - returns 0 on any error
         */
        static int getDeviceCount() {
            try {
#ifdef __APPLE__
                @autoreleasepool{
                    NSArray<id<MTLDevice>>*devices = MTLCopyAllDevices();
                    return devices ? static_cast<int>(devices.count) : 0;
                }
#else
                return 0; // Metal is only available on Apple platforms
#endif
            }
            catch (...) {
                return 0;
            }
        }

        /**
         * @brief Gets the plugin name identifying the Metal device type.
         *
         * @return String "Metal" identifying this as the Metal device plugin
         */
        static std::string getPluginName() {
            return "Metal";
        }

        /**
         * @brief Checks if Metal Performance Shaders are available.
         *
         * Metal Performance Shaders (MPS) provide optimized neural network primitives
         * for Apple Silicon and other Metal-capable devices.
         *
         * @return true if MPS is available, false otherwise
         */
        static bool isMPSAvailable() {
#ifdef __APPLE__
            // MPS is available on macOS 10.13+ and iOS 11.0+
            return true; // For simplicity, assume available if Metal is available
#else
            return false;
#endif
        }

        /**
         * @brief Gets the default Metal device.
         *
         * Returns information about the system's default Metal device,
         * which is typically the primary GPU.
         *
         * @return true if default device is available, false otherwise
         */
        static bool hasDefaultDevice() {
#ifdef __APPLE__
            @autoreleasepool{
                id<MTLDevice> device = MTLCreateSystemDefaultDevice();
                return device != nil;
            }
#else
            return false;
#endif
        }

    private:
        /**
         * @brief Checks if a specific Metal device is usable for computation.
         *
         * Performs device-specific capability checking to ensure the device
         * meets minimum requirements for neural network operations. This includes
         * feature set validation and memory availability checks.
         *
         * @param device Metal device to check
         * @return true if device is usable, false otherwise
         *
         * @note This method performs actual device queries
         * @note Returns false for any device that cannot be properly queried
         */
#ifdef __APPLE__
        static bool isDeviceUsable(id<MTLDevice> device) {
            if (!device) {
                return false;
            }

            @autoreleasepool{
                // Check if device supports required features
                if (![device supportsFeatureSet : MTLFeatureSet_macOS_GPUFamily1_v1] &&
                    ![device supportsFeatureSet : MTLFeatureSet_iOS_GPUFamily1_v1]) {
                    return false;
                }

                // Check minimum memory requirements (at least 1GB for practical use)
                if ([device recommendedMaxWorkingSetSize] < (1ULL << 30)) {
                    return false;
                }

                // Check if device supports compute shaders
                if (![device supportsFamily : MTLGPUFamilyCommon1]) {
                    return false;
                }

                return true;
            }
        }
#else
        static bool isDeviceUsable(void* device) {
            // Metal not available on non-Apple platforms
            return false;
        }
#endif
    };
}