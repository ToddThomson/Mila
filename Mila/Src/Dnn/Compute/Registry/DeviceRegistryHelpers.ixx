/**
 * @file DeviceHelpers.ixx
 * @brief Utility functions for compute device discovery and management
 * @details
 * This module provides a set of helper functions for working with compute devices
 * in the Mila framework. It abstracts device management operations such as:
 * - Discovering available compute devices in the system
 * - Verifying device availability
 * - Getting device information
 *
 * These utilities help applications interact with heterogeneous compute environments
 * in a device-agnostic way, making it easier to write code that can run across
 * different device types (CPU, CUDA, etc.) without device-specific logic.
 *
 * The module depends on the DeviceRegistry to enumerate devices and DeviceContext
 * to validate device availability.
 */

module;
#include <vector>
#include <string>

export module Compute.DeviceRegistryHelpers;

import Compute.DeviceContext;
import Compute.DeviceRegistry;
import Compute.DeviceType;
import Cuda.Helpers;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Lists all available compute devices.
     *
     * This function returns a list of all available compute devices
     * that can be used with DeviceContext.
     *
     * @return std::vector<std::string> A list of device identifiers (e.g., "CPU", "CUDA:0", "CUDA:1").
     */
    export std::vector<std::string> listDevices() {
        auto& registry = DeviceRegistry::instance();
        return registry.listDevices();
    }

    /**
     * @brief Lists compute devices of a specific type.
     *
     * Filters the available devices by their type, returning only devices
     * that match the specified type. This allows clients to efficiently
     * discover devices with specific capabilities.
     *
     * @param type The device type to filter by
     * @return std::vector<std::string> List of matching device identifiers
     */
    export std::vector<std::string> listDevicesByType( DeviceType type ) {
        auto& registry = DeviceRegistry::instance();
        auto allDevices = registry.listDevices();
        std::vector<std::string> filteredDevices;
        std::string typePrefix = deviceToString( type );

        for ( const auto& device : allDevices ) {
            if ( device.find( typePrefix ) == 0 ) {
                filteredDevices.push_back( device );
            }
        }

        return filteredDevices;
    }

    /**
     * @brief Checks if a specific device is available.
     *
     * @param device_name The name of the device to check (e.g., "CPU", "CUDA:0").
     * @return bool True if the device is available, false otherwise.
     */
    /*export bool isDeviceAvailable( const std::string& device_name ) {
        auto& registry = DeviceRegistry::instance();
        return registry.hasDevice( device_name );
    }*/

    /**
     * @brief Gets the best device of a specific type based on performance characteristics.
     *
     * @param type The device type to filter by (e.g., Cuda)
     * @param preferMemory When true, prioritizes memory bandwidth over compute capability
     * @return std::string Identifier of the best available device
     */
    std::string getBestDevice( DeviceType type, bool preferMemory = false ) {
        if ( type == DeviceType::Cuda ) {
            auto cudaDevices = listDevices();
            std::vector<std::string> filteredDevices;
            std::string typePrefix = deviceToString( type );

            // Filter devices by type
            for ( const auto& device : cudaDevices ) {
                if ( device.find( typePrefix ) == 0 ) {
                    filteredDevices.push_back( device );
                }
            }

            if ( filteredDevices.empty() ) {
                return ""; // No devices of the requested type
            }

            // For CUDA devices, use the CudaHelpers to get the best device ID
            int bestId = Compute::Cuda::getBestDeviceId( preferMemory );
            std::string bestDevice = "CUDA:" + std::to_string( bestId );

            // Verify this device is actually in our list
            for ( const auto& device : filteredDevices ) {
                if ( device == bestDevice ) {
                    return device;
                }
            }

            // If the best device isn't in our list, return the first one
            return filteredDevices[ 0 ];
        }

        // For non-CUDA devices, just return the first one of that type
        auto devices = listDevices();
        std::string typePrefix = deviceToString( type );

        for ( const auto& device : devices ) {
            if ( device.find( typePrefix ) == 0 ) {
                return device;
            }
        }

        return ""; // No devices of the requested type
    }
}