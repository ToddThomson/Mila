/**
 * @file DeviceRegistry.ixx
 * @brief Provides device registration and management for the Mila deep neural network framework.
 *
 * This file implements a singleton registry pattern for compute devices, allowing
 * different device types (CPU, CUDA GPUs) to be registered, created, and managed
 * throughout the application lifecycle. The registry functions as a central point
 * for device discovery and instantiation.
 *
 * The DeviceRegistry class enables:
 * - Registration of different device types through factory functions
 * - Device creation by name with consistent interfaces
 * - Thread-safe access to device factories
 * - Enumeration of available device types
 *
 * This registry is a core infrastructure component that supports the dynamic device
 * selection and management capabilities of the Mila framework.
 */

module;
#include <vector>
#include <map>
#include <mutex>
#include <functional>
#include <string>
#include <type_traits>
#include <stdexcept>
#include <memory>

export module Compute.DeviceRegistry;

import Compute.ComputeDevice;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Registry for compute device creation and management.
     *
     * This singleton class provides functionality to register, create, and enumerate
     * compute device types. It serves as a central registry for all available compute
     * devices in the system and provides thread-safe access to them.
     */
    export class DeviceRegistry {
    public:
        /**
         * @brief Type alias for device factory functions.
         *
         * Functions of this type are used to create instances of specific compute devices.
         */
        using DeviceFactory = std::function<std::shared_ptr<ComputeDevice>()>;

        /**
         * @brief Gets the singleton instance of the DeviceRegistry.
         *
         * @return DeviceRegistry& Reference to the singleton instance.
         */
        static DeviceRegistry& instance() {
            static DeviceRegistry registry;

            return registry;
        }

        /**
         * @brief Registers a compute device factory function with the registry.
         *
         * @param name The name identifier for the device type.
         * @param factory Factory function that creates instances of the device.
         * @throws std::invalid_argument If the name is empty or the factory is null.
         */
        void registerDevice( const std::string& name, DeviceFactory factory ) {
            if ( name.empty() ) {
                throw std::invalid_argument( "Device name cannot be empty." );
            }
            if ( !factory ) {
                throw std::invalid_argument( "Device factory cannot be null." );
            }

            std::lock_guard<std::mutex> lock( mutex_ );
            devices_[ name ] = std::move( factory );
        }

        /**
         * @brief Creates a device instance by name.
         *
         * @param device_name The name of the device to create.
         * @return std::shared_ptr<ComputeDevice> The created device, or nullptr if the device name is invalid.
         */
        std::shared_ptr<ComputeDevice> createDevice( const std::string& device_name ) const {
            std::lock_guard<std::mutex> lock( mutex_ );
            auto it = devices_.find( device_name );
            if ( it == devices_.end() ) {
                return nullptr;
            }
            return it->second();
        }

        /**
         * @brief Lists all registered device types.
         *
         * @return std::vector<std::string> Vector of registered device type names.
         * @throws std::runtime_error If the registry is not initialized.
         */
        std::vector<std::string> list_devices() const {
            std::lock_guard<std::mutex> lock( mutex_ );
            std::vector<std::string> types;

            for ( const auto& [type, _] : devices_ ) {
                types.push_back( type );
            }
            return types;
        }

        bool hasDevice( const std::string& name ) const {
            std::lock_guard<std::mutex> lock( mutex_ );
            return devices_.find( name ) != devices_.end();
        }

    private:
        /**
         * @brief Private constructor for singleton pattern.
         */
        DeviceRegistry() = default;

        /**
         * @brief Map of device names to their factory functions.
         */
        //std::map<std::string, DeviceFactory, std::less<>> devices_;
        std::unordered_map<std::string, DeviceFactory> devices_;

        //std::map<std::string, DeviceFactory> devices_;

        /**
         * @brief Mutex for thread-safe access to the devices map.
         */
        mutable std::mutex mutex_;
    };
}