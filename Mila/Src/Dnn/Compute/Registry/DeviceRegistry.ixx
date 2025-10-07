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
 * - Device creation by name with consistent interfaces
 * - Thread-safe access to device factories
 * - Enumeration of available device types
 * - Runtime device availability queries
 *
 * Separation of concerns:
 * - DeviceRegistrar: Handles automatic registration through plugins (internal)
 * - DeviceRegistry: Provides runtime device queries and creation (public API)
 *
 * This registry is the primary public API for device management in the Mila framework.
 */

module;
#include <vector>
#include <unordered_map>
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
    // Forward declaration for friend access
    export class DeviceRegistrar;

    /**
     * @brief Registry for compute device creation and management.
     *
     * This singleton class provides the primary public API for device management
     * in the Mila framework. It enables runtime device queries, creation, and
     * enumeration with thread-safe access.
     *
     * Key responsibilities:
     * - Device creation by name
     * - Device availability queries
     * - Device enumeration
     * - Thread-safe access to device factories
     *
     * Device registration is handled internally by DeviceRegistrar and device plugins.
     * Applications should only use the query and creation methods.
     *
     * @note Registration is performed by DeviceRegistrar through friend access
     * @note Thread-safe for concurrent queries and device creation
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
         * @brief Creates a device instance by name.
         *
         * Creates a new instance of the specified device. The device must have been
         * previously registered by DeviceRegistrar. Returns nullptr if the device
         * name is not found in the registry.
         *
         * @param device_name The name of the device to create (e.g., "CPU", "CUDA:0").
         * @return std::shared_ptr<ComputeDevice> The created device, or nullptr if invalid.
         *
         * @note Thread-safe for concurrent device creation
         * @note Each call creates a new device instance
         * @note Returns nullptr for unregistered device names
         */
        std::shared_ptr<ComputeDevice> createDevice( const std::string& device_name ) const {
            std::lock_guard<std::mutex> lock( mutex_ );
            auto it = devices_.find( device_name );

            if (it == devices_.end()) {
                return nullptr;
            }

            return it->second();
        }

        /**
         * @brief Lists all registered device types.
         *
         * Returns a vector containing the names of all devices that have been
         * registered and are available for creation.
         *
         * @return std::vector<std::string> Vector of registered device type names.
         *
         * @note Thread-safe for concurrent access
         * @note Returns snapshot of registered devices at call time
         */
        std::vector<std::string> listDevices() const {
            std::lock_guard<std::mutex> lock( mutex_ );
            std::vector<std::string> types;
            types.reserve( devices_.size() );

            for (const auto& [type, _] : devices_) {
                types.push_back( type );
            }
            return types;
        }

        /**
         * @brief Checks if a specific device is registered.
         *
         * Queries whether a device with the specified name has been registered
         * and is available for creation.
         *
         * @param name Device identifier to check (e.g., "CPU", "CUDA:0").
         * @return true if device is registered, false otherwise.
         *
         * @note Thread-safe for concurrent access
         * @note Does not verify if device is currently functional, only if registered
         */
        bool hasDevice( const std::string& name ) const {
            std::lock_guard<std::mutex> lock( mutex_ );
            return devices_.find( name ) != devices_.end();
        }

        // Delete copy constructor and copy assignment operator
        DeviceRegistry( const DeviceRegistry& ) = delete;
        DeviceRegistry& operator=( const DeviceRegistry& ) = delete;

    private:
        friend class DeviceRegistrar;

        /**
         * @brief Private constructor for singleton pattern.
         */
        DeviceRegistry() = default;

        /**
         * @brief Registers a compute device factory function with the registry.
         *
         * This method is private and only accessible to DeviceRegistrar through
         * friend access. Device plugins use DeviceRegistrar to register their
         * devices, which then calls this method internally.
         *
         * @param name The name identifier for the device type.
         * @param factory Factory function that creates instances of the device.
         * @throws std::invalid_argument If the name is empty or the factory is null.
         *
         * @note Only callable by DeviceRegistrar
         * @note Thread-safe for concurrent registration (though typically single-threaded)
         */
        void registerDevice( const std::string& name, DeviceFactory factory ) {
            if (name.empty()) {
                throw std::invalid_argument( "Device name cannot be empty." );
            }
            if (!factory) {
                throw std::invalid_argument( "Device factory cannot be null." );
            }

            std::lock_guard<std::mutex> lock( mutex_ );
            devices_[name] = std::move( factory );
        }

        /**
         * @brief Map of device names to their factory functions.
         */
        std::unordered_map<std::string, DeviceFactory> devices_;

        /**
         * @brief Mutex for thread-safe access to the devices map.
         */
        mutable std::mutex mutex_;
    };
}