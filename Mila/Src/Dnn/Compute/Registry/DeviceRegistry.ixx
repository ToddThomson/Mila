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
#include <format>
#include <type_traits>
#include <stdexcept>
#include <memory>

export module Compute.DeviceRegistry;

import Compute.ComputeDevice;

namespace Mila::Dnn::Compute
{
    export class DeviceRegistry 
    {
    public:
        
        // Factory signature accepts a device index for parameterized devices (e.g., "CUDA:0")
        using DeviceFactory = std::function<std::shared_ptr<ComputeDevice>( int )>;

        static DeviceRegistry& instance() {
            static DeviceRegistry registry;
            return registry;
        }

        /**
         * @brief Gets or creates a device instance by name.
         *
         * Returns a shared device instance. Multiple calls with the same device_name
         * return the same instance, ensuring shared resources (streams, handles).
         *
         * For parameterized devices like "CUDA:0", the device index is parsed and
         * passed to the factory.
         *
         * @param device_name Device identifier (e.g., "CPU", "CUDA:0", "CUDA:1")
         * @return std::shared_ptr<ComputeDevice> Shared device instance
         * @throws std::runtime_error if device type not registered
         */
        std::shared_ptr<ComputeDevice> getDevice( const std::string& device_name ) {
            std::lock_guard<std::mutex> lock( mutex_ );

            // Check instance cache first
            auto instance_it = device_instances_.find( device_name );
            if (instance_it != device_instances_.end())
            {
                return instance_it->second;
            }

            // Parse device type and index
            auto [device_type, device_index] = parseDeviceName( device_name );

            // Find factory for device type
            auto factory_it = device_factories_.find( device_type );
            if (factory_it == device_factories_.end())
            {
                throw std::runtime_error(
                    std::format( "Device type '{}' not registered", device_type )
                );
            }

            // Create new instance and cache it
            auto device = factory_it->second( device_index );
            device_instances_[device_name] = device;
            
            return device;
        }

        /**
         * @brief Lists all registered device types.
         * @return Vector of base device types (e.g., "CPU", "CUDA")
         */
        std::vector<std::string> listDeviceTypes() const {
            std::lock_guard<std::mutex> lock( mutex_ );
            std::vector<std::string> types;
            types.reserve( device_factories_.size() );

            for (const auto& [type, _] : device_factories_)
            {
                types.push_back( type );
            }
            return types;
        }

        /**
         * @brief Lists all instantiated devices.
         * @return Vector of device instances (e.g., "CPU", "CUDA:0", "CUDA:1")
         */
        std::vector<std::string> listDevices() const {
            std::lock_guard<std::mutex> lock( mutex_ );
            std::vector<std::string> instances;
            instances.reserve( device_instances_.size() );

            for (const auto& [name, _] : device_instances_)
            {
                instances.push_back( name );
            }
            
            return instances;
        }

        /**
         * @brief Checks if device type is registered.
         */
        bool hasDeviceType( const std::string& device_type ) const {
            std::lock_guard<std::mutex> lock( mutex_ );
            return device_factories_.find( device_type ) != device_factories_.end();
        }

        DeviceRegistry( const DeviceRegistry& ) = delete;
        DeviceRegistry& operator=( const DeviceRegistry& ) = delete;

        /**
         * @brief Registers a device factory.
         *
         * Factory receives a device_index parameter for parameterized devices.
         * For CPU (no index), the index parameter is ignored.
         *
         * @param device_type Base device type (e.g., "CPU", "CUDA")
         * @param factory Function that creates device instance given an index
         */
        void registerDeviceType( const std::string& device_type, DeviceFactory factory ) {
            if (device_type.empty())
            {
                throw std::invalid_argument( "Device type cannot be empty." );
            }
            if (!factory)
            {
                throw std::invalid_argument( "Device factory cannot be null." );
            }

            std::lock_guard<std::mutex> lock( mutex_ );
            device_factories_[device_type] = std::move( factory );
        }

    private:

        DeviceRegistry() = default;

        /**
         * @brief Parses device name into type and index.
         *
         * Examples:
         *   "CPU" ? ("CPU", 0)
         *   "CUDA:0" ? ("CUDA", 0)
         *   "CUDA:1" ? ("CUDA", 1)
         *   "Metal:0" ? ("Metal", 0)
         */
        std::pair<std::string, int> parseDeviceName( const std::string& device_name ) const {
            auto colon_pos = device_name.find( ':' );

            if (colon_pos == std::string::npos)
            {
                // No index specified (e.g., "CPU")
                return { device_name, 0 };
            }

            std::string device_type = device_name.substr( 0, colon_pos );
            int device_index = std::stoi( device_name.substr( colon_pos + 1 ) );

            return { device_type, device_index };
        }

        // Factory registry: "CPU" ? factory, "CUDA" ? factory
        std::unordered_map<std::string, DeviceFactory> device_factories_;

        // Instance cache: "CPU" ? instance, "CUDA:0" ? instance, "CUDA:1" ? instance
        std::unordered_map<std::string, std::shared_ptr<ComputeDevice>> device_instances_;

        mutable std::mutex mutex_;
    };
}