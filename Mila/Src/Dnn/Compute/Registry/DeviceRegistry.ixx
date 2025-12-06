/**
 * @file DeviceRegistry.ixx
 * @brief Central registry for discovered compute devices.
 *
 * Stores registered DeviceId values and creates concrete Device instances
 * on-demand with lazy initialization and caching. Registrars register lightweight
 * DeviceId values; Device objects with cached properties are created when first
 * requested via getDevice().
 */

module;
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <format>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <utility>

export module Compute.DeviceRegistry;

import Compute.DeviceId;
import Compute.DeviceType;
import Compute.Device;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Construction key for device factories.
     *
     * Passkey pattern token that restricts Device subclass construction to
     * authorized factory contexts. Only DeviceRegistry can create instances
     * of this key, ensuring devices are only constructed through the registry.
     */
    export class DeviceConstructionKey
    {
    private:
        friend class DeviceRegistry;

        DeviceConstructionKey() = default;
    };

    /**
     * @brief Registry of discovered compute devices with lazy instantiation.
     *
     * Thread-safe singleton that stores registered DeviceId values and creates
     * concrete Device instances on-demand. Device registrars register lightweight
     * DeviceId values during initialization; Device objects are created lazily
     * when first requested and then cached for subsequent calls.
     */
    export class DeviceRegistry
    {
    public:

		using DeviceFactory = std::function<std::shared_ptr<Device>( DeviceConstructionKey )>;

        static DeviceRegistry& instance()
        {
            static DeviceRegistry registry;
            
            return registry;
        }

        /**
         * @brief Register a discovered DeviceId with factory callback.
         *
         * @param id Device identifier to register
         * @param factory Factory function to create the device instance
         * @throws std::runtime_error If DeviceId is already registered
         */
        void registerDevice( DeviceId id, DeviceFactory factory )
        {
            std::lock_guard<std::mutex> lock( mutex_ );

            if ( std::ranges::find( registered_ids_, id ) != registered_ids_.end() )
            {
                throw std::runtime_error(
                    std::format( "DeviceId '{}' already registered", id.toString() )
                );
            }

            registered_ids_.push_back( id );
            factories_[id] = std::move( factory );
        }
        
        /**
         * @brief Retrieve or create a device by ID.
         *
         * Validates that the DeviceId is registered, then returns a cached
         * Device instance. If this is the first request, invokes the factory
         * callback to create the device and caches it.
         *
         * @param id Device identifier (e.g., Device::Cuda(0), Device::Cpu())
         * @return std::shared_ptr<Device> Cached device instance with properties
         * @throws std::runtime_error If DeviceId is not registered or creation fails
         */
        std::shared_ptr<Device> getDevice( DeviceId id ) const
        {
            std::lock_guard<std::mutex> lock( mutex_ );

            if ( std::ranges::find( registered_ids_, id ) == registered_ids_.end() )
            {
                throw std::runtime_error(
                    std::format( "Device '{}' not registered", id.toString() )
                );
            }

            auto cache_it = device_cache_.find( id );

            if ( cache_it != device_cache_.end() )
            {
                return cache_it->second;
            }

            auto factory_it = factories_.find( id );

            if ( factory_it == factories_.end() )
            {
                throw std::runtime_error(
                    std::format( "No factory registered for device '{}'", id.toString() )
                );
            }

            std::shared_ptr<Device> device = factory_it->second( DeviceConstructionKey{} );
            device_cache_[id] = device;

            return device;
        }

        /**
         * @brief Check whether a specific DeviceType has been registered.
         *
         * @param type Device type to check (Cpu, Cuda, Metal, Rocm)
         * @return bool True if at least one device of this type is registered
         */
        bool hasDeviceType( DeviceType type ) const
        {
            std::lock_guard<std::mutex> lock( mutex_ );

            return std::ranges::any_of( registered_ids_,
                [type]( const DeviceId& id ) { return id.type == type; } );
        }

        /**
         * @brief Non-throwing check whether a specific DeviceId has been registered.
         *
         * @param id Device identifier to check
         * @return bool True if device is registered, false otherwise (or on error)
         */
        bool hasDevice( DeviceId id ) const noexcept
        {
            try
            {
                std::lock_guard<std::mutex> lock( mutex_ );
                return std::ranges::find( registered_ids_, id ) != registered_ids_.end();
            }
            catch ( ... )
            {
                return false;
            }
        }

        /**
         * @brief List registered device types (unique).
         *
         * @return std::vector<DeviceType> Vector of unique device types
         */
        std::vector<DeviceType> listDeviceTypes() const
        {
            std::lock_guard<std::mutex> lock( mutex_ );

            std::vector<DeviceType> unique;
            unique.reserve( registered_ids_.size() );

            for ( const auto& id : registered_ids_ )
            {
                if ( std::ranges::find( unique, id.type ) == unique.end() )
                {
                    unique.push_back( id.type );
                }
            }

            return unique;
        }

        /**
         * @brief List registered DeviceId values.
         *
         * @return std::vector<DeviceId> Vector of registered device identifiers
         */
        std::vector<DeviceId> listDeviceIds() const
        {
            std::lock_guard<std::mutex> lock( mutex_ );

            std::vector<DeviceId> ids;
            ids.reserve( registered_ids_.size() );

            for ( const auto& id : registered_ids_ )
            {
                ids.push_back( id );
            }

            return ids;
        }

        /**
         * @brief Return number of registered devices for the given DeviceType.
         *
         * @param device_type Device type to count
         * @return std::size_t Number of registered devices of this type (0 on error)
         */
        std::size_t getDeviceCount( DeviceType device_type ) const noexcept
        {
            try
            {
                std::lock_guard<std::mutex> lock( mutex_ );

                return std::ranges::count_if( registered_ids_,
                    [device_type]( const DeviceId& id ) { return id.type == device_type; } );
            }
            catch ( ... )
            {
                return 0;
            }
        }

        DeviceRegistry( const DeviceRegistry& ) = delete;
        DeviceRegistry& operator=( const DeviceRegistry& ) = delete;

    private:
        DeviceRegistry() = default;

        // Use a simple vector to avoid hash-based template instantiation that
        // can trigger compiler internal errors on some toolchains.
        std::vector<DeviceId> registered_ids_;
        std::unordered_map<DeviceId, DeviceFactory> factories_;
        mutable std::unordered_map<DeviceId, std::shared_ptr<Device>> device_cache_;

        mutable std::mutex mutex_;
    };
}