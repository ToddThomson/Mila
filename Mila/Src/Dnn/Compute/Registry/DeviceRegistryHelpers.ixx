/**
 * @file DeviceRegistryHelpers.ixx
 * @brief Utility functions for compute device discovery and management
 *
 * Small helpers for listing and selecting devices via DeviceRegistry.
 */

module;
#include <vector>
#include <string>

export module Compute.DeviceRegistryHelpers;

import Compute.DeviceRegistry;
import Compute.DeviceId;
import Compute.DeviceType;
import Cuda.Helpers;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Lists all available compute devices by name.
     *
     * @return std::vector<std::string> A list of device identifiers (e.g., "CUDA:0", "CPU:0").
     */
    export std::vector<std::string> listDevicesByName()
    {
        auto& registry = DeviceRegistry::instance();

        auto deviceIds = registry.listDeviceIds();

        std::vector<std::string> names;
        names.reserve( deviceIds.size() );

        for ( const auto& id : deviceIds )
        {
            names.push_back( id.toString() );
        }

        return names;
    }

    /**
     * @brief Lists compute devices of a specific type.
     *
     * @param type The device type to filter by
     * @return std::vector<std::string> List of matching device identifiers
     */
    export std::vector<std::string> listDevicesByType( DeviceType type )
    {
        auto& registry = DeviceRegistry::instance();

        auto allDeviceIds = registry.listDeviceIds();

        std::vector<std::string> filteredDevices;
        filteredDevices.reserve( allDeviceIds.size() );

        for ( const auto& id : allDeviceIds )
        {
            if ( id.type == type )
            {
                filteredDevices.push_back( id.toString() );
            }
        }

        return filteredDevices;
    }

    /**
     * @brief Gets the best DeviceId of a specific type based on performance characteristics.
     *
     * For CUDA the function consults Cuda::getBestDeviceId(preferMemory) and returns
     * the matching DeviceId when available. If no device of the requested type is found
     * the function returns a DeviceId with the requested type and index == -1 to indicate
     * "none".
     *
     * @param type The device type to filter by (e.g., DeviceType::Cuda)
     * @param preferMemory When true, prioritizes memory bandwidth over compute capability
     * @return DeviceId Best available device id for the requested type, or index == -1 if none
     */
    export DeviceId getBestDevice( DeviceType type, bool preferMemory = false )
    {
        auto& registry = DeviceRegistry::instance();

        auto allDeviceIds = registry.listDeviceIds();

        if ( type == DeviceType::Cuda )
        {
            std::vector<DeviceId> cudaIds;
            cudaIds.reserve( allDeviceIds.size() );

            for ( const auto& id : allDeviceIds )
            {
                if ( id.type == DeviceType::Cuda )
                {
                    cudaIds.push_back( id );
                }
            }

            if ( cudaIds.empty() )
            {
                return DeviceId{ DeviceType::Cuda, -1 };
            }

            int bestId = Compute::Cuda::getBestDeviceId( preferMemory );

            DeviceId bestDeviceId{ DeviceType::Cuda, bestId };

            for ( const auto& id : cudaIds )
            {
                if ( id == bestDeviceId )
                {
                    return id;
                }
            }

            return cudaIds.front();
        }

        for ( const auto& id : allDeviceIds )
        {
            if ( id.type == type )
            {
                return id;
            }
        }

        return DeviceId{ type, -1 };
    }

    /**
     * @brief Count instantiated devices of the given DeviceType.
     *
     * Lightweight, thread-safe convenience helper intended for tests and diagnostics.
     * Non-throwing; returns 0 on error.
     */
    export std::size_t getDeviceCount( DeviceType type ) noexcept
    {
        try
        {
            auto& registry = DeviceRegistry::instance();

            auto ids = registry.listDeviceIds();

            std::size_t count = 0;

            for ( const auto& id : ids )
            {
                if ( id.type == type )
                {
                    ++count;
                }
            }

            return count;
        }
        catch ( ... )
        {
            return 0;
        }
    }
}