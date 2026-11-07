/**
 * @file DeviceRegistryHelpers.ixx
 * @brief Listing the registered devices, and finding one by its stable identity.
 *
 * Choosing among them is the deployment planner's (Specifications/Deployment.md 12.8, 12.9).
 */

export module Compute.DeviceRegistryHelpers;

import std;
import Compute.Device;
import Compute.DeviceRegistry;
import Compute.DeviceId;
import Compute.DeviceType;

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
     * @brief The registered device whose stable identity is `identity`, or nothing when no device has it.
     *
     * Matches the whole identity or, as CUDA_VISIBLE_DEVICES does, a prefix only one device has
     * ("GPU-8932f937"), ignoring case. Asking reads each registered device, so the first lookup creates every one.
     *
     * @throws std::invalid_argument when the identity is empty, or a prefix more than one device has, naming them.
     */
    export std::optional<DeviceId> findDeviceByStableIdentity( std::string_view identity )
    {
        if ( identity.empty() )
        {
            throw std::invalid_argument( "findDeviceByStableIdentity: a device identity cannot be empty" );
        }

        const auto lowered = []( std::string_view text )
            {
                std::string result( text );
                std::ranges::transform( result, result.begin(),
                    []( unsigned char c ) { return static_cast<char>( std::tolower( c ) ); } );

                return result;
            };

        const std::string wanted = lowered( identity );
        auto& registry = DeviceRegistry::instance();

        std::optional<DeviceId> exact;
        std::vector<std::pair<DeviceId, std::string>> prefixed;

        for ( const DeviceId& id : registry.listDeviceIds() )
        {
            std::string own;

            try
            {
                own = lowered( registry.getDevice( id )->getStableIdentity() );
            } catch ( const std::exception& )
            {
                // A device that refuses to be created, such as a Mac below Mila's floor, has no identity to match.
                continue;
            }

            if ( own == wanted )
            {
                exact = id;
            }
            else if ( own.starts_with( wanted ) )
            {
                prefixed.emplace_back( id, own );
            }
        }

        if ( exact )
        {
            return exact;
        }

        if ( prefixed.size() > 1 )
        {
            std::string matches;

            for ( const auto& [id, own] : prefixed )
            {
                matches += std::format( "{}{} ({})", matches.empty() ? "" : ", ", own, id.toString() );
            }

            throw std::invalid_argument( std::format(
                "'{}' is the start of more than one device's identity: {}", identity, matches ) );
        }

        if ( prefixed.size() == 1 )
        {
            return prefixed.front().first;
        }

        return std::nullopt;
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