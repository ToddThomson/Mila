/**
 * @file DeviceId.ixx
 * @brief Lightweight device identifier value type.
 *
 * Provides a compact, comparable representation of a device type and index
 * (e.g., "Cuda:0", "CPU:0"). Typically constructed via Device factory methods
 * (Device::Cuda(0), Device::Cpu()) rather than directly.
 */

module;
#include <string>
#include <string_view>
#include <compare>
#include <stdexcept>
#include <functional>

export module Compute.DeviceId;

import Compute.DeviceType;

namespace Mila::Dnn::Compute
{
    /**
     * @brief Lightweight identifier for a compute device.
     *
     * Pairs a DeviceType with a zero-based index. Used throughout the API
     * as a value type for device identification, function parameters, and
     * class members.
     *
     * Construction:
     * - Preferred: Use Device factory methods (Device::Cuda(0), Device::Cpu())
     * - Direct: DeviceId(DeviceType::Cuda, 0) for low-level code
     * - Parsing: DeviceId::parse("cuda:0") for configuration files
     *
     * Equality- and order-comparable. Suitable as key in associative containers.
     */
    export struct DeviceId
    {
        /// The device type (e.g., Cuda, Cpu, Metal, Rocm).
        DeviceType type;

        /// Zero-based device index for the given type.
        int index = -1;

        /**
         * @brief Construct a default DeviceId.
         *
         * Default constructs to Cuda device 0.
         */
        constexpr DeviceId() noexcept
            : type( DeviceType::Cuda ), index( 0 )
        {
        }

        /**
         * @brief Construct a DeviceId from components.
         *
         * @param type  DeviceType enum value.
         * @param index Zero-based device index (default 0).
         */
        constexpr DeviceId( DeviceType type, int index = 0 ) noexcept
            : type( type ), index( index )
        {
        }

        /**
         * @brief Three-way comparison for DeviceId.
         *
         * Provides total ordering and equality. Comparison is performed by
         * first comparing type, then index.
         */
        constexpr auto operator<=>( const DeviceId& ) const noexcept = default;

        /**
         * @brief Parse a textual device identifier.
         *
         * Accepted forms:
         * - "<type>"          -> interpreted as "<type>:0"
         * - "<type>:<index>"  -> explicit zero-based index
         *
         * Examples: "cuda:0", "cpu", "metal:1", "rocm:2"
         *
         * @param device_name Text to parse (case-insensitive for type).
         * @return DeviceId Parsed device identifier.
         * @throws std::invalid_argument If format is invalid or type is unknown.
         */
        static DeviceId parse( std::string_view device_name )
        {
            auto pos = device_name.find( ':' );

            if ( pos == std::string_view::npos )
            {
                return DeviceId( toDeviceType( device_name ), 0 );
            }

            auto type_str = device_name.substr( 0, pos );
            auto index_str = device_name.substr( pos + 1 );

            int idx = 0;
            try
            {
                idx = std::stoi( std::string( index_str ) );
            }
            catch ( ... )
            {
                throw std::invalid_argument( "Invalid device index" );
            }

            return DeviceId( toDeviceType( type_str ), idx );
        }

        /**
         * @brief Convert to human-readable string.
         *
         * Produces "<type>:<index>" using deviceTypeToString() for the type
         * component (e.g., "Cuda:0", "CPU:0", "Metal:1").
         *
         * Useful for logging, diagnostics, and configuration serialization.
         *
         * @return std::string Device name in "<type>:<index>" format.
         */
        std::string toString() const
        {
            return std::string( deviceTypeToString( type ) ) + ":" + std::to_string( index );
        }
    };
}

namespace std
{
    /**
     * @brief Hash specialization for DeviceId.
     *
     * Enables use of DeviceId as a key in unordered associative containers
     * (std::unordered_map, std::unordered_set).
     */
    template<>
    struct hash<Mila::Dnn::Compute::DeviceId>
    {
        /**
         * @brief Compute hash value for a DeviceId.
         *
         * Combines hashes of type and index using a deterministic mixing function.
         *
         * @param id DeviceId to hash.
         * @return size_t Combined hash value.
         */
        std::size_t operator()( const Mila::Dnn::Compute::DeviceId& id ) const noexcept
        {
            std::size_t h1 = std::hash<Mila::Dnn::Compute::DeviceType>{}( id.type );
            std::size_t h2 = std::hash<int>{}( id.index );

            std::size_t seed = h1;
            seed ^= ( h2 + 0x9e3779b97f4a7c15ULL + ( seed << 6 ) + ( seed >> 2 ) );

            return seed;
        }
    };
}