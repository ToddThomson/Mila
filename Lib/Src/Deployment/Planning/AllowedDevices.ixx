/**
 * @file AllowedDevices.ixx
 * @brief The devices a deployment may use: a set with no order, every device when it names none.
 *
 * See Specifications/Deployment.md 12.8.
 */

export module Deployment.AllowedDevices;

import std;
import Compute.DeviceId;
import Compute.DeviceType;
import Deployment.DeviceReading;

namespace Mila::Deployment
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief The devices the planner may choose among, as the user named them.
     *
     * A device is named three ways: every device of a type ("cuda"), an index in this run ("cuda:1"), or a stable
     * identity ("GPU-8932f937-..."), which does not move with CUDA_VISIBLE_DEVICES and tells two cards of one model
     * apart. The set has no order: which allowed device a model runs on is the planner's choice. A name that matches
     * no present device is refused by the planner, never replaced by another device.
     */
    export class AllowedDevices
    {
    public:

        /// Every device the build sees.
        AllowedDevices() = default;

        /// The one device: a set of one, which a plan never splits.
        AllowedDevices( DeviceId device )
        {
            entries_.push_back( Entry{ device, device.toString() } );
        }

        AllowedDevices( std::initializer_list<DeviceId> devices )
        {
            for ( const DeviceId device : devices )
            {
                entries_.push_back( Entry{ device, device.toString() } );
            }
        }

        /**
         * @brief Read the devices a person or a settings file wrote, separated by commas.
         *
         * Each name is a device type ("cuda"), a type and index ("cuda:1"), or anything else, which is taken as a
         * stable identity. Case does not matter.
         *
         * @throws std::invalid_argument on an empty list or name, or a type:index that does not parse.
         */
        static AllowedDevices parse( std::string_view text )
        {
            AllowedDevices allowed;
            std::size_t start = 0;

            while ( start <= text.size() )
            {
                const std::size_t comma = std::min( text.find( ',', start ), text.size() );
                const std::string_view name = trim( text.substr( start, comma - start ) );

                if ( name.empty() )
                {
                    throw std::invalid_argument( std::format(
                        "AllowedDevices: '{}' names an empty device; write devices as cuda, cuda:1 or a stable identity, "
                        "separated by commas", text ) );
                }

                allowed.entries_.push_back( entryOf( name ) );
                start = comma + 1;
            }

            return allowed;
        }

        /// True when the request named no device, so every device the build sees is allowed.
        [[nodiscard]] bool isEveryDevice() const noexcept
        {
            return entries_.empty();
        }

        /// True when this set allows the device the description is of.
        [[nodiscard]] bool allows( const DeviceReading& device ) const
        {
            return isEveryDevice()
                || std::ranges::any_of( entries_, [&]( const Entry& entry ) { return matches( entry, device ); } );
        }

        /**
         * @brief The names that match no present device, as they were written.
         *
         * "Every device" has none: it allows what is there.
         */
        [[nodiscard]] std::vector<std::string> absentFrom( std::span<const DeviceReading> present ) const
        {
            std::vector<std::string> absent;

            for ( const Entry& entry : entries_ )
            {
                const bool found = std::ranges::any_of( present,
                    [&]( const DeviceReading& device ) { return matches( entry, device ); } );

                if ( !found )
                    absent.push_back( entry.text );
            }

            return absent;
        }

        /// "every device", or the names as written, separated by commas.
        [[nodiscard]] std::string toString() const
        {
            if ( isEveryDevice() )
                return "every device";

            std::string text;

            for ( const Entry& entry : entries_ )
            {
                if ( !text.empty() )
                    text += ", ";

                text += entry.text;
            }

            return text;
        }

    private:

        /// A device type stands for every device of it; a string is a stable identity.
        struct Entry
        {
            std::variant<DeviceType, DeviceId, std::string> name;
            std::string text;
        };

        static Entry entryOf( std::string_view name )
        {
            if ( name.find( ':' ) != std::string_view::npos )
            {
                return Entry{ DeviceId::parse( name ), std::string( name ) };
            }

            try
            {
                return Entry{ toDeviceType( name ), std::string( name ) };
            } catch ( const std::invalid_argument& )
            {
                return Entry{ std::string( name ), std::string( name ) };
            }
        }

        static bool matches( const Entry& entry, const DeviceReading& device )
        {
            if ( const auto* type = std::get_if<DeviceType>( &entry.name ) )
                return device.device.type == *type;

            if ( const auto* id = std::get_if<DeviceId>( &entry.name ) )
                return device.device == *id;

            return equalIgnoringCase( std::get<std::string>( entry.name ), device.stable_identity );
        }

        static bool equalIgnoringCase( std::string_view left, std::string_view right ) noexcept
        {
            return left.size() == right.size()
                && std::ranges::equal( left, right, []( unsigned char a, unsigned char b )
                    {
                        return std::tolower( a ) == std::tolower( b );
                    } );
        }

        static std::string_view trim( std::string_view text ) noexcept
        {
            const std::size_t first = text.find_first_not_of( " \t" );

            if ( first == std::string_view::npos )
                return {};

            const std::size_t last = text.find_last_not_of( " \t" );

            return text.substr( first, last - first + 1 );
        }

        std::vector<Entry> entries_;
    };
}
