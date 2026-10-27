/**
 * @file PublicAddress.ixx
 * @brief Whether an IP address is one a grant's network may connect to: a public address, reachable from anywhere.
 */

export module AI.PublicAddress;

import std;

namespace Mila::AI
{
    namespace
    {
        /// Whether the first bits of an address are those of a prefix.
        bool hasPrefix( std::span<const std::uint8_t> address, std::span<const std::uint8_t> prefix, int bits ) noexcept
        {
            for ( int bit = 0; bit < bits; ++bit )
            {
                const auto byte = static_cast<std::size_t>( bit / 8 );
                const auto mask = static_cast<std::uint8_t>( 0x80u >> ( bit % 8 ) );

                if ( ( address[ byte ] & mask ) != ( prefix[ byte ] & mask ) )
                {
                    return false;
                }
            }

            return true;
        }

        bool isPublicIpv4( std::span<const std::uint8_t> address ) noexcept
        {
            // IANA's IPv4 special-purpose registry marks each of these not globally reachable; 224.0.0.0/4 is
            // multicast. 192.0.0.0/24 is refused whole, the two anycast addresses inside it included.
            struct Block
            {
                std::array<std::uint8_t, 4> prefix;
                int bits;
            };

            static constexpr std::array<Block, 16> kRefused{ {
                { { 0, 0, 0, 0 }, 8 },
                { { 10, 0, 0, 0 }, 8 },
                { { 100, 64, 0, 0 }, 10 },
                { { 127, 0, 0, 0 }, 8 },
                { { 169, 254, 0, 0 }, 16 },
                { { 172, 16, 0, 0 }, 12 },
                { { 192, 0, 0, 0 }, 24 },
                { { 192, 0, 2, 0 }, 24 },
                { { 192, 88, 99, 0 }, 24 },
                { { 192, 168, 0, 0 }, 16 },
                { { 198, 18, 0, 0 }, 15 },
                { { 198, 51, 100, 0 }, 24 },
                { { 203, 0, 113, 0 }, 24 },
                { { 224, 0, 0, 0 }, 4 },
                { { 240, 0, 0, 0 }, 4 },
                { { 255, 255, 255, 255 }, 32 },
            } };

            return std::ranges::none_of( kRefused, [address]( const Block& block )
                {
                    return hasPrefix( address, block.prefix, block.bits );
                } );
        }

        bool isPublicIpv6( std::span<const std::uint8_t> address ) noexcept
        {
            static constexpr std::array<std::uint8_t, 16> kMapped{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff };
            static constexpr std::array<std::uint8_t, 16> kTranslated{ 0x00, 0x64, 0xff, 0x9b };
            static constexpr std::array<std::uint8_t, 16> kSixToFour{ 0x20, 0x02 };

            // An address that carries an IPv4 address reaches it, so the IPv4 address decides.
            if ( hasPrefix( address, kMapped, 96 ) || hasPrefix( address, kTranslated, 96 ) )
            {
                return isPublicIpv4( address.subspan( 12, 4 ) );
            }

            if ( hasPrefix( address, kSixToFour, 16 ) )
            {
                return isPublicIpv4( address.subspan( 2, 4 ) );
            }

            // Only global unicast, 2000::/3, is public at all; every other block IANA's IPv6 registry lists is outside
            // it -- loopback, unspecified, unique-local, link-local, multicast, discard -- save NAT64's local prefix.
            static constexpr std::array<std::uint8_t, 16> kGlobalUnicast{ 0x20 };

            if ( !hasPrefix( address, kGlobalUnicast, 3 ) )
            {
                return false;
            }

            // Inside it, the IETF's protocol assignments (refused whole) and the two documentation blocks.
            static constexpr std::array<std::uint8_t, 16> kProtocolAssignments{ 0x20, 0x01 };
            static constexpr std::array<std::uint8_t, 16> kDocumentation{ 0x20, 0x01, 0x0d, 0xb8 };
            static constexpr std::array<std::uint8_t, 16> kDocumentationSince2024{ 0x3f, 0xff };

            return !hasPrefix( address, kProtocolAssignments, 23 ) && !hasPrefix( address, kDocumentation, 32 )
                && !hasPrefix( address, kDocumentationSince2024, 20 );
        }
    }

    /**
     * @brief Whether a grant's network may connect to an address (ToolCalling.md 3.4).
     *
     * Public means globally reachable as IANA's special-purpose registries record it: never private, shared,
     * loopback, link-local, unspecified, documentation, benchmarking, reserved, broadcast or multicast. An IPv6
     * address carrying an IPv4 one -- IPv4-mapped, NAT64 or 6to4 -- is public only when that IPv4 address is.
     *
     * @param address 4 bytes for IPv4 or 16 for IPv6, in network order; any other length is not an address.
     */
    export bool isPublicAddress( std::span<const std::uint8_t> address ) noexcept
    {
        if ( address.size() == 4 )
        {
            return isPublicIpv4( address );
        }

        if ( address.size() == 16 )
        {
            return isPublicIpv6( address );
        }

        return false;
    }
}
