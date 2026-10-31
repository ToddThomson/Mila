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

        template <std::size_t TBytes>
        struct Block
        {
            std::array<std::uint8_t, TBytes> prefix;
            int bits;
        };

        // IANA's IPv4 special-purpose registry marks each of these not globally reachable; 224.0.0.0/4 is
        // multicast. 192.0.0.0/24 is refused whole, the two anycast addresses inside it included.
        constexpr std::array<Block<4>, 16> kRefusedIpv4{ {
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

        // Only global unicast, 2000::/3, is public at all, so the three blocks beside it are refused; every other
        // block IANA's IPv6 registry lists lies in them -- loopback, unspecified, unique-local, link-local,
        // multicast, discard, NAT64's local prefix. Inside it, the IETF's protocol assignments (refused whole) and
        // the two documentation blocks.
        constexpr std::array<Block<16>, 6> kRefusedIpv6{ {
            { { 0x00 }, 3 },
            { { 0x40 }, 2 },
            { { 0x80 }, 1 },
            { { 0x20, 0x01 }, 23 },
            { { 0x20, 0x01, 0x0d, 0xb8 }, 32 },
            { { 0x3f, 0xff }, 20 },
        } };

        // IPv6 blocks that carry an IPv4 address, which then decides; the first two lie in ::/3.
        constexpr Block<16> kMapped{ { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff }, 96 };
        constexpr Block<16> kTranslated{ { 0x00, 0x64, 0xff, 0x9b }, 96 };
        constexpr Block<16> kSixToFour{ { 0x20, 0x02 }, 16 };

        template <std::size_t TBytes, std::size_t TCount>
        bool inAny( std::span<const std::uint8_t> address, const std::array<Block<TBytes>, TCount>& blocks ) noexcept
        {
            return std::ranges::any_of( blocks, [address]( const Block<TBytes>& block )
                {
                    return hasPrefix( address, block.prefix, block.bits );
                } );
        }

        bool isPublicIpv4( std::span<const std::uint8_t> address ) noexcept
        {
            return !inAny( address, kRefusedIpv4 );
        }

        bool isPublicIpv6( std::span<const std::uint8_t> address ) noexcept
        {
            // An address that carries an IPv4 address reaches it, so the IPv4 address decides.
            if ( hasPrefix( address, kMapped.prefix, kMapped.bits ) || hasPrefix( address, kTranslated.prefix, kTranslated.bits ) )
            {
                return isPublicIpv4( address.subspan( 12, 4 ) );
            }

            if ( hasPrefix( address, kSixToFour.prefix, kSixToFour.bits ) )
            {
                return isPublicIpv4( address.subspan( 2, 4 ) );
            }

            return !inAny( address, kRefusedIpv6 );
        }

        std::string cidr( const Block<4>& block )
        {
            return std::format( "{}.{}.{}.{}/{}", block.prefix[ 0 ], block.prefix[ 1 ], block.prefix[ 2 ], block.prefix[ 3 ], block.bits );
        }

        std::string cidr( const Block<16>& block )
        {
            std::string text;

            for ( std::size_t group = 0; group < 8; ++group )
            {
                text += std::format( "{}{:x}", group == 0 ? "" : ":", ( block.prefix[ 2 * group ] << 8 ) | block.prefix[ 2 * group + 1 ] );
            }

            return std::format( "{}/{}", text, block.bits );
        }
    }

    /**
     * @brief Every address range a grant's network refuses, as CIDR text, for a filter that decides by range.
     *
     * Drawn from the table isPublicAddress decides by, so the two cannot drift. An IPv6 range that carries an IPv4
     * address -- IPv4-mapped, NAT64, 6to4 -- is refused whole here, which is stricter than isPublicAddress, which
     * decides by the IPv4 address inside (ContainedPrograms.md 4.5).
     */
    export std::vector<std::string> nonPublicRanges()
    {
        std::vector<std::string> ranges;

        for ( const auto& block : kRefusedIpv4 )
        {
            ranges.push_back( cidr( block ) );
        }

        for ( const auto& block : kRefusedIpv6 )
        {
            ranges.push_back( cidr( block ) );
        }

        ranges.push_back( cidr( kSixToFour ) );

        return ranges;
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
