/**
 * @file PublicAddress.cpp
 * @brief The addresses a grant's network may connect to: each block refused beside a public address just outside
 *        it, allowed (ToolCalling.md 3.4).
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

import Mila;

namespace Mila::Tests::AI
{
    using namespace Mila::AI;

    namespace
    {
        std::array<std::uint8_t, 4> ipv4( std::uint8_t a, std::uint8_t b, std::uint8_t c, std::uint8_t d )
        {
            return { a, b, c, d };
        }

        /// An IPv6 address from its eight 16-bit groups.
        std::array<std::uint8_t, 16> ipv6( std::array<std::uint16_t, 8> groups )
        {
            std::array<std::uint8_t, 16> bytes{};

            for ( std::size_t index = 0; index < groups.size(); ++index )
            {
                bytes[ 2 * index ] = static_cast<std::uint8_t>( groups[ index ] >> 8 );
                bytes[ 2 * index + 1 ] = static_cast<std::uint8_t>( groups[ index ] & 0xff );
            }

            return bytes;
        }

        struct Pair
        {
            std::string block;
            std::vector<std::uint8_t> refused;
            std::vector<std::uint8_t> allowed;
        };

        template <std::size_t N>
        std::vector<std::uint8_t> bytes( const std::array<std::uint8_t, N>& address )
        {
            return { address.begin(), address.end() };
        }
    }

    TEST( PublicAddress, EachBlockThatIsNotPublicIsRefusedBesideAPublicNeighbour )
    {
        const std::vector<Pair> pairs{
            { "10.0.0.0/8", bytes( ipv4( 10, 0, 0, 1 ) ), bytes( ipv4( 11, 0, 0, 1 ) ) },
            { "172.16.0.0/12 (first)", bytes( ipv4( 172, 16, 0, 1 ) ), bytes( ipv4( 172, 15, 255, 254 ) ) },
            { "172.16.0.0/12 (last)", bytes( ipv4( 172, 31, 255, 254 ) ), bytes( ipv4( 172, 32, 0, 1 ) ) },
            { "192.168.0.0/16", bytes( ipv4( 192, 168, 1, 1 ) ), bytes( ipv4( 192, 169, 0, 1 ) ) },
            { "100.64.0.0/10", bytes( ipv4( 100, 127, 255, 254 ) ), bytes( ipv4( 100, 128, 0, 1 ) ) },
            { "127.0.0.0/8", bytes( ipv4( 127, 0, 0, 1 ) ), bytes( ipv4( 128, 0, 0, 1 ) ) },
            { "169.254.0.0/16", bytes( ipv4( 169, 254, 169, 254 ) ), bytes( ipv4( 169, 255, 0, 1 ) ) },
            { "0.0.0.0/8", bytes( ipv4( 0, 0, 0, 0 ) ), bytes( ipv4( 1, 1, 1, 1 ) ) },
            { "192.0.2.0/24", bytes( ipv4( 192, 0, 2, 1 ) ), bytes( ipv4( 192, 0, 3, 1 ) ) },
            { "198.18.0.0/15", bytes( ipv4( 198, 19, 0, 1 ) ), bytes( ipv4( 198, 20, 0, 1 ) ) },
            { "224.0.0.0/4", bytes( ipv4( 239, 255, 255, 250 ) ), bytes( ipv4( 223, 255, 255, 254 ) ) },
            { "240.0.0.0/4", bytes( ipv4( 255, 255, 255, 255 ) ), bytes( ipv4( 8, 8, 8, 8 ) ) },
            { "::1", bytes( ipv6( { 0, 0, 0, 0, 0, 0, 0, 1 } ) ), bytes( ipv6( { 0x2606, 0x4700, 0, 0, 0, 0, 0, 0x1111 } ) ) },
            { "::", bytes( ipv6( { 0, 0, 0, 0, 0, 0, 0, 0 } ) ), bytes( ipv6( { 0x2001, 0x4860, 0, 0, 0, 0, 0, 0x8888 } ) ) },
            { "fc00::/7", bytes( ipv6( { 0xfd12, 0x3456, 0, 0, 0, 0, 0, 1 } ) ), bytes( ipv6( { 0x2a00, 0x1450, 0, 0, 0, 0, 0, 1 } ) ) },
            { "fe80::/10", bytes( ipv6( { 0xfe80, 0, 0, 0, 0, 0, 0, 1 } ) ), bytes( ipv6( { 0x2a03, 0x2880, 0, 0, 0, 0, 0, 1 } ) ) },
            { "ff00::/8", bytes( ipv6( { 0xff02, 0, 0, 0, 0, 0, 0, 1 } ) ), bytes( ipv6( { 0x2400, 0xcb00, 0, 0, 0, 0, 0, 1 } ) ) },
            { "2001:db8::/32", bytes( ipv6( { 0x2001, 0x0db8, 0, 0, 0, 0, 0, 1 } ) ), bytes( ipv6( { 0x2001, 0x0db9, 0, 0, 0, 0, 0, 1 } ) ) },
            { "::ffff:0:0/96", bytes( ipv6( { 0, 0, 0, 0, 0, 0xffff, 0x0a00, 0x0001 } ) ), bytes( ipv6( { 0, 0, 0, 0, 0, 0xffff, 0x0808, 0x0808 } ) ) },
            { "64:ff9b::/96", bytes( ipv6( { 0x64, 0xff9b, 0, 0, 0, 0, 0x7f00, 0x0001 } ) ), bytes( ipv6( { 0x64, 0xff9b, 0, 0, 0, 0, 0x0808, 0x0808 } ) ) },
            { "2002::/16", bytes( ipv6( { 0x2002, 0xc0a8, 0x0101, 0, 0, 0, 0, 1 } ) ), bytes( ipv6( { 0x2002, 0x0808, 0x0808, 0, 0, 0, 0, 1 } ) ) },
        };

        for ( const auto& pair : pairs )
        {
            EXPECT_FALSE( isPublicAddress( pair.refused ) ) << pair.block << " is not public";
            EXPECT_TRUE( isPublicAddress( pair.allowed ) ) << "the address beside " << pair.block << " is public";
        }
    }

    namespace
    {
        /// A range as nonPublicRanges writes it: dotted IPv4, or IPv6 as eight groups with no group left out.
        struct Range
        {
            std::vector<std::uint8_t> address;
            int bits;
        };

        Range parse( const std::string& text )
        {
            const auto slash = text.find( '/' );
            const std::string address = text.substr( 0, slash );
            Range range{ {}, std::stoi( text.substr( slash + 1 ) ) };
            const bool v6 = address.find( ':' ) != std::string::npos;
            std::size_t start = 0;

            for ( ;; )
            {
                const auto end = address.find( v6 ? ':' : '.', start );
                const auto value = std::stoul( address.substr( start, end - start ), nullptr, v6 ? 16 : 10 );

                if ( v6 )
                {
                    range.address.push_back( static_cast<std::uint8_t>( value >> 8 ) );
                }

                range.address.push_back( static_cast<std::uint8_t>( value & 0xff ) );

                if ( end == std::string::npos )
                {
                    break;
                }

                start = end + 1;
            }

            return range;
        }

        /// The range's first address, or its last.
        std::vector<std::uint8_t> edge( const Range& range, bool last )
        {
            std::vector<std::uint8_t> address = range.address;

            for ( int bit = range.bits; bit < static_cast<int>( address.size() * 8 ); ++bit )
            {
                const auto mask = static_cast<std::uint8_t>( 0x80u >> ( bit % 8 ) );
                address[ bit / 8 ] = static_cast<std::uint8_t>( last ? address[ bit / 8 ] | mask : address[ bit / 8 ] & ~mask );
            }

            return address;
        }

        bool inRange( const Range& range, std::span<const std::uint8_t> address )
        {
            if ( address.size() != range.address.size() )
            {
                return false;
            }

            for ( int bit = 0; bit < range.bits; ++bit )
            {
                const auto mask = static_cast<std::uint8_t>( 0x80u >> ( bit % 8 ) );

                if ( ( address[ bit / 8 ] & mask ) != ( range.address[ bit / 8 ] & mask ) )
                {
                    return false;
                }
            }

            return true;
        }
    }

    // The ranges a filter that decides by range is given (ContainedPrograms.md 4.5) come from the same table.
    TEST( PublicAddress, EveryRangeRefusedByRangeIsRefusedByAddressAndHoldsNoPublicAddress )
    {
        const std::vector<std::string> ranges = nonPublicRanges();

        ASSERT_FALSE( ranges.empty() );

        for ( const std::string& text : ranges )
        {
            const Range range = parse( text );

            EXPECT_FALSE( isPublicAddress( edge( range, false ) ) ) << "the first address of " << text;
            EXPECT_FALSE( isPublicAddress( edge( range, true ) ) ) << "the last address of " << text;
            EXPECT_FALSE( inRange( range, ipv4( 8, 8, 8, 8 ) ) ) << text << " holds a public IPv4 address";
            EXPECT_FALSE( inRange( range, ipv6( { 0x2606, 0x4700, 0, 0, 0, 0, 0, 0x1111 } ) ) ) << text << " holds a public IPv6 address";
        }

        // The forms that carry an IPv4 address are refused whole, whatever address they carry.
        for ( const std::string carrying : { "0:0:0:0:0:ffff:808:808", "64:ff9b:0:0:0:0:808:808", "2002:808:808:0:0:0:0:1" } )
        {
            const Range address = parse( carrying + "/128" );

            EXPECT_TRUE( std::ranges::any_of( ranges, [&]( const std::string& text ) { return inRange( parse( text ), address.address ); } ) )
                << carrying << " lies in no range refused";
        }
    }

    TEST( PublicAddress, ABufferThatIsNoAddressIsRefused )
    {
        const std::array<std::uint8_t, 5> five{ 8, 8, 8, 8, 8 };

        EXPECT_FALSE( isPublicAddress( {} ) );
        EXPECT_FALSE( isPublicAddress( five ) );
        EXPECT_TRUE( isPublicAddress( ipv4( 8, 8, 8, 8 ) ) );
    }
}
