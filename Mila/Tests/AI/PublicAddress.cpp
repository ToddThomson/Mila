/**
 * @file PublicAddress.cpp
 * @brief The addresses a grant's network may connect to: each block refused beside a public address just outside
 *        it, allowed (ToolCalling.md 3.4).
 */

#include <gtest/gtest.h>
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

    TEST( PublicAddress, ABufferThatIsNoAddressIsRefused )
    {
        const std::array<std::uint8_t, 5> five{ 8, 8, 8, 8, 8 };

        EXPECT_FALSE( isPublicAddress( {} ) );
        EXPECT_FALSE( isPublicAddress( five ) );
        EXPECT_TRUE( isPublicAddress( ipv4( 8, 8, 8, 8 ) ) );
    }
}
