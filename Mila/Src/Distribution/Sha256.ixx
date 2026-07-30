/**
 * @file Sha256.ixx
 * @brief Streaming SHA-256, for verifying downloaded model blobs against their digests.
 *
 * Implemented rather than taken from a dependency: it is roughly eighty lines, the algorithm
 * is fixed, and the alternatives are a platform split (BCrypt / OpenSSL EVP) or a second
 * vendored library in a project whose whole third-party surface is a handful of packages.
 * Correctness is pinned by the NIST vectors in the tests, not assumed.
 */

module;
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstddef>
#include <string>

export module Distribution.Sha256;

namespace Mila::Distribution
{
    namespace
    {
        constexpr std::array<uint32_t, 64> kRoundConstants = { {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
        } };

        constexpr uint32_t rotateRight( uint32_t value, int bits )
        {
            return ( value >> bits ) | ( value << ( 32 - bits ) );
        }
    }

    /**
     * @brief Incremental SHA-256. Feed bytes with update(), read the digest with finish().
     *
     * Incremental by requirement, not convenience: a 6.33 GB artifact is hashed as it arrives
     * from the network so the bytes are touched once. A verification pass over the finished
     * file would double the I/O for no extra confidence.
     */
    export class Sha256
    {
    public:

        void update( const void* data, size_t length )
        {
            const auto* bytes = static_cast<const uint8_t*>( data );

            total_bits_ += static_cast<uint64_t>( length ) * 8;

            while ( length > 0 )
            {
                const size_t take = std::min( length, sizeof( buffer_ ) - buffer_length_ );

                for ( size_t index = 0; index < take; ++index )
                {
                    buffer_[ buffer_length_ + index ] = bytes[ index ];
                }

                buffer_length_ += take;
                bytes += take;
                length -= take;

                if ( buffer_length_ == sizeof( buffer_ ) )
                {
                    compress( buffer_ );
                    buffer_length_ = 0;
                }
            }
        }

        /**
         * @brief Finalize and return the digest as lowercase hex.
         *
         * Consumes the state; a second call returns the digest of the padded stream and is a
         * caller error.
         */
        std::string finish()
        {
            const uint64_t bit_length = total_bits_;

            // Padding: a 0x80 byte, zeros, then the length as a big-endian 64-bit count.
            const uint8_t one = 0x80;
            update( &one, 1 );
            total_bits_ = bit_length;   // update() must not count the padding.

            const uint8_t zero = 0x00;

            while ( buffer_length_ != 56 )
            {
                update( &zero, 1 );
                total_bits_ = bit_length;
            }

            uint8_t length_bytes[ 8 ];

            for ( int index = 0; index < 8; ++index )
            {
                length_bytes[ index ] = static_cast<uint8_t>( bit_length >> ( 56 - 8 * index ) );
            }

            update( length_bytes, 8 );

            static constexpr char kHexDigits[] = "0123456789abcdef";
            std::string hex;
            hex.reserve( 64 );

            for ( uint32_t word : state_ )
            {
                for ( int shift = 28; shift >= 0; shift -= 4 )
                {
                    hex.push_back( kHexDigits[ ( word >> shift ) & 0xF ] );
                }
            }

            return hex;
        }

    private:

        void compress( const uint8_t* block )
        {
            uint32_t schedule[ 64 ];

            for ( int index = 0; index < 16; ++index )
            {
                schedule[ index ] =
                    ( static_cast<uint32_t>( block[ index * 4 + 0 ] ) << 24 ) |
                    ( static_cast<uint32_t>( block[ index * 4 + 1 ] ) << 16 ) |
                    ( static_cast<uint32_t>( block[ index * 4 + 2 ] ) << 8 ) |
                    ( static_cast<uint32_t>( block[ index * 4 + 3 ] ) );
            }

            for ( int index = 16; index < 64; ++index )
            {
                const uint32_t s0 = rotateRight( schedule[ index - 15 ], 7 )
                    ^ rotateRight( schedule[ index - 15 ], 18 )
                    ^ ( schedule[ index - 15 ] >> 3 );

                const uint32_t s1 = rotateRight( schedule[ index - 2 ], 17 )
                    ^ rotateRight( schedule[ index - 2 ], 19 )
                    ^ ( schedule[ index - 2 ] >> 10 );

                schedule[ index ] = schedule[ index - 16 ] + s0 + schedule[ index - 7 ] + s1;
            }

            uint32_t a = state_[ 0 ];
            uint32_t b = state_[ 1 ];
            uint32_t c = state_[ 2 ];
            uint32_t d = state_[ 3 ];
            uint32_t e = state_[ 4 ];
            uint32_t f = state_[ 5 ];
            uint32_t g = state_[ 6 ];
            uint32_t h = state_[ 7 ];

            for ( int index = 0; index < 64; ++index )
            {
                const uint32_t big_sigma1 =
                    rotateRight( e, 6 ) ^ rotateRight( e, 11 ) ^ rotateRight( e, 25 );
                const uint32_t choose = ( e & f ) ^ ( ~e & g );
                const uint32_t temp1 = h + big_sigma1 + choose + kRoundConstants[ index ] + schedule[ index ];

                const uint32_t big_sigma0 =
                    rotateRight( a, 2 ) ^ rotateRight( a, 13 ) ^ rotateRight( a, 22 );
                const uint32_t majority = ( a & b ) ^ ( a & c ) ^ ( b & c );
                const uint32_t temp2 = big_sigma0 + majority;

                h = g;
                g = f;
                f = e;
                e = d + temp1;
                d = c;
                c = b;
                b = a;
                a = temp1 + temp2;
            }

            state_[ 0 ] += a;
            state_[ 1 ] += b;
            state_[ 2 ] += c;
            state_[ 3 ] += d;
            state_[ 4 ] += e;
            state_[ 5 ] += f;
            state_[ 6 ] += g;
            state_[ 7 ] += h;
        }

        std::array<uint32_t, 8> state_ = { {
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
        } };

        uint8_t buffer_[ 64 ]{};
        size_t buffer_length_{ 0 };
        uint64_t total_bits_{ 0 };
    };

    /**
     * @brief One-shot convenience for small inputs.
     */
    export inline std::string sha256Hex( const void* data, size_t length )
    {
        Sha256 hash;
        hash.update( data, length );

        return hash.finish();
    }
}
