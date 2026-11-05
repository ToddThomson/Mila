/**
 * @file WeightFormats.h
 * @brief Mila's weight formats on the host: encode normal values as Lib/Src/Dnn/Quantization/Weight lays them out, and decode them.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace WeightFormats
{
    enum class Kind { None, Fp8, Fp4, Int4, Int6, Codebook2, Codebook3 };

    struct Format
    {
        Kind kind{ Kind::None };
        int group{ 0 };
        std::string name{ "none" };

        /// Elements that share one scale. FP8's scale covers a whole row.
        std::uint32_t groupOf( std::uint32_t inputs ) const
        {
            return kind == Kind::Fp8 || kind == Kind::None ? inputs : static_cast<std::uint32_t>( group );
        }

        bool floatScales() const
        {
            return kind == Kind::Fp8 || kind == Kind::Fp4;
        }
    };

    inline Format parseFormat( const std::string& name )
    {
        if ( name == "none" ) { return { Kind::None, 0, name }; }
        if ( name == "fp8" ) { return { Kind::Fp8, 0, name }; }
        if ( name == "fp4-128" ) { return { Kind::Fp4, 128, name }; }
        if ( name == "fp4-64" ) { return { Kind::Fp4, 64, name }; }
        if ( name == "q4" ) { return { Kind::Int4, 32, name }; }
        if ( name == "q6" ) { return { Kind::Int6, 32, name }; }
        if ( name == "cb2-32" ) { return { Kind::Codebook2, 32, name }; }
        if ( name == "cb2-64" ) { return { Kind::Codebook2, 64, name }; }
        if ( name == "cb3-64" ) { return { Kind::Codebook3, 64, name }; }
        if ( name == "cb3-128" ) { return { Kind::Codebook3, 128, name }; }

        throw std::invalid_argument( "unknown format " + name );
    }

    /// The bytes of one format's weights, ready to copy to the GPU.
    struct Encoded
    {
        std::vector<std::uint8_t> codes;
        std::vector<std::uint8_t> high_bits;
        std::vector<float> float_scales;
        std::vector<std::uint16_t> half_scales;
        std::vector<float> table;
        std::size_t row_bytes{ 0 };
        std::size_t high_row_bytes{ 0 };
        std::uint32_t groups{ 0 };
    };

    inline std::uint16_t toHalfBits( float value )
    {
        const __fp16 half = static_cast<__fp16>( value );
        std::uint16_t bits = 0;
        std::memcpy( &bits, &half, sizeof( bits ) );

        return bits;
    }

    inline float fromHalfBits( std::uint16_t bits )
    {
        __fp16 half;
        std::memcpy( &half, &bits, sizeof( bits ) );

        return half;
    }

    /// Sign-magnitude nibble: bit 3 is the sign, and bits 0-2 index {0, 0.5, 1, 1.5, 2, 3, 4, 6}.
    inline std::vector<float> fp4Table()
    {
        const float magnitudes[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
        std::vector<float> table( 16 );

        for ( int i = 0; i < 8; ++i )
        {
            table[i] = magnitudes[i];
            table[8 + i] = -magnitudes[i];
        }

        return table;
    }

    /// FP8 E4M3: 3 mantissa bits, bias 7, largest 448, and the top code of each sign is not a number.
    inline std::vector<float> fp8Table()
    {
        std::vector<float> table( 256, 0.0f );

        for ( int code = 0; code < 128; ++code )
        {
            const int exponent = code >> 3;
            const int mantissa = code & 7;
            const float value = exponent == 0
                ? std::ldexp( static_cast<float>( mantissa ), -9 )
                : std::ldexp( 1.0f + mantissa / 8.0f, exponent - 7 );

            table[code] = code == 127 ? 0.0f : value;
            table[128 + code] = -table[code];
        }

        return table;
    }

    inline int nearest( const std::vector<float>& table, int count, float value )
    {
        int best = 0;
        float distance = std::numeric_limits<float>::max();

        for ( int i = 0; i < count; ++i )
        {
            const float d = std::fabs( table[i] - value );

            if ( d < distance )
            {
                distance = d;
                best = i;
            }
        }

        return best;
    }

    inline std::vector<float> sourceTable()
    {
        std::mt19937_64 generator( 12345 );
        std::normal_distribution<float> distribution( 0.0f, 1.0f );
        std::vector<float> table( std::size_t{ 1 } << 20 );

        for ( float& value : table )
        {
            value = distribution( generator );
        }

        return table;
    }

    /// Encode rows x columns of normal values with this sigma, each format by its own rule.
    inline Encoded encode( const Format& format, std::uint32_t rows, std::uint32_t columns, float sigma, std::uint64_t seed )
    {
        static const std::vector<float> normal = sourceTable();

        Encoded out;
        const std::uint32_t group = format.groupOf( columns );
        out.groups = columns / group;

        switch ( format.kind )
        {
            case Kind::Fp8:
                out.row_bytes = columns;
                out.table = fp8Table();
                out.float_scales.resize( rows );
                break;
            case Kind::Fp4:
                out.row_bytes = columns / 2;
                out.table = fp4Table();
                out.float_scales.resize( std::size_t{ rows } * out.groups );
                break;
            case Kind::Int4:
                out.row_bytes = columns / 2;
                out.half_scales.resize( std::size_t{ rows } * out.groups );
                break;
            case Kind::Int6:
                out.row_bytes = columns / 2 + columns / 4;
                out.half_scales.resize( std::size_t{ rows } * out.groups );
                break;
            case Kind::Codebook2:
                out.row_bytes = columns / 4;
                out.table = { -1.0f, -0.35f, 0.35f, 1.0f };
                out.half_scales.resize( std::size_t{ rows } * out.groups );
                break;
            case Kind::Codebook3:
                out.row_bytes = columns / 4;
                out.high_row_bytes = columns / 8;
                out.table = { -1.0f, -0.6f, -0.3f, -0.1f, 0.1f, 0.3f, 0.6f, 1.0f };
                out.high_bits.assign( std::size_t{ rows } * out.high_row_bytes, 0 );
                out.half_scales.resize( std::size_t{ rows } * out.groups );
                break;
            default:
                throw std::invalid_argument( "nothing to encode" );
        }

        out.codes.assign( std::size_t{ rows } * out.row_bytes, 0 );

        const auto encodeRows = [&]( std::uint32_t first, std::uint32_t last, std::uint64_t worker_seed )
        {
            std::uint64_t x = worker_seed * 2685821657736338717ULL + 88172645463325252ULL;
            std::vector<float> values( group );

            for ( std::uint32_t row = first; row < last; ++row )
            {
                std::uint8_t* const codes = out.codes.data() + std::size_t{ row } * out.row_bytes;
                std::uint8_t* const high = format.kind == Kind::Codebook3 ? out.high_bits.data() + std::size_t{ row } * out.high_row_bytes : nullptr;

                for ( std::uint32_t g = 0; g < out.groups; ++g )
                {
                    float extreme = 0.0f;
                    float magnitude = 0.0f;

                    for ( std::uint32_t i = 0; i < group; ++i )
                    {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        values[i] = normal[x >> 44] * sigma;

                        if ( magnitude < std::fabs( values[i] ) )
                        {
                            magnitude = std::fabs( values[i] );
                            extreme = values[i];
                        }
                    }

                    const std::size_t slot = std::size_t{ row } * out.groups + g;

                    if ( format.kind == Kind::Fp8 ) { out.float_scales[row] = magnitude / 448.0f; }
                    else if ( format.kind == Kind::Fp4 ) { out.float_scales[slot] = magnitude / 6.0f; }
                    else if ( format.kind == Kind::Int4 ) { out.half_scales[slot] = toHalfBits( extreme / -8.0f ); }
                    else if ( format.kind == Kind::Int6 ) { out.half_scales[slot] = toHalfBits( extreme / -32.0f ); }
                    else { out.half_scales[slot] = toHalfBits( magnitude ); }

                    const float inverse = format.kind == Kind::Int4 ? ( extreme != 0.0f ? -8.0f / extreme : 0.0f )
                        : format.kind == Kind::Int6 ? ( extreme != 0.0f ? -32.0f / extreme : 0.0f )
                        : format.kind == Kind::Fp8 ? ( magnitude != 0.0f ? 448.0f / magnitude : 0.0f )
                        : format.kind == Kind::Fp4 ? ( magnitude != 0.0f ? 6.0f / magnitude : 0.0f )
                        : ( magnitude != 0.0f ? 1.0f / magnitude : 0.0f );

                    for ( std::uint32_t i = 0; i < group; ++i )
                    {
                        const std::uint32_t j = g * group + i;
                        const float scaled = values[i] * inverse;

                        switch ( format.kind )
                        {
                            case Kind::Fp8:
                                codes[j] = static_cast<std::uint8_t>( nearest( out.table, 256, scaled ) );
                                break;
                            case Kind::Fp4:
                                codes[j >> 1] |= static_cast<std::uint8_t>( nearest( out.table, 16, scaled ) << ( ( j & 1 ) * 4 ) );
                                break;
                            case Kind::Int4:
                            {
                                const int code = std::min( 15, static_cast<int>( scaled + 8.5f ) );
                                codes[j >> 1] |= static_cast<std::uint8_t>( code << ( ( j & 1 ) * 4 ) );
                                break;
                            }
                            case Kind::Int6:
                            {
                                const int code = std::min( 63, static_cast<int>( scaled + 32.5f ) );
                                codes[j >> 1] |= static_cast<std::uint8_t>( ( code & 15 ) << ( ( j & 1 ) * 4 ) );
                                codes[columns / 2 + ( j >> 2 )] |= static_cast<std::uint8_t>( ( code >> 4 ) << ( ( j & 3 ) * 2 ) );
                                break;
                            }
                            case Kind::Codebook2:
                                codes[j >> 2] |= static_cast<std::uint8_t>( nearest( out.table, 4, scaled ) << ( ( j & 3 ) * 2 ) );
                                break;
                            default:
                            {
                                const int code = nearest( out.table, 8, scaled );
                                codes[j >> 2] |= static_cast<std::uint8_t>( ( code & 3 ) << ( ( j & 3 ) * 2 ) );
                                high[j >> 3] |= static_cast<std::uint8_t>( ( code >> 2 ) << ( j & 7 ) );
                                break;
                            }
                        }
                    }
                }
            }
        };

        std::vector<std::thread> workers;
        const std::uint32_t block = rows / 8 + 1;

        for ( std::uint32_t worker = 0; worker < 8; ++worker )
        {
            workers.emplace_back( encodeRows, block * worker, std::min( rows, block * ( worker + 1 ) ), seed + worker );
        }

        for ( std::thread& worker : workers )
        {
            worker.join();
        }

        return out;
    }

    /// One weight's value, read back the way a kernel must.
    inline float decode( const Format& format, const Encoded& encoded, std::uint32_t row, std::uint32_t column, std::uint32_t columns )
    {
        const std::uint8_t* const codes = encoded.codes.data() + std::size_t{ row } * encoded.row_bytes;
        const std::uint32_t g = column / format.groupOf( columns );
        const std::size_t slot = std::size_t{ row } * encoded.groups + g;

        switch ( format.kind )
        {
            case Kind::Fp8:
                return encoded.table[codes[column]] * encoded.float_scales[row];
            case Kind::Fp4:
                return encoded.table[( codes[column >> 1] >> ( ( column & 1 ) * 4 ) ) & 15] * encoded.float_scales[slot];
            case Kind::Int4:
                return ( static_cast<float>( ( codes[column >> 1] >> ( ( column & 1 ) * 4 ) ) & 15 ) - 8.0f ) * fromHalfBits( encoded.half_scales[slot] );
            case Kind::Int6:
            {
                const int low = ( codes[column >> 1] >> ( ( column & 1 ) * 4 ) ) & 15;
                const int high = ( codes[columns / 2 + ( column >> 2 )] >> ( ( column & 3 ) * 2 ) ) & 3;

                return ( static_cast<float>( low | ( high << 4 ) ) - 32.0f ) * fromHalfBits( encoded.half_scales[slot] );
            }
            case Kind::Codebook2:
                return encoded.table[( codes[column >> 2] >> ( ( column & 3 ) * 2 ) ) & 3] * fromHalfBits( encoded.half_scales[slot] );
            case Kind::Codebook3:
            {
                const int low = ( codes[column >> 2] >> ( ( column & 3 ) * 2 ) ) & 3;
                const int high = ( encoded.high_bits[std::size_t{ row } * encoded.high_row_bytes + ( column >> 3 )] >> ( column & 7 ) ) & 1;

                return encoded.table[low | ( high << 2 )] * fromHalfBits( encoded.half_scales[slot] );
            }
            default:
                throw std::invalid_argument( "nothing to decode" );
        }
    }
}
