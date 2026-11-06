/**
 * @file QuantizedReference.h
 * @brief FP8 and FP4 weights on the host: CUDA's quantize, step for step, and the value each code and scale stands for.
 *
 * Included after MetalReference.h, so it names its types and includes nothing.
 */

#pragma once

namespace Mila::Tests::Dnn::QuantizedReference
{
    using Mila::Tests::Dnn::MetalReference::fromBf16Bits;

    /// Codes and their scales, laid out as a Linear stores them.
    struct QuantizedWeights
    {
        std::vector<std::uint8_t> codes;
        std::vector<float> scales;
    };

    /// The E4M3 code nearest a value, ties to even, saturating at 448: CUDA's `__nv_fp8_e4m3` with saturation.
    inline std::uint8_t fp8E4M3Code( float value )
    {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>( value );
        const std::uint8_t sign = static_cast<std::uint8_t>( ( bits >> 24 ) & 0x80u );
        const float magnitude = std::bit_cast<float>( bits & 0x7FFFFFFFu );

        if ( magnitude > 448.0f )
        {
            return sign | 0x7Eu;
        }

        if ( magnitude < 0.015625f )
        {
            return sign | static_cast<std::uint8_t>( std::nearbyint( magnitude * 512.0f ) );
        }

        const std::uint32_t exponent = ( ( bits >> 23 ) & 0xFFu ) - 127u + 7u;
        const std::uint32_t mantissa = bits & 0x7FFFFFu;
        const std::uint32_t kept = mantissa >> 20;
        const std::uint32_t rest = mantissa & 0xFFFFFu;
        const std::uint32_t rounds_up = ( rest > 0x80000u || ( rest == 0x80000u && ( kept & 1u ) != 0u ) ) ? 1u : 0u;

        return sign | static_cast<std::uint8_t>( std::min( ( exponent << 3 ) + kept + rounds_up, 0x7Eu ) );
    }

    inline double fp8E4M3Value( std::uint8_t code )
    {
        const int magnitude_code = code & 0x7F;
        const int exponent = magnitude_code >> 3;
        const int mantissa = magnitude_code & 7;
        const double magnitude = magnitude_code == 0x7F ? 0.0
            : exponent == 0 ? std::ldexp( static_cast<double>( mantissa ), -9 )
            : std::ldexp( 1.0 + mantissa / 8.0, exponent - 7 );

        return ( code & 0x80 ) != 0 ? -magnitude : magnitude;
    }

    /// The E2M1 code nearest a value, at the midpoints CUDA's quantize uses.
    inline std::uint8_t fp4E2M1Code( float value )
    {
        const std::uint8_t sign = value < 0.0f ? 8 : 0;
        const float magnitude = std::fabs( value );
        constexpr float kMidpoints[] = { 0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f };

        std::uint8_t code = 7;

        for ( std::uint8_t i = 0; i < 7; ++i )
        {
            if ( magnitude < kMidpoints[i] )
            {
                code = i;
                break;
            }
        }

        return sign | code;
    }

    inline double fp4E2M1Value( std::uint8_t code )
    {
        constexpr double kMagnitudes[] = { 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 };

        return ( code & 8 ) != 0 ? -kMagnitudes[code & 7] : kMagnitudes[code & 7];
    }

    /// One scale per output, max |w| / 448, or 1 for a row of zeros.
    inline QuantizedWeights quantizeFp8( std::span<const std::uint16_t> bf16, std::size_t outputs, std::size_t inputs )
    {
        QuantizedWeights result{ std::vector<std::uint8_t>( outputs * inputs ), std::vector<float>( outputs ) };

        for ( std::size_t o = 0; o < outputs; ++o )
        {
            float largest = 0.0f;

            for ( std::size_t i = 0; i < inputs; ++i )
            {
                largest = std::fmax( largest, std::fabs( fromBf16Bits( bf16[o * inputs + i] ) ) );
            }

            const float scale = largest > 0.0f ? largest / 448.0f : 1.0f;
            const float inverse = 1.0f / scale;

            result.scales[o] = scale;

            for ( std::size_t i = 0; i < inputs; ++i )
            {
                result.codes[o * inputs + i] = fp8E4M3Code( fromBf16Bits( bf16[o * inputs + i] ) * inverse );
            }
        }

        return result;
    }

    /// One scale per group of `group` inputs, max |w| / 6, or 1 for a group of zeros; two codes to a byte, the even input low.
    inline QuantizedWeights quantizeFp4( std::span<const std::uint16_t> bf16, std::size_t outputs, std::size_t inputs, std::size_t group )
    {
        const std::size_t groups = inputs / group;
        QuantizedWeights result{ std::vector<std::uint8_t>( outputs * inputs / 2, 0 ), std::vector<float>( outputs * groups ) };

        for ( std::size_t o = 0; o < outputs; ++o )
        {
            for ( std::size_t g = 0; g < groups; ++g )
            {
                float largest = 0.0f;

                for ( std::size_t i = g * group; i < ( g + 1 ) * group; ++i )
                {
                    largest = std::fmax( largest, std::fabs( fromBf16Bits( bf16[o * inputs + i] ) ) );
                }

                const float scale = largest > 0.0f ? largest / 6.0f : 1.0f;
                const float inverse = 1.0f / scale;

                result.scales[o * groups + g] = scale;

                for ( std::size_t i = g * group; i < ( g + 1 ) * group; ++i )
                {
                    const std::uint8_t code = fp4E2M1Code( fromBf16Bits( bf16[o * inputs + i] ) * inverse );
                    result.codes[( o * inputs + i ) / 2] |= static_cast<std::uint8_t>( code << ( ( i & 1 ) * 4 ) );
                }
            }
        }

        return result;
    }

    /// Every weight's value, as the codes and scales hold it.
    inline std::vector<double> decodeFp8( const QuantizedWeights& weights, std::size_t outputs, std::size_t inputs )
    {
        std::vector<double> values( outputs * inputs );

        for ( std::size_t o = 0; o < outputs; ++o )
        {
            for ( std::size_t i = 0; i < inputs; ++i )
            {
                values[o * inputs + i] = fp8E4M3Value( weights.codes[o * inputs + i] ) * weights.scales[o];
            }
        }

        return values;
    }

    inline std::vector<double> decodeFp4( const QuantizedWeights& weights, std::size_t outputs, std::size_t inputs, std::size_t group )
    {
        std::vector<double> values( outputs * inputs );

        for ( std::size_t o = 0; o < outputs; ++o )
        {
            for ( std::size_t i = 0; i < inputs; ++i )
            {
                const std::uint8_t code = ( weights.codes[( o * inputs + i ) / 2] >> ( ( i & 1 ) * 4 ) ) & 15;

                values[o * inputs + i] = fp4E2M1Value( code ) * weights.scales[o * ( inputs / group ) + i / group];
            }
        }

        return values;
    }
}
