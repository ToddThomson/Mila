/**
 * @file CodebookPacking.ixx
 * @brief Normative packed layout and CPU reference codec for the codebook policies.
 * The Python packer (converter) and the CUDA kernels must both match this file exactly;
 * a disagreement between them is resolved in favor of this file.
 */

module;
#include <bit>
#include <cstdint>

export module Experimental.Quantization.CodebookPacking;

import Dnn.TensorTypes;

namespace Mila::Dnn::Experimental::Quantization
{
    // -------------------------------------------------------------------------
    // Packed layout contract
    //
    // A quantized tensor of [rows, columns] (rows = output features, columns =
    // input features) is stored as:
    //
    //   codes    2-bit: one buffer, row-major; code j of a row lives in byte
    //            j / 4 at bit offset (j % 4) * 2.
    //            3-bit: two byte-aligned planes. The low two bits of each code
    //            pack exactly as the 2-bit layout; the high bit packs one bit
    //            per code, code j in byte j / 8 at bit offset j % 8.
    //   scales   IEEE half bits (uint16_t), row-major [row][group], where a row
    //            has ceil(columns / kGroupSize) groups. scale = group absmax.
    //   codebook FP32, kCodebookEntries values, indexed directly by code. The
    //            converter emits entries sorted ascending; consumers rely only
    //            on index -> value.
    //
    //   dequantized[row][column] =
    //       codebook[code] * half(scales[row][column / kGroupSize])
    // -------------------------------------------------------------------------

    export constexpr dim_t packedRowBytesForTwoBitCodes( dim_t columns ) noexcept
    {
        return ( columns + 3 ) / 4;
    }

    export constexpr dim_t packedRowBytesForOneBitPlane( dim_t columns ) noexcept
    {
        return ( columns + 7 ) / 8;
    }

    export constexpr dim_t groupsPerRow( dim_t columns, dim_t groupSize ) noexcept
    {
        return ( columns + groupSize - 1 ) / groupSize;
    }

    // -------------------------------------------------------------------------
    // IEEE half conversions (host side). Round-to-nearest-even, subnormals
    // preserved: absmax scales of near-zero groups can land below 2^-14.
    // -------------------------------------------------------------------------

    export inline std::uint16_t floatToHalfBits( float value ) noexcept
    {
        const std::uint32_t bits = std::bit_cast<std::uint32_t>( value );
        const std::uint32_t sign = ( bits >> 16 ) & 0x8000u;
        const std::uint32_t magnitude = bits & 0x7FFFFFFFu;

        if ( magnitude >= 0x7F800000u )
            return static_cast<std::uint16_t>(
                sign | 0x7C00u | ( magnitude > 0x7F800000u ? 0x0200u : 0u ) );

        if ( magnitude >= 0x47800000u )
            return static_cast<std::uint16_t>( sign | 0x7C00u );

        if ( magnitude < 0x33000000u )
            return static_cast<std::uint16_t>( sign );

        if ( magnitude < 0x38800000u ) {
            const std::uint32_t mantissa = ( magnitude & 0x007FFFFFu ) | 0x00800000u;
            const int shift = 126 - static_cast<int>( magnitude >> 23 );
            const std::uint32_t rounded =
                ( mantissa + ( 1u << ( shift - 1 ) ) - 1u + ( ( mantissa >> shift ) & 1u ) ) >> shift;

            return static_cast<std::uint16_t>( sign | rounded );
        }

        std::uint32_t half = sign | ( ( ( magnitude >> 23 ) - 112u ) << 10 )
            | ( ( magnitude >> 13 ) & 0x3FFu );
        const std::uint32_t remainder = magnitude & 0x1FFFu;

        if ( remainder > 0x1000u || ( remainder == 0x1000u && ( half & 1u ) ) )
            ++half;

        return static_cast<std::uint16_t>( half );
    }

    export inline float halfBitsToFloat( std::uint16_t half ) noexcept
    {
        const std::uint32_t sign = ( static_cast<std::uint32_t>( half ) & 0x8000u ) << 16;
        const std::uint32_t exponent = ( half >> 10 ) & 0x1Fu;
        const std::uint32_t mantissa = half & 0x3FFu;

        if ( exponent == 0x1Fu )
            return std::bit_cast<float>( sign | 0x7F800000u | ( mantissa << 13 ) );

        if ( exponent == 0 ) {
            if ( mantissa == 0 )
                return std::bit_cast<float>( sign );

            std::uint32_t normalized = mantissa;
            std::uint32_t deficit = 0;

            while ( ( normalized & 0x400u ) == 0 ) {
                normalized <<= 1;
                ++deficit;
            }

            return std::bit_cast<float>(
                sign | ( ( 113u - deficit ) << 23 ) | ( ( normalized & 0x3FFu ) << 13 ) );
        }

        return std::bit_cast<float>( sign | ( ( exponent + 112u ) << 23 ) | ( mantissa << 13 ) );
    }

    // -------------------------------------------------------------------------
    // Code packing. Callers pass one unpacked code per byte; packed buffers are
    // sized by the row-bytes helpers above and are fully overwritten.
    // -------------------------------------------------------------------------

    export inline void packTwoBitCodes(
        const std::uint8_t* codes, dim_t rows, dim_t columns, std::uint8_t* packed ) noexcept
    {
        const std::size_t rowBytes = static_cast<std::size_t>( packedRowBytesForTwoBitCodes( columns ) );

        for ( dim_t row = 0; row < rows; ++row ) {
            std::uint8_t* packedRow = packed + static_cast<std::size_t>( row ) * rowBytes;

            for ( std::size_t byte = 0; byte < rowBytes; ++byte )
                packedRow[byte] = 0;

            for ( dim_t column = 0; column < columns; ++column ) {
                const std::uint8_t code = codes[row * columns + column] & 0x3u;
                packedRow[column >> 2] |= static_cast<std::uint8_t>( code << ( ( column & 3 ) * 2 ) );
            }
        }
    }

    export inline void unpackTwoBitCodes(
        const std::uint8_t* packed, dim_t rows, dim_t columns, std::uint8_t* codes ) noexcept
    {
        const std::size_t rowBytes = static_cast<std::size_t>( packedRowBytesForTwoBitCodes( columns ) );

        for ( dim_t row = 0; row < rows; ++row ) {
            const std::uint8_t* packedRow = packed + static_cast<std::size_t>( row ) * rowBytes;

            for ( dim_t column = 0; column < columns; ++column )
                codes[row * columns + column] =
                    static_cast<std::uint8_t>( ( packedRow[column >> 2] >> ( ( column & 3 ) * 2 ) ) & 0x3u );
        }
    }

    export inline void packThreeBitCodes(
        const std::uint8_t* codes, dim_t rows, dim_t columns,
        std::uint8_t* planeTwoBits, std::uint8_t* planeOneBit ) noexcept
    {
        packTwoBitCodes( codes, rows, columns, planeTwoBits );

        const std::size_t planeRowBytes = static_cast<std::size_t>( packedRowBytesForOneBitPlane( columns ) );

        for ( dim_t row = 0; row < rows; ++row ) {
            std::uint8_t* planeRow = planeOneBit + static_cast<std::size_t>( row ) * planeRowBytes;

            for ( std::size_t byte = 0; byte < planeRowBytes; ++byte )
                planeRow[byte] = 0;

            for ( dim_t column = 0; column < columns; ++column ) {
                const std::uint8_t highBit = ( codes[row * columns + column] >> 2 ) & 0x1u;
                planeRow[column >> 3] |= static_cast<std::uint8_t>( highBit << ( column & 7 ) );
            }
        }
    }

    export inline void unpackThreeBitCodes(
        const std::uint8_t* planeTwoBits, const std::uint8_t* planeOneBit,
        dim_t rows, dim_t columns, std::uint8_t* codes ) noexcept
    {
        unpackTwoBitCodes( planeTwoBits, rows, columns, codes );

        const std::size_t planeRowBytes = static_cast<std::size_t>( packedRowBytesForOneBitPlane( columns ) );

        for ( dim_t row = 0; row < rows; ++row ) {
            const std::uint8_t* planeRow = planeOneBit + static_cast<std::size_t>( row ) * planeRowBytes;

            for ( dim_t column = 0; column < columns; ++column ) {
                const std::uint8_t highBit =
                    static_cast<std::uint8_t>( ( planeRow[column >> 3] >> ( column & 7 ) ) & 0x1u );
                codes[row * columns + column] |= static_cast<std::uint8_t>( highBit << 2 );
            }
        }
    }

    // -------------------------------------------------------------------------
    // Reference encode and dequantize. Encode exists for the converter and for
    // tests -- inference never encodes. Tie-break contract: equidistant values
    // take the lower index; the Python packer must conform.
    // -------------------------------------------------------------------------

    export inline std::uint8_t encodeNearestCode(
        float normalizedValue, const float* codebook, int entryCount ) noexcept
    {
        std::uint8_t best = 0;
        float bestDistance = normalizedValue - codebook[0];
        bestDistance = bestDistance < 0.0f ? -bestDistance : bestDistance;

        for ( int entry = 1; entry < entryCount; ++entry ) {
            float distance = normalizedValue - codebook[entry];
            distance = distance < 0.0f ? -distance : distance;

            if ( distance < bestDistance ) {
                bestDistance = distance;
                best = static_cast<std::uint8_t>( entry );
            }
        }

        return best;
    }

    export inline void dequantizeCodes(
        const std::uint8_t* codes, const std::uint16_t* scaleBits, const float* codebook,
        dim_t rows, dim_t columns, dim_t groupSize, float* output ) noexcept
    {
        const dim_t rowGroups = groupsPerRow( columns, groupSize );

        for ( dim_t row = 0; row < rows; ++row ) {
            for ( dim_t column = 0; column < columns; ++column ) {
                const float scale = halfBitsToFloat( scaleBits[row * rowGroups + column / groupSize] );
                output[row * columns + column] = codebook[codes[row * columns + column]] * scale;
            }
        }
    }
}
