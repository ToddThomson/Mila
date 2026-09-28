/**
 * @file Int4Packing.ixx
 * @brief Normative packed layout and CPU reference codec for the PerGroupInt4 policy.
 * The CUDA quantizer and kernels must match this file exactly; a disagreement is resolved in its favor.
 */

module;
#include <algorithm>
#include <cmath>
#include <cstdint>

export module Dnn.Quantization.Weight.Int4Packing;

import Dnn.TensorTypes;
import Dnn.Quantization.Weight.CodebookPacking;

namespace Mila::Dnn::Quant::Weight
{
    // -------------------------------------------------------------------------
    // Packed layout contract
    //
    // A quantized tensor of [rows, columns] (rows = output features, columns =
    // input features, a multiple of the group size) is stored as two planes:
    //
    //   codes    4-bit, row-major; code j of a row lives in byte j / 2, the low
    //            nibble for even j and the high nibble for odd j. A stored code
    //            is the signed step plus 8, so 0..15 means -8..7.
    //   scales   IEEE half bits (uint16_t), row-major [row][group]. A scale may
    //            be negative, including negative zero.
    //
    //   dequantized[row][column] =
    //       ( code - 8 ) * half( scales[row][column / groupSize] )
    //
    // At a group of 32 the codes and scales are Q4_0's.
    // -------------------------------------------------------------------------

    export constexpr dim_t packedRowBytesForFourBitCodes( dim_t columns ) noexcept
    {
        return ( columns + 1 ) / 2;
    }

    // -------------------------------------------------------------------------
    // Reference encode: the Q4_0 reference rounding.
    //
    //   extreme = the value of largest magnitude, the first one on a tie
    //   d       = extreme / -8                       (FP32)
    //   inverse = d != 0 ? 1 / d : 0                 (FP32)
    //   code    = min( 15, trunc( fl( x * inverse ) + 8.5 ) )
    //   scale   = half( d ), round to nearest even
    //
    // The product is rounded before 8.5 is added; a fused multiply-add is not
    // the same rule.
    // -------------------------------------------------------------------------

    /**
     * @brief Encode one group into unpacked codes (one per byte, 0..15) and its scale.
     *
     * @param values    groupSize input values.
     * @param groupSize Elements in the group.
     * @param codes     Receives groupSize codes.
     * @return          The group's scale as IEEE half bits.
     */
    export inline std::uint16_t encodeInt4Group(
        const float* values, dim_t groupSize, std::uint8_t* codes ) noexcept
    {
        float magnitude = 0.0f;
        float extreme = 0.0f;

        for ( dim_t index = 0; index < groupSize; ++index )
        {
            if ( magnitude < std::fabs( values[ index ] ) )
            {
                magnitude = std::fabs( values[ index ] );
                extreme = values[ index ];
            }
        }

        const float scale = extreme / -8.0f;
        const float inverse = scale != 0.0f ? 1.0f / scale : 0.0f;

        for ( dim_t index = 0; index < groupSize; ++index )
        {
            // Two statements, so no compiler contracts them into one fused rounding.
            const float product = values[ index ] * inverse;
            const float shifted = product + 8.5f;

            codes[ index ] = static_cast<std::uint8_t>(
                std::min( 15, static_cast<int>( static_cast<std::int8_t>( shifted ) ) ) );
        }

        return floatToHalfBits( scale );
    }

    export inline void packFourBitCodes(
        const std::uint8_t* codes, dim_t rows, dim_t columns, std::uint8_t* packed ) noexcept
    {
        const std::size_t rowBytes = static_cast<std::size_t>( packedRowBytesForFourBitCodes( columns ) );

        for ( dim_t row = 0; row < rows; ++row )
        {
            std::uint8_t* packedRow = packed + static_cast<std::size_t>( row ) * rowBytes;

            for ( std::size_t byte = 0; byte < rowBytes; ++byte )
                packedRow[ byte ] = 0;

            for ( dim_t column = 0; column < columns; ++column )
            {
                const std::uint8_t code = codes[ row * columns + column ] & 0xFu;
                packedRow[ column >> 1 ] |= static_cast<std::uint8_t>( code << ( ( column & 1 ) * 4 ) );
            }
        }
    }

    export inline void unpackFourBitCodes(
        const std::uint8_t* packed, dim_t rows, dim_t columns, std::uint8_t* codes ) noexcept
    {
        const std::size_t rowBytes = static_cast<std::size_t>( packedRowBytesForFourBitCodes( columns ) );

        for ( dim_t row = 0; row < rows; ++row )
        {
            const std::uint8_t* packedRow = packed + static_cast<std::size_t>( row ) * rowBytes;

            for ( dim_t column = 0; column < columns; ++column )
                codes[ row * columns + column ] =
                    static_cast<std::uint8_t>( ( packedRow[ column >> 1 ] >> ( ( column & 1 ) * 4 ) ) & 0xFu );
        }
    }

    /**
     * @brief Quantize a row-major matrix into the packed code plane and the scale plane.
     *
     * @param weights     [rows, columns] FP32 (a BF16 source widens exactly).
     * @param rows        Rows of the matrix.
     * @param columns     A multiple of groupSize.
     * @param groupSize   Even, and at most 256.
     * @param packed      [rows, columns / 2] bytes, fully overwritten.
     * @param scaleBits   [rows, columns / groupSize] IEEE half bits.
     */
    export inline void quantizeInt4(
        const float* weights, dim_t rows, dim_t columns, dim_t groupSize,
        std::uint8_t* packed, std::uint16_t* scaleBits ) noexcept
    {
        const dim_t rowGroups = columns / groupSize;
        const std::size_t rowBytes = static_cast<std::size_t>( packedRowBytesForFourBitCodes( columns ) );
        std::uint8_t groupCodes[ 256 ];

        for ( dim_t row = 0; row < rows; ++row )
        {
            std::uint8_t* packedRow = packed + static_cast<std::size_t>( row ) * rowBytes;

            for ( dim_t group = 0; group < rowGroups; ++group )
            {
                const dim_t first = group * groupSize;

                scaleBits[ row * rowGroups + group ] =
                    encodeInt4Group( weights + row * columns + first, groupSize, groupCodes );

                for ( dim_t index = 0; index < groupSize; index += 2 )
                {
                    packedRow[ ( first + index ) >> 1 ] = static_cast<std::uint8_t>(
                        groupCodes[ index ] | ( groupCodes[ index + 1 ] << 4 ) );
                }
            }
        }
    }

    export inline void dequantizeInt4(
        const std::uint8_t* packed, const std::uint16_t* scaleBits,
        dim_t rows, dim_t columns, dim_t groupSize, float* output ) noexcept
    {
        const dim_t rowGroups = columns / groupSize;
        const std::size_t rowBytes = static_cast<std::size_t>( packedRowBytesForFourBitCodes( columns ) );

        for ( dim_t row = 0; row < rows; ++row )
        {
            const std::uint8_t* packedRow = packed + static_cast<std::size_t>( row ) * rowBytes;

            for ( dim_t column = 0; column < columns; ++column )
            {
                const int code = ( packedRow[ column >> 1 ] >> ( ( column & 1 ) * 4 ) ) & 0xF;
                const float scale = halfBitsToFloat( scaleBits[ row * rowGroups + column / groupSize ] );

                output[ row * columns + column ] = static_cast<float>( code - 8 ) * scale;
            }
        }
    }
}
