/**
 * @file Int6Packing.ixx
 * @brief Normative packed layout and CPU reference codec for the PerGroupInt6 policy.
 * The CUDA quantizer and kernels must match this file exactly; a disagreement is resolved in its favor.
 */

module;
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

export module Dnn.Quantization.Weight.Int6Packing;

import Dnn.TensorTypes;
import Dnn.Quantization.Weight.CodebookPacking;

namespace Mila::Dnn::Quant::Weight
{
    // -------------------------------------------------------------------------
    // Packed layout contract
    //
    // A quantized tensor of [rows, columns] (rows = output features, columns =
    // input features, a multiple of the group size and of 4) is stored as two
    // tensors:
    //
    //   codes    [rows, 3 * columns / 4] bytes. Each row is its codes' low
    //            nibbles followed by their high two bits:
    //              bytes [0, columns / 2)             code j's low four bits in
    //                                                 byte j / 2, the low nibble
    //                                                 for even j (Int4Packing's
    //                                                 order)
    //              bytes [columns / 2, 3 * columns / 4)
    //                                                 code j's high two bits in
    //                                                 byte columns / 2 + j / 4,
    //                                                 at bit 2 * ( j % 4 )
    //            A stored code is the signed step plus 32, so 0..63 means
    //            -32..31.
    //   scales   IEEE half bits (uint16_t), row-major [row][group]. A scale may
    //            be negative, including negative zero.
    //
    //   dequantized[row][column] =
    //       ( code - 32 ) * half( scales[row][column / groupSize] )
    //
    // Each row is one contiguous run of bytes, so a row gather and a strip of
    // rows address the tensor as any other weight; the two planes keep a
    // group's low nibbles at Q4_0's offsets and its high bits in 8 bytes.
    // -------------------------------------------------------------------------

    export constexpr dim_t packedRowBytesForSixBitCodes( dim_t columns ) noexcept
    {
        return columns / 2 + columns / 4;
    }

    // -------------------------------------------------------------------------
    // Reference encode: Q4_0's reference rounding, widened to six bits.
    //
    //   extreme = the value of largest magnitude, the first one on a tie
    //   d       = extreme / -32                      (FP32)
    //   inverse = d != 0 ? 1 / d : 0                 (FP32)
    //   code    = min( 63, trunc( fl( x * inverse ) + 32.5 ) )
    //   scale   = half( d ), round to nearest even
    //
    // The product is rounded before 32.5 is added; a fused multiply-add is not
    // the same rule.
    // -------------------------------------------------------------------------

    /**
     * @brief Encode one group into unpacked codes (one per byte, 0..63) and its scale.
     *
     * @param values    groupSize input values.
     * @param groupSize Elements in the group.
     * @param codes     Receives groupSize codes.
     * @return          The group's scale as IEEE half bits.
     */
    export inline std::uint16_t encodeInt6Group(
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

        const float scale = extreme / -32.0f;
        const float inverse = scale != 0.0f ? 1.0f / scale : 0.0f;

        for ( dim_t index = 0; index < groupSize; ++index )
        {
            // Two statements, so no compiler contracts them into one fused rounding.
            const float product = values[ index ] * inverse;
            const float shifted = product + 32.5f;

            codes[ index ] = static_cast<std::uint8_t>( std::min( 63, static_cast<int>( shifted ) ) );
        }

        return floatToHalfBits( scale );
    }

    /**
     * @brief Pack unpacked codes (one per byte, 0..63) into the code tensor's row layout.
     */
    export inline void packSixBitCodes(
        const std::uint8_t* codes, dim_t rows, dim_t columns, std::uint8_t* packed ) noexcept
    {
        const std::size_t rowBytes = static_cast<std::size_t>( packedRowBytesForSixBitCodes( columns ) );
        const std::size_t highOffset = static_cast<std::size_t>( columns / 2 );

        for ( dim_t row = 0; row < rows; ++row )
        {
            std::uint8_t* packedRow = packed + static_cast<std::size_t>( row ) * rowBytes;

            for ( std::size_t byte = 0; byte < rowBytes; ++byte )
                packedRow[ byte ] = 0;

            for ( dim_t column = 0; column < columns; ++column )
            {
                const std::uint8_t code = codes[ row * columns + column ] & 0x3Fu;

                packedRow[ column >> 1 ] |= static_cast<std::uint8_t>( ( code & 0xFu ) << ( ( column & 1 ) * 4 ) );
                packedRow[ highOffset + ( column >> 2 ) ] |= static_cast<std::uint8_t>( ( code >> 4 ) << ( ( column & 3 ) * 2 ) );
            }
        }
    }

    export inline void unpackSixBitCodes(
        const std::uint8_t* packed, dim_t rows, dim_t columns, std::uint8_t* codes ) noexcept
    {
        const std::size_t rowBytes = static_cast<std::size_t>( packedRowBytesForSixBitCodes( columns ) );
        const std::size_t highOffset = static_cast<std::size_t>( columns / 2 );

        for ( dim_t row = 0; row < rows; ++row )
        {
            const std::uint8_t* packedRow = packed + static_cast<std::size_t>( row ) * rowBytes;

            for ( dim_t column = 0; column < columns; ++column )
            {
                const int low = ( packedRow[ column >> 1 ] >> ( ( column & 1 ) * 4 ) ) & 0xF;
                const int high = ( packedRow[ highOffset + ( column >> 2 ) ] >> ( ( column & 3 ) * 2 ) ) & 0x3;

                codes[ row * columns + column ] = static_cast<std::uint8_t>( low | ( high << 4 ) );
            }
        }
    }

    /**
     * @brief Quantize a row-major matrix into the code tensor and the scale plane.
     *
     * @param weights     [rows, columns] FP32 (a BF16 source widens exactly).
     * @param rows        Rows of the matrix.
     * @param columns     A multiple of groupSize and of 4.
     * @param groupSize   Elements that share one scale.
     * @param packed      [rows, 3 * columns / 4] bytes, fully overwritten.
     * @param scaleBits   [rows, columns / groupSize] IEEE half bits.
     */
    export inline void quantizeInt6(
        const float* weights, dim_t rows, dim_t columns, dim_t groupSize,
        std::uint8_t* packed, std::uint16_t* scaleBits ) noexcept
    {
        const dim_t rowGroups = columns / groupSize;
        const std::size_t rowBytes = static_cast<std::size_t>( packedRowBytesForSixBitCodes( columns ) );
        std::vector<std::uint8_t> rowCodes( static_cast<std::size_t>( columns ) );

        for ( dim_t row = 0; row < rows; ++row )
        {
            for ( dim_t group = 0; group < rowGroups; ++group )
            {
                scaleBits[ row * rowGroups + group ] = encodeInt6Group(
                    weights + row * columns + group * groupSize, groupSize, rowCodes.data() + group * groupSize );
            }

            packSixBitCodes( rowCodes.data(), 1, columns, packed + static_cast<std::size_t>( row ) * rowBytes );
        }
    }

    export inline void dequantizeInt6(
        const std::uint8_t* packed, const std::uint16_t* scaleBits,
        dim_t rows, dim_t columns, dim_t groupSize, float* output ) noexcept
    {
        const dim_t rowGroups = columns / groupSize;
        const std::size_t rowBytes = static_cast<std::size_t>( packedRowBytesForSixBitCodes( columns ) );
        const std::size_t highOffset = static_cast<std::size_t>( columns / 2 );

        for ( dim_t row = 0; row < rows; ++row )
        {
            const std::uint8_t* packedRow = packed + static_cast<std::size_t>( row ) * rowBytes;

            for ( dim_t column = 0; column < columns; ++column )
            {
                const int low = ( packedRow[ column >> 1 ] >> ( ( column & 1 ) * 4 ) ) & 0xF;
                const int high = ( packedRow[ highOffset + ( column >> 2 ) ] >> ( ( column & 3 ) * 2 ) ) & 0x3;
                const float scale = halfBitsToFloat( scaleBits[ row * rowGroups + column / groupSize ] );

                output[ row * columns + column ] = static_cast<float>( ( low | ( high << 4 ) ) - 32 ) * scale;
            }
        }
    }
}
