/**
 * @file Int6Packing.Cpu.cpp
 * @brief Validates the normative layout and reference rounding of the PerGroupInt6 codec: Q4_0's rule widened to
 * six bits, with a row's low nibbles ahead of its high two bits.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

import Dnn.TensorTypes;
import Dnn.Quantization.Weight.CodebookPacking;
import Dnn.Quantization.Weight.Int6Packing;

using namespace Mila::Dnn;
using namespace Mila::Dnn::Quant::Weight;

namespace
{
    constexpr dim_t kGroup = 32;

    std::vector<float> group( std::initializer_list<float> leading )
    {
        std::vector<float> values( kGroup, 0.0f );
        std::copy( leading.begin(), leading.end(), values.begin() );

        return values;
    }

    int truncatedCode( float shifted )
    {
        return std::min( 63, static_cast<int>( shifted ) );
    }

    // The rule as written: round the product, then add.
    int referenceCode( float value, float inverse )
    {
        const float product = value * inverse;
        const float shifted = product + 32.5f;

        return truncatedCode( shifted );
    }
}

TEST( Int6Packing, LayoutIsNormative )
{
    // Eight codes whose low nibbles and high bits are all distinct: 0x00, 0x11, 0x22, 0x33, 0x0C, 0x1D, 0x2E, 0x3F.
    const std::vector<std::uint8_t> codes{ 0x00, 0x11, 0x22, 0x33, 0x0C, 0x1D, 0x2E, 0x3F };
    std::vector<std::uint8_t> packed( packedRowBytesForSixBitCodes( 8 ) );

    packSixBitCodes( codes.data(), 1, 8, packed.data() );

    // Low nibbles first, even column in the low nibble; then the high two bits, column j at bit 2 * ( j % 4 ).
    const std::vector<std::uint8_t> expected{ 0x10, 0x32, 0xDC, 0xFE, 0xE4, 0xE4 };
    EXPECT_EQ( packed, expected );
}

TEST( Int6Packing, RowIsThreeQuartersOfItsColumns )
{
    EXPECT_EQ( packedRowBytesForSixBitCodes( 32 ), 24 );
    EXPECT_EQ( packedRowBytesForSixBitCodes( 2816 ), 2112 );
    EXPECT_EQ( packedRowBytesForSixBitCodes( 3840 ), 2880 );
}

TEST( Int6Packing, PackUnpackRoundtrip )
{
    const dim_t rows = 3;
    const dim_t columns = 96;
    std::mt19937 generator( 7 );
    std::uniform_int_distribution<int> distribution( 0, 63 );
    std::vector<std::uint8_t> codes( rows * columns );

    for ( auto& code : codes )
        code = static_cast<std::uint8_t>( distribution( generator ) );

    std::vector<std::uint8_t> packed( rows * packedRowBytesForSixBitCodes( columns ) );
    std::vector<std::uint8_t> unpacked( codes.size() );
    packSixBitCodes( codes.data(), rows, columns, packed.data() );
    unpackSixBitCodes( packed.data(), rows, columns, unpacked.data() );

    EXPECT_EQ( unpacked, codes );
}

TEST( Int6Packing, NegativeExtremeTakesCodeZeroAndTheOppositeValueClampsToSixtyThree )
{
    const auto values = group( { -8.0f, 8.0f, 0.25f, 0.0f } );
    std::vector<std::uint8_t> codes( kGroup );

    const std::uint16_t scale = encodeInt6Group( values.data(), kGroup, codes.data() );

    EXPECT_EQ( halfBitsToFloat( scale ), 0.25f );
    EXPECT_EQ( codes[ 0 ], 0 );
    EXPECT_EQ( codes[ 1 ], 63 );
    EXPECT_EQ( codes[ 2 ], 33 );
    EXPECT_EQ( codes[ 3 ], 32 );
}

TEST( Int6Packing, PositiveExtremeGivesANegativeScale )
{
    const auto values = group( { 8.0f, -8.0f } );
    std::vector<std::uint8_t> codes( kGroup );

    const std::uint16_t scale = encodeInt6Group( values.data(), kGroup, codes.data() );

    EXPECT_EQ( halfBitsToFloat( scale ), -0.25f );
    EXPECT_EQ( codes[ 0 ], 0 );
    EXPECT_EQ( codes[ 1 ], 63 );

    // The extreme decodes exactly whatever its sign.
    EXPECT_EQ( ( codes[ 0 ] - 32 ) * halfBitsToFloat( scale ), 8.0f );
}

TEST( Int6Packing, MagnitudeTieTakesTheFirstValue )
{
    std::vector<std::uint8_t> codes( kGroup );

    auto positive_first = group( { 1.0f, -1.0f } );
    EXPECT_EQ( halfBitsToFloat( encodeInt6Group( positive_first.data(), kGroup, codes.data() ) ), -0.03125f );

    auto negative_first = group( { -1.0f, 1.0f } );
    EXPECT_EQ( halfBitsToFloat( encodeInt6Group( negative_first.data(), kGroup, codes.data() ) ), 0.03125f );
}

TEST( Int6Packing, AllZeroGroupHasNegativeZeroScaleAndMiddleCodes )
{
    const auto values = group( {} );
    std::vector<std::uint8_t> codes( kGroup );

    // 0 / -32 is negative zero, and the reference stores it as such.
    EXPECT_EQ( encodeInt6Group( values.data(), kGroup, codes.data() ), 0x8000 );
    EXPECT_TRUE( std::all_of( codes.begin(), codes.end(), []( std::uint8_t code ) { return code == 32; } ) );

    // A group of negative zeros takes the same scale: the extreme stays at its initial +0.
    std::vector<float> negative_zeros( kGroup, -0.0f );
    EXPECT_EQ( encodeInt6Group( negative_zeros.data(), kGroup, codes.data() ), 0x8000 );
}

// Scans values placed on every code boundary, for several scales, and requires the codec to follow the
// two-rounding rule everywhere. The scan must also FIND values where a fused multiply-add, or dividing by
// the scale, picks a different code -- otherwise it would pass for a codec that did either.
TEST( Int6Packing, ProductIsRoundedBeforeAddingAndUsesTheReciprocal )
{
    int fusedDiffers = 0;
    int divisionDiffers = 0;
    int scanned = 0;
    std::vector<std::uint8_t> codes( kGroup );

    for ( const float extreme : { -3.0f, -2.6f, -5.0f, -7.3f, -0.37f, -11.0f } )
    {
        const float scale = extreme / -32.0f;
        const float inverse = 1.0f / scale;

        for ( int boundary = 1; boundary <= 63; ++boundary )
        {
            const float centre = ( static_cast<float>( boundary ) - 32.5f ) / inverse;
            float value = centre;

            for ( int step = 0; step < 1000; ++step )
                value = std::nextafter( value, -INFINITY );

            for ( int step = 0; step < 2000; ++step, value = std::nextafter( value, INFINITY ) )
            {
                if ( std::fabs( value ) >= std::fabs( extreme ) )
                    continue;

                auto values = group( { extreme, value } );
                encodeInt6Group( values.data(), kGroup, codes.data() );

                const int expected = referenceCode( value, inverse );
                ASSERT_EQ( codes[ 1 ], expected ) << "value " << value << " scale " << scale;
                ++scanned;

                if ( truncatedCode( std::fma( value, inverse, 32.5f ) ) != expected )
                    ++fusedDiffers;

                const float quotient = value / scale;

                if ( truncatedCode( quotient + 32.5f ) != expected )
                    ++divisionDiffers;
            }
        }
    }

    EXPECT_GT( scanned, 0 );
    EXPECT_GT( fusedDiffers, 0 ) << "the scan never reached a value the fused rounding decides differently";
    EXPECT_GT( divisionDiffers, 0 ) << "the scan never reached a value division decides differently";
}

TEST( Int6Packing, QuantizeThenDequantizeIsCodeTimesScale )
{
    const dim_t rows = 2;
    const dim_t columns = 128;
    std::mt19937 generator( 11 );
    std::normal_distribution<float> distribution( 0.0f, 0.02f );
    std::vector<float> weights( rows * columns );

    for ( auto& weight : weights )
        weight = distribution( generator );

    std::vector<std::uint8_t> packed( rows * packedRowBytesForSixBitCodes( columns ) );
    std::vector<std::uint16_t> scales( rows * columns / kGroup );
    quantizeInt6( weights.data(), rows, columns, kGroup, packed.data(), scales.data() );

    std::vector<float> dequantized( weights.size() );
    dequantizeInt6( packed.data(), scales.data(), rows, columns, kGroup, dequantized.data() );

    std::vector<std::uint8_t> codes( weights.size() );
    unpackSixBitCodes( packed.data(), rows, columns, codes.data() );

    for ( dim_t row = 0; row < rows; ++row )
    {
        for ( dim_t group_index = 0; group_index < columns / kGroup; ++group_index )
        {
            std::vector<std::uint8_t> group_codes( kGroup );
            const std::uint16_t group_scale = encodeInt6Group(
                weights.data() + row * columns + group_index * kGroup, kGroup, group_codes.data() );

            EXPECT_EQ( scales[ row * ( columns / kGroup ) + group_index ], group_scale );

            for ( dim_t index = 0; index < kGroup; ++index )
            {
                const dim_t element = row * columns + group_index * kGroup + index;

                EXPECT_EQ( codes[ element ], group_codes[ index ] );
                EXPECT_EQ( dequantized[ element ],
                    static_cast<float>( group_codes[ index ] - 32 ) * halfBitsToFloat( group_scale ) );
            }
        }
    }
}

// Every weight lands within half a step of its value, except a value near the opposite extreme, which the clamp to
// 63 holds a whole step short.
TEST( Int6Packing, ErrorIsBoundedByHalfAStep )
{
    const dim_t rows = 4;
    const dim_t columns = 256;
    std::mt19937 generator( 13 );
    std::normal_distribution<float> distribution( 0.0f, 0.02f );
    std::vector<float> weights( rows * columns );

    for ( auto& weight : weights )
        weight = distribution( generator );

    std::vector<std::uint8_t> packed( rows * packedRowBytesForSixBitCodes( columns ) );
    std::vector<std::uint16_t> scales( rows * columns / kGroup );
    quantizeInt6( weights.data(), rows, columns, kGroup, packed.data(), scales.data() );

    std::vector<float> dequantized( weights.size() );
    dequantizeInt6( packed.data(), scales.data(), rows, columns, kGroup, dequantized.data() );

    std::vector<std::uint8_t> codes( weights.size() );
    unpackSixBitCodes( packed.data(), rows, columns, codes.data() );

    for ( dim_t element = 0; element < rows * columns; ++element )
    {
        const float step = std::fabs( halfBitsToFloat( scales[ element / kGroup ] ) );
        const float rounding = codes[ element ] == 63 ? step : 0.5f * step;

        // Plus what rounding d to FP16 moves the largest code and the half step by.
        EXPECT_LE( std::fabs( dequantized[ element ] - weights[ element ] ), rounding + 33.0f * step * 0x1p-11f )
            << "element " << element;
    }
}
