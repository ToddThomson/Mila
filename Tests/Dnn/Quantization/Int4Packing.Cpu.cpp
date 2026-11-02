/**
 * @file Int4Packing.Cpu.cpp
 * @brief Validates the normative layout and reference rounding of the PerGroupInt4 codec, including the two
 * rounding details that decide individual Q4_0 codes.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

import Dnn.TensorTypes;
import Dnn.Quantization.Weight.CodebookPacking;
import Dnn.Quantization.Weight.Int4Packing;

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
        return std::min( 15, static_cast<int>( static_cast<std::int8_t>( shifted ) ) );
    }

    // The rule as written: round the product, then add.
    int referenceCode( float value, float inverse )
    {
        const float product = value * inverse;
        const float shifted = product + 8.5f;

        return truncatedCode( shifted );
    }
}

TEST( Int4Packing, LayoutIsNormative )
{
    std::vector<std::uint8_t> codes( 16 );

    for ( int index = 0; index < 16; ++index )
        codes[ index ] = static_cast<std::uint8_t>( index );

    std::vector<std::uint8_t> packed( 8 );
    packFourBitCodes( codes.data(), 1, 16, packed.data() );

    // Even column in the low nibble, odd column in the high nibble.
    const std::vector<std::uint8_t> expected{ 0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE };
    EXPECT_EQ( packed, expected );
}

TEST( Int4Packing, PackUnpackRoundtrip )
{
    const dim_t rows = 3;
    const dim_t columns = 64;
    std::mt19937 generator( 7 );
    std::uniform_int_distribution<int> distribution( 0, 15 );
    std::vector<std::uint8_t> codes( rows * columns );

    for ( auto& code : codes )
        code = static_cast<std::uint8_t>( distribution( generator ) );

    std::vector<std::uint8_t> packed( rows * packedRowBytesForFourBitCodes( columns ) );
    std::vector<std::uint8_t> unpacked( codes.size() );
    packFourBitCodes( codes.data(), rows, columns, packed.data() );
    unpackFourBitCodes( packed.data(), rows, columns, unpacked.data() );

    EXPECT_EQ( unpacked, codes );
}

TEST( Int4Packing, NegativeExtremeTakesCodeZeroAndTheOppositeValueClampsToFifteen )
{
    const auto values = group( { -2.0f, 2.0f, 0.25f, 0.0f } );
    std::vector<std::uint8_t> codes( kGroup );

    const std::uint16_t scale = encodeInt4Group( values.data(), kGroup, codes.data() );

    EXPECT_EQ( halfBitsToFloat( scale ), 0.25f );
    EXPECT_EQ( codes[ 0 ], 0 );
    EXPECT_EQ( codes[ 1 ], 15 );
    EXPECT_EQ( codes[ 2 ], 9 );
    EXPECT_EQ( codes[ 3 ], 8 );
}

TEST( Int4Packing, PositiveExtremeGivesANegativeScale )
{
    const auto values = group( { 2.0f, -2.0f } );
    std::vector<std::uint8_t> codes( kGroup );

    const std::uint16_t scale = encodeInt4Group( values.data(), kGroup, codes.data() );

    EXPECT_EQ( halfBitsToFloat( scale ), -0.25f );
    EXPECT_EQ( codes[ 0 ], 0 );
    EXPECT_EQ( codes[ 1 ], 15 );

    // The extreme decodes exactly whatever its sign.
    EXPECT_EQ( ( codes[ 0 ] - 8 ) * halfBitsToFloat( scale ), 2.0f );
}

TEST( Int4Packing, MagnitudeTieTakesTheFirstValue )
{
    std::vector<std::uint8_t> codes( kGroup );

    auto positive_first = group( { 1.0f, -1.0f } );
    EXPECT_EQ( halfBitsToFloat( encodeInt4Group( positive_first.data(), kGroup, codes.data() ) ), -0.125f );

    auto negative_first = group( { -1.0f, 1.0f } );
    EXPECT_EQ( halfBitsToFloat( encodeInt4Group( negative_first.data(), kGroup, codes.data() ) ), 0.125f );
}

TEST( Int4Packing, AllZeroGroupHasNegativeZeroScaleAndMiddleCodes )
{
    const auto values = group( {} );
    std::vector<std::uint8_t> codes( kGroup );

    // 0 / -8 is negative zero, and the reference stores it as such.
    EXPECT_EQ( encodeInt4Group( values.data(), kGroup, codes.data() ), 0x8000 );
    EXPECT_TRUE( std::all_of( codes.begin(), codes.end(), []( std::uint8_t code ) { return code == 8; } ) );

    // A group of negative zeros takes the same scale: the extreme stays at its initial +0.
    std::vector<float> negative_zeros( kGroup, -0.0f );
    EXPECT_EQ( encodeInt4Group( negative_zeros.data(), kGroup, codes.data() ), 0x8000 );
}

// Scans values placed on every code boundary, for several scales, and requires the codec to follow the
// two-rounding rule everywhere. The scan must also FIND values where a fused multiply-add, or dividing by
// the scale, picks a different code -- otherwise it would pass for a codec that did either.
TEST( Int4Packing, ProductIsRoundedBeforeAddingAndUsesTheReciprocal )
{
    int fusedDiffers = 0;
    int divisionDiffers = 0;
    int scanned = 0;
    std::vector<std::uint8_t> codes( kGroup );

    for ( const float extreme : { -3.0f, -2.6f, -5.0f, -7.3f, -0.37f, -11.0f } )
    {
        const float scale = extreme / -8.0f;
        const float inverse = 1.0f / scale;

        for ( int boundary = 1; boundary <= 15; ++boundary )
        {
            const float centre = ( static_cast<float>( boundary ) - 8.5f ) / inverse;
            float value = centre;

            for ( int step = 0; step < 4000; ++step )
                value = std::nextafter( value, -INFINITY );

            for ( int step = 0; step < 8000; ++step, value = std::nextafter( value, INFINITY ) )
            {
                if ( std::fabs( value ) >= std::fabs( extreme ) )
                    continue;

                auto values = group( { extreme, value } );
                encodeInt4Group( values.data(), kGroup, codes.data() );

                const int expected = referenceCode( value, inverse );
                ASSERT_EQ( codes[ 1 ], expected ) << "value " << value << " scale " << scale;
                ++scanned;

                if ( truncatedCode( std::fma( value, inverse, 8.5f ) ) != expected )
                    ++fusedDiffers;

                const float quotient = value / scale;

                if ( truncatedCode( quotient + 8.5f ) != expected )
                    ++divisionDiffers;
            }
        }
    }

    EXPECT_GT( scanned, 0 );
    EXPECT_GT( fusedDiffers, 0 ) << "the scan never reached a value the fused rounding decides differently";
    EXPECT_GT( divisionDiffers, 0 ) << "the scan never reached a value division decides differently";
}

TEST( Int4Packing, QuantizeThenDequantizeIsCodeTimesScale )
{
    const dim_t rows = 2;
    const dim_t columns = 64;
    std::mt19937 generator( 11 );
    std::normal_distribution<float> distribution( 0.0f, 0.02f );
    std::vector<float> weights( rows * columns );

    for ( auto& weight : weights )
        weight = distribution( generator );

    std::vector<std::uint8_t> packed( rows * packedRowBytesForFourBitCodes( columns ) );
    std::vector<std::uint16_t> scales( rows * columns / kGroup );
    quantizeInt4( weights.data(), rows, columns, kGroup, packed.data(), scales.data() );

    std::vector<float> dequantized( weights.size() );
    dequantizeInt4( packed.data(), scales.data(), rows, columns, kGroup, dequantized.data() );

    std::vector<std::uint8_t> codes( weights.size() );
    unpackFourBitCodes( packed.data(), rows, columns, codes.data() );

    for ( dim_t row = 0; row < rows; ++row )
    {
        for ( dim_t group_index = 0; group_index < columns / kGroup; ++group_index )
        {
            std::vector<std::uint8_t> group_codes( kGroup );
            const std::uint16_t group_scale = encodeInt4Group(
                weights.data() + row * columns + group_index * kGroup, kGroup, group_codes.data() );

            EXPECT_EQ( scales[ row * ( columns / kGroup ) + group_index ], group_scale );

            for ( dim_t index = 0; index < kGroup; ++index )
            {
                const dim_t element = row * columns + group_index * kGroup + index;

                EXPECT_EQ( codes[ element ], group_codes[ index ] );
                EXPECT_EQ( dequantized[ element ],
                    static_cast<float>( group_codes[ index ] - 8 ) * halfBitsToFloat( group_scale ) );
            }
        }
    }
}
