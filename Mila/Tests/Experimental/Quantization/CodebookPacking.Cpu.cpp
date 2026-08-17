/**
 * @file CodebookPacking.Cpu.cpp
 * @brief Validates the normative packed layout and CPU reference codec for the
 * codebook policies against independently computed expectations.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

import Dnn.TensorTypes;
import Experimental.Quantization.CodebookPolicies;
import Experimental.Quantization.CodebookPacking;

using namespace Mila::Dnn;
using namespace Mila::Dnn::Experimental::Quantization;

namespace
{
    std::vector<std::uint8_t> randomCodes( dim_t count, int entryCount, unsigned seed )
    {
        std::mt19937 generator( seed );
        std::uniform_int_distribution<int> distribution( 0, entryCount - 1 );
        std::vector<std::uint8_t> codes( static_cast<std::size_t>( count ) );

        for ( auto& code : codes )
            code = static_cast<std::uint8_t>( distribution( generator ) );

        return codes;
    }
}

TEST( CodebookPolicies, CompileTimeContract )
{
    EXPECT_EQ( PerGroupCodebook2<>::kQuantizationGroupSize, 32 );
    EXPECT_EQ( PerGroupCodebook2<>::kCodeBits, 2 );
    EXPECT_EQ( PerGroupCodebook2<>::kCodebookEntries, 4 );
    EXPECT_EQ( PerGroupCodebook3<>::kQuantizationGroupSize, 64 );
    EXPECT_EQ( PerGroupCodebook3<>::kCodeBits, 3 );
    EXPECT_EQ( PerGroupCodebook3<>::kCodebookEntries, 8 );
    EXPECT_EQ( PerGroupCodebook2<>::kScaleDtype, TensorDataType::FP16 );
    EXPECT_EQ( PerGroupCodebook3<>::kScaleDtype, TensorDataType::FP16 );
}

TEST( CodebookPacking, TwoBitRoundtripWithTailColumns )
{
    // 101 columns exercises the partial final byte of each packed row.
    const dim_t rows = 3;
    const dim_t columns = 101;
    const auto codes = randomCodes( rows * columns, 4, 20260816u );

    std::vector<std::uint8_t> packed(
        static_cast<std::size_t>( rows * packedRowBytesForTwoBitCodes( columns ) ) );
    std::vector<std::uint8_t> unpacked( codes.size() );

    packTwoBitCodes( codes.data(), rows, columns, packed.data() );
    unpackTwoBitCodes( packed.data(), rows, columns, unpacked.data() );

    EXPECT_EQ( codes, unpacked );
}

TEST( CodebookPacking, TwoBitLayoutIsNormative )
{
    // Codes 0,1,2,3 in the first four columns must produce the exact byte
    // 0b11100100: code j at bit offset (j % 4) * 2. Kernels depend on this.
    const std::uint8_t codes[4] = { 0, 1, 2, 3 };
    std::uint8_t packed[1] = { 0xFF };

    packTwoBitCodes( codes, 1, 4, packed );

    EXPECT_EQ( packed[0], 0xE4 );
}

TEST( CodebookPacking, ThreeBitRoundtripWithTailColumns )
{
    // 77 columns exercises partial final bytes in both planes.
    const dim_t rows = 2;
    const dim_t columns = 77;
    const auto codes = randomCodes( rows * columns, 8, 20260817u );

    std::vector<std::uint8_t> planeTwoBits(
        static_cast<std::size_t>( rows * packedRowBytesForTwoBitCodes( columns ) ) );
    std::vector<std::uint8_t> planeOneBit(
        static_cast<std::size_t>( rows * packedRowBytesForOneBitPlane( columns ) ) );
    std::vector<std::uint8_t> unpacked( codes.size() );

    packThreeBitCodes( codes.data(), rows, columns, planeTwoBits.data(), planeOneBit.data() );
    unpackThreeBitCodes( planeTwoBits.data(), planeOneBit.data(), rows, columns, unpacked.data() );

    EXPECT_EQ( codes, unpacked );
}

TEST( CodebookPacking, HalfConversionRoundtrip )
{
    // Exactly representable halves survive the roundtrip bit-for-bit.
    const float exact[] = { 0.0f, 1.0f, -1.0f, 0.5f, 1.5f, 2.0f, -3.25f, 65504.0f };

    for ( const float value : exact )
        EXPECT_EQ( halfBitsToFloat( floatToHalfBits( value ) ), value );

    // A non-representable value lands within half precision (2^-11 relative).
    const float pi = 3.14159265f;
    EXPECT_NEAR( halfBitsToFloat( floatToHalfBits( pi ) ), pi, pi * 0.0005f );

    // A subnormal-range scale survives with the subnormal quantum, not as zero.
    const float tiny = 1.0e-6f;
    const float recovered = halfBitsToFloat( floatToHalfBits( tiny ) );
    EXPECT_GT( recovered, 0.0f );
    EXPECT_NEAR( recovered, tiny, 6.0e-8f );
}

TEST( CodebookPacking, EncodeNearestPicksNearestAndBreaksTiesLow )
{
    const float codebook[4] = { -1.0f, -0.25f, 0.25f, 1.0f };

    EXPECT_EQ( encodeNearestCode( -0.9f, codebook, 4 ), 0 );
    EXPECT_EQ( encodeNearestCode( -0.3f, codebook, 4 ), 1 );
    EXPECT_EQ( encodeNearestCode( 0.2f, codebook, 4 ), 2 );
    EXPECT_EQ( encodeNearestCode( 5.0f, codebook, 4 ), 3 );

    // Exactly equidistant between entries 1 and 2: the lower index wins.
    EXPECT_EQ( encodeNearestCode( 0.0f, codebook, 4 ), 1 );
}

TEST( CodebookPacking, DequantizeMatchesManualComputation )
{
    const dim_t rows = 2;
    const dim_t columns = 64;
    const dim_t groupSize = 32;
    const float codebook[4] = { -1.0f, -0.25f, 0.25f, 1.0f };

    std::vector<std::uint8_t> codes( static_cast<std::size_t>( rows * columns ) );

    for ( dim_t index = 0; index < rows * columns; ++index )
        codes[static_cast<std::size_t>( index )] = static_cast<std::uint8_t>( index % 4 );

    // scales[row][group], two groups per row.
    const float scaleValues[4] = { 0.5f, 2.0f, 1.0f, 4.0f };
    std::uint16_t scaleBits[4];

    for ( int scale = 0; scale < 4; ++scale )
        scaleBits[scale] = floatToHalfBits( scaleValues[scale] );

    std::vector<float> output( codes.size() );
    dequantizeCodes( codes.data(), scaleBits, codebook, rows, columns, groupSize, output.data() );

    for ( dim_t row = 0; row < rows; ++row ) {
        for ( dim_t column = 0; column < columns; ++column ) {
            const float expected =
                codebook[( row * columns + column ) % 4] * scaleValues[row * 2 + column / groupSize];
            EXPECT_EQ( output[static_cast<std::size_t>( row * columns + column )], expected );
        }
    }
}

TEST( CodebookPacking, EncodePackDequantizeIsExactlyCodebookTimesScale )
{
    // End to end over the full reference path: whatever encode selected, the
    // dequantized value must be exactly codebook[code] * half(scale) -- the
    // identity the CUDA kernels and the Python packer are both held to.
    const dim_t rows = 4;
    const dim_t columns = 96;
    const dim_t groupSize = 32;
    const float codebook[8] = { -1.0f, -0.61f, -0.34f, -0.11f, 0.09f, 0.32f, 0.58f, 1.0f };

    std::mt19937 generator( 20260818u );
    std::normal_distribution<float> distribution( 0.0f, 0.02f );
    std::vector<float> weights( static_cast<std::size_t>( rows * columns ) );

    for ( auto& weight : weights )
        weight = distribution( generator );

    const dim_t rowGroups = groupsPerRow( columns, groupSize );
    std::vector<std::uint16_t> scaleBits( static_cast<std::size_t>( rows * rowGroups ) );
    std::vector<std::uint8_t> codes( weights.size() );

    for ( dim_t row = 0; row < rows; ++row ) {
        for ( dim_t group = 0; group < rowGroups; ++group ) {
            float absmax = 0.0f;

            for ( dim_t column = group * groupSize; column < ( group + 1 ) * groupSize; ++column ) {
                const float magnitude = weights[static_cast<std::size_t>( row * columns + column )];
                absmax = std::max( absmax, magnitude < 0.0f ? -magnitude : magnitude );
            }

            const std::uint16_t bits = floatToHalfBits( absmax );
            scaleBits[static_cast<std::size_t>( row * rowGroups + group )] = bits;
            const float storedScale = halfBitsToFloat( bits );

            for ( dim_t column = group * groupSize; column < ( group + 1 ) * groupSize; ++column ) {
                const std::size_t index = static_cast<std::size_t>( row * columns + column );
                codes[index] = encodeNearestCode( weights[index] / storedScale, codebook, 8 );
            }
        }
    }

    std::vector<std::uint8_t> planeTwoBits(
        static_cast<std::size_t>( rows * packedRowBytesForTwoBitCodes( columns ) ) );
    std::vector<std::uint8_t> planeOneBit(
        static_cast<std::size_t>( rows * packedRowBytesForOneBitPlane( columns ) ) );
    std::vector<std::uint8_t> unpacked( codes.size() );
    std::vector<float> output( codes.size() );

    packThreeBitCodes( codes.data(), rows, columns, planeTwoBits.data(), planeOneBit.data() );
    unpackThreeBitCodes( planeTwoBits.data(), planeOneBit.data(), rows, columns, unpacked.data() );
    dequantizeCodes( unpacked.data(), scaleBits.data(), codebook, rows, columns, groupSize,
        output.data() );

    for ( std::size_t index = 0; index < output.size(); ++index ) {
        const dim_t row = static_cast<dim_t>( index ) / columns;
        const dim_t group = ( static_cast<dim_t>( index ) % columns ) / groupSize;
        const float scale = halfBitsToFloat( scaleBits[static_cast<std::size_t>( row * rowGroups + group )] );
        EXPECT_EQ( output[index], codebook[codes[index]] * scale );
    }
}
