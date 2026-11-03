/**
 * @file CodebookOracle.Cpu.cpp
 * @brief Bit-match between the Python packer (Tools/Quantization/packing.py)
 * and the C++ reference codec, over a generated fixture. A failure here means the
 * two implementations of the packed layout have diverged; CodebookPacking.ixx wins.
 */

#include <gtest/gtest.h>
#include <bit>
#include <cstdint>
#include <vector>

#include "Dnn/Quantization/CodebookOracle.Fixture.h"

import Dnn.TensorTypes;
import Dnn.Quantization.Weight.CodebookPacking;

using namespace Mila::Dnn;
using namespace Mila::Dnn::Quant::Weight;
namespace Fixture = CodebookOracleFixture;

namespace
{
    std::vector<float> bitsToFloats( const std::uint32_t* bits, std::size_t count )
    {
        std::vector<float> values( count );

        for ( std::size_t index = 0; index < count; ++index )
            values[index] = std::bit_cast<float>( bits[index] );

        return values;
    }
}

TEST( CodebookOracle, TwoBitPackedBytesMatchPython )
{
    const dim_t rows = Fixture::kTwoBitRows;
    const dim_t columns = Fixture::kTwoBitColumns;
    std::vector<std::uint8_t> packed(
        static_cast<std::size_t>( rows * packedRowBytesForTwoBitCodes( columns ) ) );

    packTwoBitCodes( Fixture::kTwoBitCodes, rows, columns, packed.data() );

    ASSERT_EQ( packed.size(), std::size( Fixture::kTwoBitPackedPlaneTwoBits ) );

    for ( std::size_t index = 0; index < packed.size(); ++index )
        EXPECT_EQ( packed[index], Fixture::kTwoBitPackedPlaneTwoBits[index] ) << "byte " << index;
}

TEST( CodebookOracle, TwoBitUnpackOfPythonBytesRecoversCodes )
{
    const dim_t rows = Fixture::kTwoBitRows;
    const dim_t columns = Fixture::kTwoBitColumns;
    std::vector<std::uint8_t> codes( static_cast<std::size_t>( rows * columns ) );

    unpackTwoBitCodes( Fixture::kTwoBitPackedPlaneTwoBits, rows, columns, codes.data() );

    for ( std::size_t index = 0; index < codes.size(); ++index )
        EXPECT_EQ( codes[index], Fixture::kTwoBitCodes[index] ) << "code " << index;
}

TEST( CodebookOracle, TwoBitDequantizeBitMatchesPython )
{
    const dim_t rows = Fixture::kTwoBitRows;
    const dim_t columns = Fixture::kTwoBitColumns;
    const auto codebook = bitsToFloats(
        Fixture::kTwoBitCodebookBits, std::size( Fixture::kTwoBitCodebookBits ) );
    std::vector<float> output( static_cast<std::size_t>( rows * columns ) );

    dequantizeCodes( Fixture::kTwoBitCodes, Fixture::kTwoBitScaleBits, codebook.data(),
        rows, columns, Fixture::kTwoBitGroupSize, output.data() );

    for ( std::size_t index = 0; index < output.size(); ++index )
        EXPECT_EQ( std::bit_cast<std::uint32_t>( output[index] ),
            Fixture::kTwoBitExpectedBits[index] ) << "element " << index;
}

TEST( CodebookOracle, ThreeBitPackedPlanesMatchPython )
{
    const dim_t rows = Fixture::kThreeBitRows;
    const dim_t columns = Fixture::kThreeBitColumns;
    std::vector<std::uint8_t> planeTwoBits(
        static_cast<std::size_t>( rows * packedRowBytesForTwoBitCodes( columns ) ) );
    std::vector<std::uint8_t> planeOneBit(
        static_cast<std::size_t>( rows * packedRowBytesForOneBitPlane( columns ) ) );

    packThreeBitCodes( Fixture::kThreeBitCodes, rows, columns,
        planeTwoBits.data(), planeOneBit.data() );

    ASSERT_EQ( planeTwoBits.size(), std::size( Fixture::kThreeBitPackedPlaneTwoBits ) );
    ASSERT_EQ( planeOneBit.size(), std::size( Fixture::kThreeBitPackedPlaneOneBit ) );

    for ( std::size_t index = 0; index < planeTwoBits.size(); ++index )
        EXPECT_EQ( planeTwoBits[index], Fixture::kThreeBitPackedPlaneTwoBits[index] )
            << "plane-two byte " << index;

    for ( std::size_t index = 0; index < planeOneBit.size(); ++index )
        EXPECT_EQ( planeOneBit[index], Fixture::kThreeBitPackedPlaneOneBit[index] )
            << "plane-one byte " << index;
}

TEST( CodebookOracle, ThreeBitDequantizeOfPythonPlanesBitMatchesPython )
{
    const dim_t rows = Fixture::kThreeBitRows;
    const dim_t columns = Fixture::kThreeBitColumns;
    const auto codebook = bitsToFloats(
        Fixture::kThreeBitCodebookBits, std::size( Fixture::kThreeBitCodebookBits ) );
    std::vector<std::uint8_t> codes( static_cast<std::size_t>( rows * columns ) );
    std::vector<float> output( codes.size() );

    unpackThreeBitCodes( Fixture::kThreeBitPackedPlaneTwoBits,
        Fixture::kThreeBitPackedPlaneOneBit, rows, columns, codes.data() );
    dequantizeCodes( codes.data(), Fixture::kThreeBitScaleBits, codebook.data(),
        rows, columns, Fixture::kThreeBitGroupSize, output.data() );

    for ( std::size_t index = 0; index < output.size(); ++index )
        EXPECT_EQ( std::bit_cast<std::uint32_t>( output[index] ),
            Fixture::kThreeBitExpectedBits[index] ) << "element " << index;
}
