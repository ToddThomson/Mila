/**
 * @file CodebookGemv.Cuda.cpp
 * @brief Validates the W2A16/W3A16 decode GEMV kernels against a CPU reference built
 * from the normative codec (CodebookPacking.ixx). The reference rounds activations and
 * bias through BF16 first -- exactly what the host entry uploads -- so the tolerance
 * covers only accumulation order and the final BF16 store.
 *
 * Compiled only under MILA_ENABLE_CUDA; each test skips if no device is present.
 */

#include <gtest/gtest.h>
#include <bit>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

import Dnn.TensorTypes;
import Experimental.Quantization.CodebookPacking;
import Experimental.Quantization.CodebookGemv;

using namespace Mila::Dnn;
using namespace Mila::Dnn::Experimental::Quantization;

namespace
{
    float roundThroughBf16( float value )
    {
        std::uint32_t bits = std::bit_cast<std::uint32_t>( value );
        bits += 0x7FFFu + ( ( bits >> 16 ) & 1u );

        return std::bit_cast<float>( bits & 0xFFFF0000u );
    }

    struct GemvCase
    {
        dim_t rows;
        dim_t columns;
        dim_t groupSize;
        int entries;
        bool withBias;
        std::vector<std::uint8_t> codes;
        std::vector<std::uint16_t> scaleBits;
        std::vector<float> codebook;
        std::vector<float> activations;
        std::vector<float> bias;
    };

    GemvCase makeCase( dim_t rows, dim_t columns, dim_t groupSize, int entries,
        bool withBias, unsigned seed )
    {
        GemvCase testCase;
        testCase.rows = rows;
        testCase.columns = columns;
        testCase.groupSize = groupSize;
        testCase.entries = entries;
        testCase.withBias = withBias;

        std::mt19937 generator( seed );
        std::uniform_int_distribution<int> codeDistribution( 0, entries - 1 );
        std::uniform_real_distribution<float> scaleDistribution( 0.01f, 2.0f );
        std::normal_distribution<float> activationDistribution( 0.0f, 1.0f );

        testCase.codes.resize( static_cast<std::size_t>( rows * columns ) );

        for ( auto& code : testCase.codes )
            code = static_cast<std::uint8_t>( codeDistribution( generator ) );

        const dim_t rowGroups = groupsPerRow( columns, groupSize );
        testCase.scaleBits.resize( static_cast<std::size_t>( rows * rowGroups ) );

        for ( auto& scale : testCase.scaleBits )
            scale = floatToHalfBits( scaleDistribution( generator ) );

        testCase.codebook.resize( static_cast<std::size_t>( entries ) );

        for ( auto& entry : testCase.codebook )
            entry = std::uniform_real_distribution<float>( -1.0f, 1.0f )( generator );

        testCase.activations.resize( static_cast<std::size_t>( columns ) );

        for ( auto& activation : testCase.activations )
            activation = activationDistribution( generator );

        if ( withBias ) {
            testCase.bias.resize( static_cast<std::size_t>( rows ) );

            for ( auto& value : testCase.bias )
                value = activationDistribution( generator );
        }

        return testCase;
    }

    std::vector<float> referenceGemv( const GemvCase& testCase )
    {
        std::vector<float> dequantized(
            static_cast<std::size_t>( testCase.rows * testCase.columns ) );
        dequantizeCodes( testCase.codes.data(), testCase.scaleBits.data(),
            testCase.codebook.data(), testCase.rows, testCase.columns, testCase.groupSize,
            dequantized.data() );

        std::vector<float> expected( static_cast<std::size_t>( testCase.rows ) );

        for ( dim_t row = 0; row < testCase.rows; ++row ) {
            double accumulator = 0.0;

            for ( dim_t column = 0; column < testCase.columns; ++column ) {
                const float activation = roundThroughBf16(
                    testCase.activations[static_cast<std::size_t>( column )] );
                accumulator += static_cast<double>( activation )
                    * dequantized[static_cast<std::size_t>( row * testCase.columns + column )];
            }

            if ( testCase.withBias )
                accumulator += roundThroughBf16( testCase.bias[static_cast<std::size_t>( row )] );

            expected[static_cast<std::size_t>( row )] = static_cast<float>( accumulator );
        }

        return expected;
    }

    void runAndCompare( const GemvCase& testCase )
    {
        const std::vector<float> expected = referenceGemv( testCase );
        std::vector<float> output( static_cast<std::size_t>( testCase.rows ) );
        int status = 0;

        if ( testCase.entries == 4 ) {
            std::vector<std::uint8_t> packed( static_cast<std::size_t>(
                testCase.rows * packedRowBytesForTwoBitCodes( testCase.columns ) ) );
            packTwoBitCodes( testCase.codes.data(), testCase.rows, testCase.columns,
                packed.data() );

            status = runCodebook2MatVecDecodeOnDevice(
                testCase.activations.data(), packed.data(), testCase.scaleBits.data(),
                testCase.codebook.data(), testCase.withBias ? testCase.bias.data() : nullptr,
                output.data(), static_cast<int>( testCase.columns ),
                static_cast<int>( testCase.rows ), static_cast<int>( testCase.groupSize ) );
        } else {
            std::vector<std::uint8_t> planeTwoBits( static_cast<std::size_t>(
                testCase.rows * packedRowBytesForTwoBitCodes( testCase.columns ) ) );
            std::vector<std::uint8_t> planeOneBit( static_cast<std::size_t>(
                testCase.rows * packedRowBytesForOneBitPlane( testCase.columns ) ) );
            packThreeBitCodes( testCase.codes.data(), testCase.rows, testCase.columns,
                planeTwoBits.data(), planeOneBit.data() );

            status = runCodebook3MatVecDecodeOnDevice(
                testCase.activations.data(), planeTwoBits.data(), planeOneBit.data(),
                testCase.scaleBits.data(), testCase.codebook.data(),
                testCase.withBias ? testCase.bias.data() : nullptr, output.data(),
                static_cast<int>( testCase.columns ), static_cast<int>( testCase.rows ),
                static_cast<int>( testCase.groupSize ) );
        }

        ASSERT_EQ( status, 0 );

        for ( dim_t row = 0; row < testCase.rows; ++row ) {
            const float reference = expected[static_cast<std::size_t>( row )];
            // BF16 output store dominates: 2^-8 relative, plus a small absolute floor
            // for accumulation-order differences near zero.
            const float tolerance = 0.01f + 0.005f * std::abs( reference );
            EXPECT_NEAR( output[static_cast<std::size_t>( row )], reference, tolerance )
                << "row " << row;
        }
    }
}

#define SKIP_WITHOUT_DEVICE() \
    if ( !codebookGemvDeviceAvailable() ) GTEST_SKIP() << "No CUDA device available."

TEST( CodebookGemvCuda, TwoBitGroup32WithBiasAndBlockTail )
{
    SKIP_WITHOUT_DEVICE();
    // 27 rows exercises the partial final block (kBlockOutputs = 8); 544 columns is
    // 17 groups of 32 and a non-power-of-two walk.
    runAndCompare( makeCase( 27, 544, 32, 4, true, 20260819u ) );
}

TEST( CodebookGemvCuda, ThreeBitGroup64NoBias )
{
    SKIP_WITHOUT_DEVICE();
    runAndCompare( makeCase( 17, 640, 64, 8, false, 20260820u ) );
}

TEST( CodebookGemvCuda, TwoBitGroup128SmokesAlternateInstantiation )
{
    SKIP_WITHOUT_DEVICE();
    runAndCompare( makeCase( 8, 512, 128, 4, false, 20260821u ) );
}

TEST( CodebookGemvCuda, ThreeBitModelShapedRow )
{
    SKIP_WITHOUT_DEVICE();
    // A Llama 3.2 3B shaped slice: 3072 columns, the width the Phase 1 validation
    // model actually runs.
    runAndCompare( makeCase( 16, 3072, 64, 8, false, 20260822u ) );
}
