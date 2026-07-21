/**
 * @file RopeConfig.cpp
 * @brief Config-archetype tests for RopeConfig.
 *
 * Reference instance of the config archetype (see Specifications/Testing.md):
 * the required-constructor / optional-fluent split (channels, n_heads,
 * n_kv_heads, max_seq_len are structural constructor args; base and rotary_dim
 * are fluent), the derived head_dim accessor, validate() (the full set of
 * documented @throws), and the metadata round-trip.
 *
 * Device-agnostic, so this rides the MILA_ENABLE_CUDA=OFF CI gate even though the
 * Rope component is CUDA-only (no CPU RopeOp specialization).
 */

#include <gtest/gtest.h>
#include <string>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Components::Encodings::Rope
{
    using namespace Mila::Dnn;
    using Mila::Dnn::Serialization::SerializationMetadata;

    class RopeConfigTests : public ::testing::Test
    {
    };

    // ====================================================================
    // A. Construction & Defaults
    // ====================================================================

    TEST_F( RopeConfigTests, Construct_StoresRequiredFields )
    {
        RopeConfig config( 64, 8, 4, 128 );

        EXPECT_EQ( config.getEmbeddingDim(), 64u );
        EXPECT_EQ( config.getNumHeads(), 8u );
        EXPECT_EQ( config.getNumKVHeads(), 4u );
        EXPECT_EQ( config.getMaxSequenceLength(), 128u );
    }

    TEST_F( RopeConfigTests, Defaults_BaseTenThousandFullRotary )
    {
        RopeConfig config( 64, 8, 4, 128 );

        EXPECT_FLOAT_EQ( config.getBase(), 10000.0f );
        EXPECT_EQ( config.getRotaryDim(), 0u );
    }

    TEST_F( RopeConfigTests, HeadDim_DerivedFromChannelsAndHeads )
    {
        RopeConfig config( 64, 8, 4, 128 );

        EXPECT_EQ( config.getHeadDim(), 8u );
    }

    // ====================================================================
    // Fluent interface (optional behavioural parameters)
    // ====================================================================

    TEST_F( RopeConfigTests, WithBase_SetsValue )
    {
        auto config = RopeConfig( 64, 8, 4, 128 ).withBase( 500000.0f );

        EXPECT_FLOAT_EQ( config.getBase(), 500000.0f );
    }

    TEST_F( RopeConfigTests, WithRotaryDim_SetsValue )
    {
        auto config = RopeConfig( 64, 8, 4, 128 ).withRotaryDim( 4 );

        EXPECT_EQ( config.getRotaryDim(), 4u );
    }

    // ====================================================================
    // A. Validation (each documented @throws is one negative test)
    // ====================================================================

    TEST_F( RopeConfigTests, Validate_PassesForValid )
    {
        EXPECT_NO_THROW( RopeConfig( 64, 8, 4, 128 ).validate() );
    }

    TEST_F( RopeConfigTests, Validate_ThrowsForZeroChannels )
    {
        EXPECT_THROW( RopeConfig( 0, 8, 4, 128 ).validate(), std::invalid_argument );
    }

    TEST_F( RopeConfigTests, Validate_ThrowsForZeroHeads )
    {
        EXPECT_THROW( RopeConfig( 64, 0, 4, 128 ).validate(), std::invalid_argument );
    }

    TEST_F( RopeConfigTests, Validate_ThrowsForChannelsNotDivisibleByHeads )
    {
        EXPECT_THROW( RopeConfig( 60, 8, 4, 128 ).validate(), std::invalid_argument );
    }

    TEST_F( RopeConfigTests, Validate_ThrowsForOddHeadDim )
    {
        // channels / n_heads = 24 / 8 = 3 (odd) — RoPE requires paired dimensions.
        EXPECT_THROW( RopeConfig( 24, 8, 4, 128 ).validate(), std::invalid_argument );
    }

    TEST_F( RopeConfigTests, Validate_ThrowsWhenKVHeadsExceedHeads )
    {
        EXPECT_THROW( RopeConfig( 64, 4, 8, 128 ).validate(), std::invalid_argument );
    }

    TEST_F( RopeConfigTests, Validate_ThrowsForZeroMaxSeqLen )
    {
        EXPECT_THROW( RopeConfig( 64, 8, 4, 0 ).validate(), std::invalid_argument );
    }

    TEST_F( RopeConfigTests, Validate_ThrowsForNonPositiveBase )
    {
        EXPECT_THROW( RopeConfig( 64, 8, 4, 128 ).withBase( 0.0f ).validate(), std::invalid_argument );
    }

    TEST_F( RopeConfigTests, Validate_ThrowsWhenRotaryDimExceedsHeadDim )
    {
        // head_dim = 8; rotary_dim = 16 > head_dim.
        EXPECT_THROW( RopeConfig( 64, 8, 4, 128 ).withRotaryDim( 16 ).validate(), std::invalid_argument );
    }

    // ====================================================================
    // H. Serialization round-trip
    // ====================================================================

    TEST_F( RopeConfigTests, Metadata_RoundTripPreservesFields )
    {
        RopeConfig source( 64, 8, 4, 128 );
        source.withBase( 500000.0f ).withRotaryDim( 4 );

        SerializationMetadata meta = source.toMetadata();

        RopeConfig loaded( 1, 1, 1, 1 );
        loaded.fromMetadata( meta );

        EXPECT_EQ( loaded.getEmbeddingDim(), 64u );
        EXPECT_EQ( loaded.getNumHeads(), 8u );
        EXPECT_EQ( loaded.getNumKVHeads(), 4u );
        EXPECT_EQ( loaded.getMaxSequenceLength(), 128u );
        EXPECT_FLOAT_EQ( loaded.getBase(), 500000.0f );
        EXPECT_EQ( loaded.getRotaryDim(), 4u );
    }

    // ====================================================================
    // I. Diagnostics
    // ====================================================================

    TEST_F( RopeConfigTests, ToString_NamesConfig )
    {
        RopeConfig config( 64, 8, 4, 128 );

        EXPECT_NE( config.toString().find( "RopeConfig" ), std::string::npos );
    }
}
