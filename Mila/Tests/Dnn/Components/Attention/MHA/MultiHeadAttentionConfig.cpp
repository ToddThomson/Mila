/**
 * @file MultiHeadAttentionConfig.cpp
 * @brief Config-archetype tests for MultiHeadAttentionConfig.
 *
 * Reference instance of the config archetype (see Specifications/Testing.md):
 * required-constructor fields (model_dim, num_heads), fluent overrides, validate()
 * (the documented @throws), and the metadata round-trip. Device-agnostic, rides
 * the CPU gate.
 */

#include <gtest/gtest.h>
#include <string>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Components::Attention::MHA
{
    using namespace Mila::Dnn;
    using Mila::Dnn::Serialization::SerializationMetadata;

    class MultiHeadAttentionConfigTests : public ::testing::Test
    {
    };

    // ====================================================================
    // A. Construction & Defaults
    // ====================================================================

    TEST_F( MultiHeadAttentionConfigTests, Construct_StoresFields )
    {
        MultiHeadAttentionConfig config( 64, 8 );

        EXPECT_EQ( config.getModelDim(), 64 );
        EXPECT_EQ( config.getNumHeads(), 8 );
    }

    // ====================================================================
    // Fluent interface
    // ====================================================================

    TEST_F( MultiHeadAttentionConfigTests, WithModelDim_SetsValue )
    {
        auto config = MultiHeadAttentionConfig( 64, 8 ).withModelDim( 128 );

        EXPECT_EQ( config.getModelDim(), 128 );
    }

    TEST_F( MultiHeadAttentionConfigTests, WithNumHeads_SetsValue )
    {
        auto config = MultiHeadAttentionConfig( 64, 8 ).withNumHeads( 4 );

        EXPECT_EQ( config.getNumHeads(), 4 );
    }

    // ====================================================================
    // A. Validation
    // ====================================================================

    TEST_F( MultiHeadAttentionConfigTests, Validate_PassesForValid )
    {
        EXPECT_NO_THROW( MultiHeadAttentionConfig( 64, 8 ).validate() );
    }

    TEST_F( MultiHeadAttentionConfigTests, Validate_ThrowsForNonPositiveModelDim )
    {
        EXPECT_THROW( MultiHeadAttentionConfig( 0, 8 ).validate(), std::invalid_argument );
    }

    TEST_F( MultiHeadAttentionConfigTests, Validate_ThrowsForTooFewHeads )
    {
        EXPECT_THROW( MultiHeadAttentionConfig( 64, 1 ).validate(), std::invalid_argument );
    }

    TEST_F( MultiHeadAttentionConfigTests, Validate_ThrowsWhenModelDimNotDivisibleByHeads )
    {
        EXPECT_THROW( MultiHeadAttentionConfig( 60, 8 ).validate(), std::invalid_argument );
    }

    // ====================================================================
    // H. Serialization round-trip
    // ====================================================================

    TEST_F( MultiHeadAttentionConfigTests, Metadata_RoundTripPreservesFields )
    {
        MultiHeadAttentionConfig source( 128, 16 );

        SerializationMetadata meta = source.toMetadata();

        MultiHeadAttentionConfig loaded( 1, 1 );
        loaded.fromMetadata( meta );

        EXPECT_EQ( loaded.getModelDim(), 128 );
        EXPECT_EQ( loaded.getNumHeads(), 16 );
    }

    // ====================================================================
    // I. Diagnostics
    // ====================================================================

    TEST_F( MultiHeadAttentionConfigTests, ToString_NamesConfig )
    {
        MultiHeadAttentionConfig config( 64, 8 );

        EXPECT_NE( config.toString().find( "MultiHeadAttentionConfig" ), std::string::npos );
    }
}
