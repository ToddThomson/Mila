/**
 * @file TokenEmbeddingConfig.cpp
 * @brief Config-archetype tests for TokenEmbeddingConfig.
 *
 * Reference instance of the config archetype (see Specifications/Testing.md):
 * defaults, fluent setters (withVocabSize / withEmbeddingDim), validate()
 * (including the float4-vectorization embedding_dim % 4 contract), and the
 * metadata round-trip.
 *
 * Device-agnostic, so this rides the MILA_ENABLE_CUDA=OFF CI gate even though the
 * TokenEmbedding component is CUDA-only (no CPU TokenEmbeddingOp specialization).
 */

#include <gtest/gtest.h>
#include <string>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Components::Embeddings
{
    using namespace Mila::Dnn;
    using Mila::Dnn::Serialization::SerializationMetadata;

    class TokenEmbeddingConfigTests : public ::testing::Test
    {
    };

    // ====================================================================
    // A. Construction & Defaults
    // ====================================================================

    TEST_F( TokenEmbeddingConfigTests, Defaults_ZeroVocabAndDim )
    {
        TokenEmbeddingConfig config;

        EXPECT_EQ( config.getVocabSize(), 0 );
        EXPECT_EQ( config.getEmbeddingDim(), 0 );
    }

    // ====================================================================
    // Fluent interface
    // ====================================================================

    TEST_F( TokenEmbeddingConfigTests, WithVocabSize_SetsValue )
    {
        auto config = TokenEmbeddingConfig().withVocabSize( 50257 );

        EXPECT_EQ( config.getVocabSize(), 50257 );
    }

    TEST_F( TokenEmbeddingConfigTests, WithEmbeddingDim_SetsValue )
    {
        auto config = TokenEmbeddingConfig().withEmbeddingDim( 768 );

        EXPECT_EQ( config.getEmbeddingDim(), 768 );
    }

    TEST_F( TokenEmbeddingConfigTests, EmbeddingScale_DefaultsToIdentity )
    {
        EXPECT_FLOAT_EQ( TokenEmbeddingConfig().getEmbeddingScale(), 1.0f );
    }

    TEST_F( TokenEmbeddingConfigTests, WithEmbeddingScale_SetsValue )
    {
        auto config = TokenEmbeddingConfig().withEmbeddingScale( 8.0f );

        EXPECT_FLOAT_EQ( config.getEmbeddingScale(), 8.0f );
    }

    // ====================================================================
    // A. Validation
    // ====================================================================

    TEST_F( TokenEmbeddingConfigTests, Validate_PassesForValid )
    {
        EXPECT_NO_THROW(
            TokenEmbeddingConfig().withVocabSize( 32000 ).withEmbeddingDim( 64 ).validate() );
    }

    TEST_F( TokenEmbeddingConfigTests, Validate_ThrowsForZeroVocab )
    {
        EXPECT_THROW(
            TokenEmbeddingConfig().withEmbeddingDim( 64 ).validate(), std::invalid_argument );
    }

    TEST_F( TokenEmbeddingConfigTests, Validate_ThrowsForZeroEmbeddingDim )
    {
        EXPECT_THROW(
            TokenEmbeddingConfig().withVocabSize( 32000 ).validate(), std::invalid_argument );
    }

    TEST_F( TokenEmbeddingConfigTests, Validate_ThrowsForEmbeddingDimNotMultipleOfFour )
    {
        EXPECT_THROW(
            TokenEmbeddingConfig().withVocabSize( 32000 ).withEmbeddingDim( 66 ).validate(),
            std::invalid_argument );
    }

    TEST_F( TokenEmbeddingConfigTests, Validate_ThrowsForNonPositiveEmbeddingScale )
    {
        EXPECT_THROW(
            TokenEmbeddingConfig().withVocabSize( 32000 ).withEmbeddingDim( 64 )
                .withEmbeddingScale( 0.0f ).validate(),
            std::invalid_argument );

        EXPECT_THROW(
            TokenEmbeddingConfig().withVocabSize( 32000 ).withEmbeddingDim( 64 )
                .withEmbeddingScale( -2.0f ).validate(),
            std::invalid_argument );
    }

    // ====================================================================
    // H. Serialization round-trip
    // ====================================================================

    TEST_F( TokenEmbeddingConfigTests, Metadata_RoundTripPreservesFields )
    {
        TokenEmbeddingConfig source;
        source.withVocabSize( 50257 ).withEmbeddingDim( 768 ).withEmbeddingScale( 27.7128f );

        SerializationMetadata meta = source.toMetadata();

        TokenEmbeddingConfig loaded;
        loaded.fromMetadata( meta );

        EXPECT_EQ( loaded.getVocabSize(), 50257 );
        EXPECT_EQ( loaded.getEmbeddingDim(), 768 );
        EXPECT_FLOAT_EQ( loaded.getEmbeddingScale(), 27.7128f );
    }

    // ====================================================================
    // I. Diagnostics
    // ====================================================================

    TEST_F( TokenEmbeddingConfigTests, ToString_NamesFields )
    {
        TokenEmbeddingConfig config;
        config.withVocabSize( 32000 ).withEmbeddingDim( 64 );

        const std::string text = config.toString();

        EXPECT_NE( text.find( "vocab_size" ), std::string::npos );
        EXPECT_NE( text.find( "embedding_dim" ), std::string::npos );
        EXPECT_NE( text.find( "embedding_scale" ), std::string::npos );
    }
}
