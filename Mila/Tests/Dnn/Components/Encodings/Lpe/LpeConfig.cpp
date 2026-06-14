/**
 * @file LpeConfig.cpp
 * @brief Config-archetype tests for LpeConfig (learned positional encoder).
 *
 * Reference instance of the config archetype (see Specifications/Testing.md):
 * fluent setters (withEmbeddingDim / withMaxSequenceLength / withVocabularyLength),
 * validate(), and the metadata round-trip. Device-agnostic, rides the CPU gate.
 *
 * Supersedes the pre-methodology EncoderConfig.cpp (retired in place, Section 3).
 */

#include <gtest/gtest.h>
#include <string>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Components::Encodings::Lpe
{
    using namespace Mila::Dnn;
    using Mila::Dnn::Serialization::SerializationMetadata;

    class LpeConfigTests : public ::testing::Test
    {
    };

    // ====================================================================
    // A. Construction & Defaults
    // ====================================================================

    TEST_F( LpeConfigTests, Defaults_MaxSeqAndVocabPreset )
    {
        LpeConfig config;

        EXPECT_EQ( config.getEmbeddingDim(), 0u );
        EXPECT_EQ( config.getMaxSequenceLength(), 512u );
        EXPECT_EQ( config.getVocabularyLength(), 50000u );
    }

    // ====================================================================
    // Fluent interface
    // ====================================================================

    TEST_F( LpeConfigTests, WithEmbeddingDim_SetsValue )
    {
        auto config = LpeConfig().withEmbeddingDim( 768 );

        EXPECT_EQ( config.getEmbeddingDim(), 768u );
    }

    TEST_F( LpeConfigTests, WithMaxSequenceLength_SetsValue )
    {
        auto config = LpeConfig().withMaxSequenceLength( 1024 );

        EXPECT_EQ( config.getMaxSequenceLength(), 1024u );
    }

    TEST_F( LpeConfigTests, WithVocabularyLength_SetsValue )
    {
        auto config = LpeConfig().withVocabularyLength( 50257 );

        EXPECT_EQ( config.getVocabularyLength(), 50257u );
    }

    // ====================================================================
    // A. Validation
    // ====================================================================

    TEST_F( LpeConfigTests, Validate_PassesForValid )
    {
        EXPECT_NO_THROW(
            LpeConfig().withEmbeddingDim( 64 ).withMaxSequenceLength( 128 ).withVocabularyLength( 1000 ).validate() );
    }

    TEST_F( LpeConfigTests, Validate_ThrowsForZeroEmbeddingDim )
    {
        EXPECT_THROW(
            LpeConfig().withMaxSequenceLength( 128 ).withVocabularyLength( 1000 ).validate(),
            std::invalid_argument );
    }

    TEST_F( LpeConfigTests, Validate_ThrowsForZeroMaxSequenceLength )
    {
        EXPECT_THROW(
            LpeConfig().withEmbeddingDim( 64 ).withMaxSequenceLength( 0 ).withVocabularyLength( 1000 ).validate(),
            std::invalid_argument );
    }

    TEST_F( LpeConfigTests, Validate_ThrowsForZeroVocabularyLength )
    {
        EXPECT_THROW(
            LpeConfig().withEmbeddingDim( 64 ).withMaxSequenceLength( 128 ).withVocabularyLength( 0 ).validate(),
            std::invalid_argument );
    }

    // ====================================================================
    // H. Serialization round-trip
    // ====================================================================

    TEST_F( LpeConfigTests, Metadata_RoundTripPreservesFields )
    {
        LpeConfig source;
        source.withEmbeddingDim( 768 ).withMaxSequenceLength( 1024 ).withVocabularyLength( 50257 );

        SerializationMetadata meta = source.toMetadata();

        LpeConfig loaded;
        loaded.fromMetadata( meta );

        EXPECT_EQ( loaded.getEmbeddingDim(), 768u );
        EXPECT_EQ( loaded.getMaxSequenceLength(), 1024u );
        EXPECT_EQ( loaded.getVocabularyLength(), 50257u );
    }

    // ====================================================================
    // I. Diagnostics
    // ====================================================================

    TEST_F( LpeConfigTests, ToString_NamesFields )
    {
        LpeConfig config;
        config.withEmbeddingDim( 64 ).withMaxSequenceLength( 128 ).withVocabularyLength( 1000 );

        const std::string text = config.toString();

        EXPECT_NE( text.find( "channels" ), std::string::npos );
        EXPECT_NE( text.find( "vocabulary_length" ), std::string::npos );
    }
}
