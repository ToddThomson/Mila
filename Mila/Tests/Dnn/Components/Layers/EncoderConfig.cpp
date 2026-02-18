#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <string>

import Mila;

namespace Components::Layers::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class EncoderConfigTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {}
    };

    TEST_F( EncoderConfigTests, FluentSettersAndAccessors )
    {
        LearnedEncoderConfig config;
        auto& result = config
            .withEmbeddingDim( 256 )
            .withMaxSequenceLength( 1024 )
            .withVocabularyLength( 50257 );

        EXPECT_EQ( &result, &config );
        EXPECT_EQ( config.getEmbeddingDim(), 256u );
        EXPECT_EQ( config.getMaxSequenceLength(), 1024u );
        EXPECT_EQ( config.getVocabularyLength(), 50257u );
    }

    TEST_F( EncoderConfigTests, ValidationSuccess )
    {
        LearnedEncoderConfig config;
        config.withEmbeddingDim( 128 )
            .withMaxSequenceLength( 512 )
            .withVocabularyLength( 10000 );

        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( EncoderConfigTests, ValidationFailure_ZeroChannels )
    {
        LearnedEncoderConfig config;
        config.withEmbeddingDim( 0 )
            .withMaxSequenceLength( 512 )
            .withVocabularyLength( 10000 );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( EncoderConfigTests, ValidationFailure_ZeroMaxSequenceLength )
    {
        LearnedEncoderConfig config;
        config.withEmbeddingDim( 128 )
            .withMaxSequenceLength( 0 )
            .withVocabularyLength( 10000 );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( EncoderConfigTests, ValidationFailure_ZeroVocabularyLength )
    {
        LearnedEncoderConfig config;
        config.withEmbeddingDim( 128 )
            .withMaxSequenceLength( 512 )
            .withVocabularyLength( 0 );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( EncoderConfigTests, ValidationErrorMessage_Channels )
    {
        LearnedEncoderConfig config;
        config.withEmbeddingDim( 0 )
            .withMaxSequenceLength( 16 )
            .withVocabularyLength( 100 );

        try
        {
            config.validate();
            FAIL() << "Expected std::invalid_argument to be thrown";
        }
        catch ( const std::invalid_argument& e )
        {
            std::string msg = e.what();
            EXPECT_NE( msg.find( "channels" ), std::string::npos );
        }
    }

    TEST_F( EncoderConfigTests, ConfigurationPersistence_Copy )
    {
        LearnedEncoderConfig config;
        config.withEmbeddingDim( 64 )
            .withMaxSequenceLength( 256 )
            .withVocabularyLength( 20000 );

        LearnedEncoderConfig copy = config;

        EXPECT_EQ( copy.getEmbeddingDim(), 64u );
        EXPECT_EQ( copy.getMaxSequenceLength(), 256u );
        EXPECT_EQ( copy.getVocabularyLength(), 20000u );
    }

    TEST_F( EncoderConfigTests, ConfigurationPersistence_Metadata_RoundTrip )
    {
        LearnedEncoderConfig config;
        config.withEmbeddingDim( 512 )
            .withMaxSequenceLength( 2048 )
            .withVocabularyLength( 30000 )
            .withPrecisionPolicy( ComputePrecision::Policy::Performance );

        auto meta = config.toMetadata();

        LearnedEncoderConfig restored;
        restored.fromMetadata( meta );

        EXPECT_EQ( restored.getEmbeddingDim(), 512u );
        EXPECT_EQ( restored.getMaxSequenceLength(), 2048u );
        EXPECT_EQ( restored.getVocabularyLength(), 30000u );
        EXPECT_EQ( restored.getPrecisionPolicy(), ComputePrecision::Policy::Performance );
    }

    TEST_F( EncoderConfigTests, EdgeCases_LargeDimensions )
    {
        LearnedEncoderConfig config;
        config.withEmbeddingDim( 4096 )
            .withMaxSequenceLength( 16384 )
            .withVocabularyLength( 1000000 );

        EXPECT_EQ( config.getEmbeddingDim(), 4096u );
        EXPECT_EQ( config.getMaxSequenceLength(), 16384u );
        EXPECT_EQ( config.getVocabularyLength(), 1000000u );
        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( EncoderConfigTests, EdgeCases_MinimalDimensions )
    {
        LearnedEncoderConfig config;
        config.withEmbeddingDim( 1 )
            .withMaxSequenceLength( 1 )
            .withVocabularyLength( 1 );

        EXPECT_EQ( config.getEmbeddingDim(), 1u );
        EXPECT_EQ( config.getMaxSequenceLength(), 1u );
        EXPECT_EQ( config.getVocabularyLength(), 1u );
        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( EncoderConfigTests, DefaultValuesBehavior )
    {
        LearnedEncoderConfig config; // defaults: channels=0, max_seq_len=512, vocab_len=50000

        EXPECT_EQ( config.getEmbeddingDim(), 0u );
        EXPECT_EQ( config.getMaxSequenceLength(), 512u );
        EXPECT_EQ( config.getVocabularyLength(), 50000u );

        // Default config should fail validation because channels default to 0
        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( EncoderConfigTests, MethodChainPreservation )
    {
        LearnedEncoderConfig config;
        config.withEmbeddingDim( 128 )
            .withMaxSequenceLength( 256 )
            .withVocabularyLength( 1000 );

        // chaining again and ensuring values persist / update correctly
        config.withEmbeddingDim( 256 );

        EXPECT_EQ( config.getEmbeddingDim(), 256u );
        EXPECT_EQ( config.getMaxSequenceLength(), 256u );
        EXPECT_EQ( config.getVocabularyLength(), 1000u );
    }
}