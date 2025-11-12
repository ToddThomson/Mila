#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>

import Mila;

namespace Modules::Layers::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class EncoderConfigTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
        }
    };

    TEST_F( EncoderConfigTests, FluentSettersAndAccessors )
    {
        EncoderConfig config;
        auto& result = config
            .withChannels( 256 )
            .withMaxSequenceLength( 1024 )
            .withVocabularyLength( 50257 )
            .withName( "test_encoder" );

        EXPECT_EQ( &result, &config );
        EXPECT_EQ( config.getChannels(), 256u );
        EXPECT_EQ( config.getMaxSequenceLength(), 1024u );
        EXPECT_EQ( config.getVocabularyLength(), 50257u );
        EXPECT_EQ( config.getName(), "test_encoder" );
    }

    TEST_F( EncoderConfigTests, ValidationSuccess )
    {
        EncoderConfig config;
        config.withChannels( 128 )
            .withMaxSequenceLength( 512 )
            .withVocabularyLength( 10000 )
            .withName( "valid_encoder" );

        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( EncoderConfigTests, ValidationFailure_ZeroChannels )
    {
        EncoderConfig config;
        config.withChannels( 0 )
            .withMaxSequenceLength( 512 )
            .withVocabularyLength( 10000 )
            .withName( "bad_channels" );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( EncoderConfigTests, ValidationFailure_ZeroMaxSequenceLength )
    {
        EncoderConfig config;
        config.withChannels( 128 )
            .withMaxSequenceLength( 0 )
            .withVocabularyLength( 10000 )
            .withName( "bad_max_seq" );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( EncoderConfigTests, ValidationFailure_ZeroVocabularyLength )
    {
        EncoderConfig config;
        config.withChannels( 128 )
            .withMaxSequenceLength( 512 )
            .withVocabularyLength( 0 )
            .withName( "bad_vocab" );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( EncoderConfigTests, ValidationErrorMessage_Channels )
    {
        EncoderConfig config;
        config.withChannels( 0 )
            .withMaxSequenceLength( 16 )
            .withVocabularyLength( 100 )
            .withName( "err_msg" );

        try
        {
            config.validate();
            FAIL() << "Expected std::invalid_argument to be thrown";
        }
        catch (const std::invalid_argument& e)
        {
            std::string msg = e.what();
            EXPECT_NE( msg.find( "Embedding dimension" ), std::string::npos );
        }
    }

    TEST_F( EncoderConfigTests, ConfigurationPersistence_Copy )
    {
        EncoderConfig config;
        config.withChannels( 64 )
            .withMaxSequenceLength( 256 )
            .withVocabularyLength( 20000 )
            .withName( "persistent_encoder" );

        EncoderConfig copy = config;

        EXPECT_EQ( copy.getChannels(), 64u );
        EXPECT_EQ( copy.getMaxSequenceLength(), 256u );
        EXPECT_EQ( copy.getVocabularyLength(), 20000u );
        EXPECT_EQ( copy.getName(), "persistent_encoder" );
    }

    TEST_F( EncoderConfigTests, EdgeCases_LargeDimensions )
    {
        EncoderConfig config;
        config.withChannels( 4096 )
            .withMaxSequenceLength( 16384 )
            .withVocabularyLength( 1000000 )
            .withName( "large_encoder" );

        EXPECT_EQ( config.getChannels(), 4096u );
        EXPECT_EQ( config.getMaxSequenceLength(), 16384u );
        EXPECT_EQ( config.getVocabularyLength(), 1000000u );
        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( EncoderConfigTests, EdgeCases_MinimalDimensions )
    {
        EncoderConfig config;
        config.withChannels( 1 )
            .withMaxSequenceLength( 1 )
            .withVocabularyLength( 1 )
            .withName( "minimal_encoder" );

        EXPECT_EQ( config.getChannels(), 1u );
        EXPECT_EQ( config.getMaxSequenceLength(), 1u );
        EXPECT_EQ( config.getVocabularyLength(), 1u );
        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( EncoderConfigTests, DefaultValuesBehavior )
    {
        EncoderConfig config; // defaults: channels=0, max_seq_len=512, vocab_len=50000

        EXPECT_EQ( config.getChannels(), 0u );
        EXPECT_EQ( config.getMaxSequenceLength(), 512u );
        EXPECT_EQ( config.getVocabularyLength(), 50000u );

        // Default config should fail validation because channels default to 0
        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( EncoderConfigTests, MethodChainPreservation )
    {
        EncoderConfig config;
        config.withChannels( 128 )
            .withMaxSequenceLength( 256 )
            .withVocabularyLength( 1000 );

        // chaining again and ensuring values persist / update correctly
        config.withChannels( 256 ).withName( "chain_preserve" );

        EXPECT_EQ( config.getChannels(), 256u );
        EXPECT_EQ( config.getMaxSequenceLength(), 256u );
        EXPECT_EQ( config.getVocabularyLength(), 1000u );
        EXPECT_EQ( config.getName(), "chain_preserve" );
    }
}