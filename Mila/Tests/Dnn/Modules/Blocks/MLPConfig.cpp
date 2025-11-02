#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <vector>
#include <cstdint>

import Mila;

namespace Modules::Blocks::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class MLPConfigTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
        }
    };

    TEST_F( MLPConfigTests, Constructor_WithInputFeaturesAndHiddenSize )
    {
        int64_t input_features = 768;
        int64_t hidden_size = 3072;

        MLPConfig config( input_features, hidden_size );

        EXPECT_EQ( config.getInputFeatures(), input_features );
        EXPECT_EQ( config.getHiddenSize(), hidden_size );
        EXPECT_TRUE( config.hasBias() );
        EXPECT_EQ( config.getActivationType(), ActivationType::Gelu );
        EXPECT_FALSE( config.useLayerNorm() );
    }

    TEST_F( MLPConfigTests, Constructor_MinimalValues )
    {
        int64_t input_features = 1;
        int64_t hidden_size = 1;

        MLPConfig config( input_features, hidden_size );

        EXPECT_EQ( config.getInputFeatures(), input_features );
        EXPECT_EQ( config.getHiddenSize(), hidden_size );
        EXPECT_TRUE( config.hasBias() );
    }

    TEST_F( MLPConfigTests, WithBias_FluentInterface )
    {
        MLPConfig config( 256, 1024 );

        auto& result = config.withBias( false );

        EXPECT_EQ( &result, &config );
        EXPECT_FALSE( config.hasBias() );

        config.withBias( true );
        EXPECT_TRUE( config.hasBias() );
    }

    TEST_F( MLPConfigTests, WithActivation_FluentInterface )
    {
        MLPConfig config( 256, 1024 );

        auto& result = config.withActivation( ActivationType::Gelu );

        EXPECT_EQ( &result, &config );
        EXPECT_EQ( config.getActivationType(), ActivationType::Gelu );
    }

    TEST_F( MLPConfigTests, WithLayerNorm_FluentInterface )
    {
        MLPConfig config( 256, 1024 );

        auto& result = config.withLayerNorm( true );

        EXPECT_EQ( &result, &config );
        EXPECT_TRUE( config.useLayerNorm() );

        config.withLayerNorm( false );
        EXPECT_FALSE( config.useLayerNorm() );
    }

    TEST_F( MLPConfigTests, FluentInterface_Chaining )
    {
        int64_t input_features = 512;
        int64_t hidden_size = 2048;

        MLPConfig config( input_features, hidden_size );

        auto& result = config
            .withBias( false )
            .withActivation( ActivationType::Gelu )
            .withLayerNorm( true )
            .withName( "test_mlp" )
            .withPrecisionPolicy( ComputePrecision::Policy::Performance )
            .withTraining( true );

        EXPECT_EQ( &result, &config );
        EXPECT_EQ( config.getInputFeatures(), input_features );
        EXPECT_EQ( config.getHiddenSize(), hidden_size );
        EXPECT_FALSE( config.hasBias() );
        EXPECT_EQ( config.getActivationType(), ActivationType::Gelu );
        EXPECT_TRUE( config.useLayerNorm() );
        EXPECT_EQ( config.getName(), "test_mlp" );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Performance );
        EXPECT_TRUE( config.isTraining() );
    }

    TEST_F( MLPConfigTests, Validation_Success )
    {
        MLPConfig config( 512, 2048 );
        config.withName( "valid_config" );

        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( MLPConfigTests, Validation_Failure_ZeroInputFeatures )
    {
        MLPConfig config( 0, 1024 );
        config.withName( "test_config" );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( MLPConfigTests, Validation_Failure_NegativeInputFeatures )
    {
        MLPConfig config( -1, 1024 );
        config.withName( "test_config" );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( MLPConfigTests, Validation_Failure_ZeroHiddenSize )
    {
        MLPConfig config( 512, 0 );
        config.withName( "test_config" );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( MLPConfigTests, Validation_Failure_NegativeHiddenSize )
    {
        MLPConfig config( 512, -1 );
        config.withName( "test_config" );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( MLPConfigTests, Validation_ErrorMessage_ZeroInputFeatures )
    {
        MLPConfig config( 0, 1024 );
        config.withName( "test_config" );

        try
        {
            config.validate();
            FAIL() << "Expected std::invalid_argument to be thrown";
        }
        catch (const std::invalid_argument& e)
        {
            std::string error_msg = e.what();
            EXPECT_TRUE( error_msg.find( "MLPConfig:" ) != std::string::npos );
            EXPECT_TRUE( error_msg.find( "Input features must be greater than zero" ) != std::string::npos );
        }
    }

    TEST_F( MLPConfigTests, Validation_ErrorMessage_ZeroHiddenSize )
    {
        MLPConfig config( 512, 0 );
        config.withName( "test_config" );

        try
        {
            config.validate();
            FAIL() << "Expected std::invalid_argument to be thrown";
        }
        catch (const std::invalid_argument& e)
        {
            std::string error_msg = e.what();
            EXPECT_TRUE( error_msg.find( "MLPConfig:" ) != std::string::npos );
            EXPECT_TRUE( error_msg.find( "Hidden size must be greater than zero" ) != std::string::npos );
        }
    }

    TEST_F( MLPConfigTests, BaseClass_Interaction )
    {
        MLPConfig config( 768, 3072 );

        config.withBias( false )
            .withActivation( ActivationType::Gelu )
            .withLayerNorm( true )
            .withName( "test_mlp" )
            .withPrecisionPolicy( ComputePrecision::Policy::Accuracy )
            .withTraining( false );

        EXPECT_EQ( config.getInputFeatures(), 768 );
        EXPECT_EQ( config.getHiddenSize(), 3072 );
        EXPECT_FALSE( config.hasBias() );
        EXPECT_EQ( config.getActivationType(), ActivationType::Gelu );
        EXPECT_TRUE( config.useLayerNorm() );
        EXPECT_EQ( config.getName(), "test_mlp" );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Accuracy );
        EXPECT_FALSE( config.isTraining() );
    }

    TEST_F( MLPConfigTests, Configuration_Persistence )
    {
        int64_t input_features = 256;
        int64_t hidden_size = 1024;

        MLPConfig config( input_features, hidden_size );
        config.withBias( false )
            .withActivation( ActivationType::Gelu )
            .withLayerNorm( true )
            .withName( "persistent_mlp" );

        MLPConfig copied_config = config;

        EXPECT_EQ( copied_config.getInputFeatures(), input_features );
        EXPECT_EQ( copied_config.getHiddenSize(), hidden_size );
        EXPECT_FALSE( copied_config.hasBias() );
        EXPECT_EQ( copied_config.getActivationType(), ActivationType::Gelu );
        EXPECT_TRUE( copied_config.useLayerNorm() );
        EXPECT_EQ( copied_config.getName(), "persistent_mlp" );
    }

    TEST_F( MLPConfigTests, EdgeCase_LargeFeatureDimensions )
    {
        int64_t large_features = 4096;
        int64_t large_hidden = 16384;

        MLPConfig config( large_features, large_hidden );

        EXPECT_EQ( config.getInputFeatures(), large_features );
        EXPECT_EQ( config.getHiddenSize(), large_hidden );
    }

    TEST_F( MLPConfigTests, EdgeCase_SingleFeature )
    {
        MLPConfig config( 1, 1 );

        EXPECT_EQ( config.getInputFeatures(), 1 );
        EXPECT_EQ( config.getHiddenSize(), 1 );
        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( MLPConfigTests, MultipleConfiguration_Updates )
    {
        MLPConfig config( 512, 2048 );

        config.withBias( false );
        EXPECT_FALSE( config.hasBias() );

        config.withBias( true );
        EXPECT_TRUE( config.hasBias() );

        config.withLayerNorm( true );
        EXPECT_TRUE( config.useLayerNorm() );

        config.withLayerNorm( false );
        EXPECT_FALSE( config.useLayerNorm() );
    }

    TEST_F( MLPConfigTests, TypicalTransformer_MLPConfiguration )
    {
        int64_t transformer_features = 768;
        int64_t ffn_hidden_size = 3072;

        MLPConfig config( transformer_features, ffn_hidden_size );
        config.withBias( true )
            .withActivation( ActivationType::Gelu )
            .withLayerNorm( false )
            .withName( "transformer_ffn" )
            .withTraining( true );

        EXPECT_EQ( config.getInputFeatures(), transformer_features );
        EXPECT_EQ( config.getHiddenSize(), ffn_hidden_size );
        EXPECT_TRUE( config.hasBias() );
        EXPECT_EQ( config.getActivationType(), ActivationType::Gelu );
        EXPECT_FALSE( config.useLayerNorm() );
        EXPECT_EQ( config.getName(), "transformer_ffn" );
        EXPECT_TRUE( config.isTraining() );

        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( MLPConfigTests, Default_Validation )
    {
        MLPConfig config( 768, 3072 );

        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( MLPConfigTests, MethodChain_Preservation )
    {
        int64_t original_features = 64;
        int64_t original_hidden = 256;

        MLPConfig config( original_features, original_hidden );
        config.withBias( false )
            .withLayerNorm( true );

        config.withName( "chained_config" );

        EXPECT_EQ( config.getInputFeatures(), original_features );
        EXPECT_EQ( config.getHiddenSize(), original_hidden );
        EXPECT_FALSE( config.hasBias() );
        EXPECT_TRUE( config.useLayerNorm() );
        EXPECT_EQ( config.getName(), "chained_config" );
    }

    TEST_F( MLPConfigTests, GPT2_Style_Configuration )
    {
        // GPT-2 small: 768 ? 3072
        MLPConfig gpt2_small( 768, 3072 );
        gpt2_small.withBias( true )
            .withActivation( ActivationType::Gelu )
            .withName( "gpt2_small_mlp" );

        EXPECT_EQ( gpt2_small.getInputFeatures(), 768 );
        EXPECT_EQ( gpt2_small.getHiddenSize(), 3072 );
        EXPECT_TRUE( gpt2_small.hasBias() );
        EXPECT_NO_THROW( gpt2_small.validate() );
    }

    TEST_F( MLPConfigTests, Getter_Noexcept )
    {
        MLPConfig config( 512, 2048 );

        EXPECT_TRUE( noexcept(config.getInputFeatures()) );
        EXPECT_TRUE( noexcept(config.getHiddenSize()) );
        EXPECT_TRUE( noexcept(config.hasBias()) );
        EXPECT_TRUE( noexcept(config.getActivationType()) );
        EXPECT_TRUE( noexcept(config.useLayerNorm()) );
    }
}