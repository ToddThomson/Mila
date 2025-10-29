#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <vector>

import Mila;

namespace Modules::Blocks::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class MLPConfigTests : public ::testing::Test {
    protected:
        void SetUp() override {}
    };

    TEST_F( MLPConfigTests, ConstructorWithInputShapeAndHiddenSize ) {
        shape_t input_shape = { 32, 128, 768 };
        size_t hidden_size = 3072;

        MLPConfig config( input_shape, hidden_size );

        EXPECT_EQ( config.getInputShape(), input_shape );
        EXPECT_EQ( config.getInputFeatures(), 768 );
        EXPECT_EQ( config.getHiddenSize(), hidden_size );
        EXPECT_TRUE( config.hasBias() );
        EXPECT_EQ( config.getActivationType(), ActivationType::Gelu );
        EXPECT_FALSE( config.useLayerNorm() );
    }

    TEST_F( MLPConfigTests, ConstructorWithEmptyInputShape ) {
        shape_t empty_shape = {};
        size_t hidden_size = 1024;

        MLPConfig config( empty_shape, hidden_size );

        EXPECT_EQ( config.getInputShape(), empty_shape );
        EXPECT_EQ( config.getInputFeatures(), 0 );
        EXPECT_EQ( config.getHiddenSize(), hidden_size );
    }

    TEST_F( MLPConfigTests, AlternativeConstructorWithInputFeatures ) {
        size_t input_features = 512;
        size_t hidden_size = 2048;

        MLPConfig config( input_features, hidden_size );

        EXPECT_EQ( config.getInputShape(), shape_t{ input_features } );
        EXPECT_EQ( config.getInputFeatures(), input_features );
        EXPECT_EQ( config.getHiddenSize(), hidden_size );
        EXPECT_TRUE( config.hasBias() );
    }

    TEST_F( MLPConfigTests, WithBias ) {
        MLPConfig config( 256, 1024 );

        auto& result = config.withBias( false );

        EXPECT_EQ( &result, &config );
        EXPECT_FALSE( config.hasBias() );

        config.withBias( true );
        EXPECT_TRUE( config.hasBias() );
    }

    TEST_F( MLPConfigTests, WithActivation ) {
        MLPConfig config( 256, 1024 );

        auto& result = config.withActivation( ActivationType::Gelu );

        EXPECT_EQ( &result, &config );
        EXPECT_EQ( config.getActivationType(), ActivationType::Gelu );
    }

    TEST_F( MLPConfigTests, WithLayerNorm ) {
        MLPConfig config( 256, 1024 );

        auto& result = config.withLayerNorm( true );

        EXPECT_EQ( &result, &config );
        EXPECT_TRUE( config.useLayerNorm() );

        config.withLayerNorm( false );
        EXPECT_FALSE( config.useLayerNorm() );
    }

    TEST_F( MLPConfigTests, FluentInterfaceChaining ) {
        shape_t input_shape = { 16, 64, 512 };
        size_t hidden_size = 2048;

        MLPConfig config( input_shape, hidden_size );

        auto& result = config
            .withBias( false )
            .withActivation( ActivationType::Gelu )
            .withLayerNorm( true )
            .withName( "test_mlp" )
            .withPrecisionPolicy( ComputePrecision::Policy::Performance )
            .withTraining( true );

        EXPECT_EQ( &result, &config );
        EXPECT_EQ( config.getInputShape(), input_shape );
        EXPECT_EQ( config.getInputFeatures(), 512 );
        EXPECT_EQ( config.getHiddenSize(), hidden_size );
        EXPECT_FALSE( config.hasBias() );
        EXPECT_EQ( config.getActivationType(), ActivationType::Gelu );
        EXPECT_TRUE( config.useLayerNorm() );
        EXPECT_EQ( config.getName(), "test_mlp" );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Performance );
        EXPECT_TRUE( config.isTraining() );
    }

    TEST_F( MLPConfigTests, ValidationSuccess ) {
        MLPConfig config( 512, 2048 );
        config.withName( "valid_config" );

        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( MLPConfigTests, ValidationFailure_ZeroInputFeatures ) {
        shape_t empty_shape = {};
        MLPConfig config( empty_shape, 1024 );
        config.withName( "test_config" );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( MLPConfigTests, ValidationFailure_ZeroHiddenSize ) {
        MLPConfig config( 512, 0 );
        config.withName( "test_config" );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( MLPConfigTests, ValidationErrorMessage_ZeroInputFeatures ) {
        shape_t empty_shape = {};
        MLPConfig config( empty_shape, 1024 );
        config.withName( "test_config" );

        try {
            config.validate();
            FAIL() << "Expected std::invalid_argument to be thrown";
        }
        catch ( const std::invalid_argument& e ) {
            std::string error_msg = e.what();
            EXPECT_TRUE( error_msg.find( "Input features must be greater than zero" ) != std::string::npos );
        }
    }

    TEST_F( MLPConfigTests, ValidationErrorMessage_ZeroHiddenSize ) {
        MLPConfig config( 512, 0 );
        config.withName( "test_config" );

        try {
            config.validate();
            FAIL() << "Expected std::invalid_argument to be thrown";
        }
        catch ( const std::invalid_argument& e ) {
            std::string error_msg = e.what();
            EXPECT_TRUE( error_msg.find( "Hidden size must be greater than zero" ) != std::string::npos );
        }
    }

    TEST_F( MLPConfigTests, BaseClassInteraction ) {
        shape_t input_shape = { 32, 512, 768 };
        MLPConfig config( input_shape, 3072 );

        config.withBias( false )
            .withActivation( ActivationType::Gelu )
            .withLayerNorm( true )
            .withName( "test_mlp" )
            .withPrecisionPolicy( ComputePrecision::Policy::Accuracy )
            .withTraining( false );

        EXPECT_EQ( config.getInputShape(), input_shape );
        EXPECT_EQ( config.getInputFeatures(), 768 );
        EXPECT_EQ( config.getHiddenSize(), 3072 );
        EXPECT_FALSE( config.hasBias() );
        EXPECT_EQ( config.getActivationType(), ActivationType::Gelu );
        EXPECT_TRUE( config.useLayerNorm() );
        EXPECT_EQ( config.getName(), "test_mlp" );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Accuracy );
        EXPECT_FALSE( config.isTraining() );
    }

    TEST_F( MLPConfigTests, ConfigurationPersistence ) {
        shape_t input_shape = { 8, 32, 256 };
        size_t hidden_size = 1024;

        MLPConfig config( input_shape, hidden_size );
        config.withBias( false )
            .withActivation( ActivationType::Gelu )
            .withLayerNorm( true )
            .withName( "persistent_mlp" );

        MLPConfig copied_config = config;

        EXPECT_EQ( copied_config.getInputShape(), input_shape );
        EXPECT_EQ( copied_config.getInputFeatures(), 256 );
        EXPECT_EQ( copied_config.getHiddenSize(), hidden_size );
        EXPECT_FALSE( copied_config.hasBias() );
        EXPECT_EQ( copied_config.getActivationType(), ActivationType::Gelu );
        EXPECT_TRUE( copied_config.useLayerNorm() );
        EXPECT_EQ( copied_config.getName(), "persistent_mlp" );
    }

    TEST_F( MLPConfigTests, EdgeCases_LargeInputShape ) {
        shape_t large_shape = { 128, 2048, 4096 };
        size_t hidden_size = 16384;

        MLPConfig config( large_shape, hidden_size );

        EXPECT_EQ( config.getInputShape(), large_shape );
        EXPECT_EQ( config.getInputFeatures(), 4096 );
        EXPECT_EQ( config.getHiddenSize(), hidden_size );
    }

    TEST_F( MLPConfigTests, EdgeCases_SingleInputFeature ) {
        MLPConfig config( 1, 1 );

        EXPECT_EQ( config.getInputFeatures(), 1 );
        EXPECT_EQ( config.getHiddenSize(), 1 );
        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( MLPConfigTests, EdgeCases_HighDimensionalInputShape ) {
        shape_t high_dim_shape = { 2, 4, 8, 16, 32 };
        size_t hidden_size = 128;

        MLPConfig config( high_dim_shape, hidden_size );

        EXPECT_EQ( config.getInputShape(), high_dim_shape );
        EXPECT_EQ( config.getInputFeatures(), 32 );
        EXPECT_EQ( config.getHiddenSize(), hidden_size );
    }

    TEST_F( MLPConfigTests, MultipleConfigurationUpdates ) {
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

    TEST_F( MLPConfigTests, TypicalTransformerMLPConfiguration ) {
        shape_t transformer_shape = { 32, 512, 768 };
        size_t ffn_hidden_size = 3072;

        MLPConfig config( transformer_shape, ffn_hidden_size );
        config.withBias( true )
            .withActivation( ActivationType::Gelu )
            .withLayerNorm( false )
            .withName( "transformer_ffn" )
            .withTraining( true );

        EXPECT_EQ( config.getInputShape(), transformer_shape );
        EXPECT_EQ( config.getInputFeatures(), 768 );
        EXPECT_EQ( config.getHiddenSize(), ffn_hidden_size );
        EXPECT_TRUE( config.hasBias() );
        EXPECT_EQ( config.getActivationType(), ActivationType::Gelu );
        EXPECT_FALSE( config.useLayerNorm() );
        EXPECT_EQ( config.getName(), "transformer_ffn" );
        EXPECT_TRUE( config.isTraining() );

        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( MLPConfigTests, DefaultValidation ) {
        MLPConfig config( 768, 3072 );

        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( MLPConfigTests, MethodChainPreservation ) {
        shape_t original_shape = { 16, 32, 64 };
        size_t original_hidden = 256;

        MLPConfig config( original_shape, original_hidden );
        config.withBias( false )
            .withLayerNorm( true );

        config.withName( "chained_config" );

        EXPECT_EQ( config.getInputShape(), original_shape );
        EXPECT_EQ( config.getInputFeatures(), 64 );
        EXPECT_EQ( config.getHiddenSize(), original_hidden );
        EXPECT_FALSE( config.hasBias() );
        EXPECT_TRUE( config.useLayerNorm() );
        EXPECT_EQ( config.getName(), "chained_config" );
    }
}