#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>

import Mila;

namespace Modules::Layers::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class LinearConfigTests : public ::testing::Test {
    protected:
        void SetUp() override {}
    };

    TEST_F( LinearConfigTests, ConstructorWithRequiredParameters ) {
        size_t input_features = 128;
        size_t output_features = 64;

        LinearConfig config( input_features, output_features );

        EXPECT_EQ( config.getInputFeatures(), input_features );
        EXPECT_EQ( config.getOutputFeatures(), output_features );
        EXPECT_TRUE( config.hasBias() );
    }

    TEST_F( LinearConfigTests, WithBias ) {
        LinearConfig config( 128, 64 );

        auto& result = config.withBias( false );

        EXPECT_EQ( &result, &config );
        EXPECT_FALSE( config.hasBias() );

        config.withBias( true );
        EXPECT_TRUE( config.hasBias() );
    }

    TEST_F( LinearConfigTests, FluentInterfaceChaining ) {
        size_t input_features = 256;
        size_t output_features = 128;

        LinearConfig config( input_features, output_features );

        auto& result = config
            .withBias( false )
            .withName( "test_linear" )
            .withPrecisionPolicy( ComputePrecision::Policy::Performance )
            .withTraining( true );

        EXPECT_EQ( &result, &config );
        EXPECT_EQ( config.getInputFeatures(), input_features );
        EXPECT_EQ( config.getOutputFeatures(), output_features );
        EXPECT_FALSE( config.hasBias() );
        EXPECT_EQ( config.getName(), "test_linear" );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Performance );
        EXPECT_TRUE( config.isTraining() );
    }

    TEST_F( LinearConfigTests, ValidationSuccess ) {
        LinearConfig config( 128, 64 );
        config.withName( "valid_config" );

        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( LinearConfigTests, ValidationFailure_ZeroInputFeatures ) {
        LinearConfig config( 0, 64 );
        config.withName( "test_config" );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( LinearConfigTests, ValidationFailure_ZeroOutputFeatures ) {
        LinearConfig config( 128, 0 );
        config.withName( "test_config" );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( LinearConfigTests, ValidationFailure_BothZeroFeatures ) {
        LinearConfig config( 0, 0 );
        config.withName( "test_config" );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( LinearConfigTests, ValidationErrorMessage_ZeroFeatures ) {
        LinearConfig config( 0, 64 );
        config.withName( "test_config" );

        try {
            config.validate();
            FAIL() << "Expected std::invalid_argument to be thrown";
        }
        catch ( const std::invalid_argument& e ) {
            std::string error_msg = e.what();
            EXPECT_TRUE( error_msg.find( "Input and output features must be greater than zero" ) != std::string::npos );
        }
    }

    TEST_F( LinearConfigTests, BaseClassInteraction ) {
        LinearConfig config( 512, 256 );

        config.withBias( false )
            .withName( "test_linear" )
            .withPrecisionPolicy( ComputePrecision::Policy::Accuracy )
            .withTraining( false );

        EXPECT_EQ( config.getInputFeatures(), 512 );
        EXPECT_EQ( config.getOutputFeatures(), 256 );
        EXPECT_FALSE( config.hasBias() );
        EXPECT_EQ( config.getName(), "test_linear" );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Accuracy );
        EXPECT_FALSE( config.isTraining() );
    }

    TEST_F( LinearConfigTests, ConfigurationPersistence ) {
        size_t input_features = 1024;
        size_t output_features = 512;

        LinearConfig config( input_features, output_features );
        config.withBias( false )
            .withName( "persistent_linear" );

        LinearConfig copied_config = config;

        EXPECT_EQ( copied_config.getInputFeatures(), input_features );
        EXPECT_EQ( copied_config.getOutputFeatures(), output_features );
        EXPECT_FALSE( copied_config.hasBias() );
        EXPECT_EQ( copied_config.getName(), "persistent_linear" );
    }

    TEST_F( LinearConfigTests, EdgeCases_LargeFeatureDimensions ) {
        size_t large_input = 4096;
        size_t large_output = 8192;

        LinearConfig config( large_input, large_output );

        EXPECT_EQ( config.getInputFeatures(), large_input );
        EXPECT_EQ( config.getOutputFeatures(), large_output );
        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( LinearConfigTests, EdgeCases_SingleFeatureDimensions ) {
        LinearConfig config( 1, 1 );

        EXPECT_EQ( config.getInputFeatures(), 1 );
        EXPECT_EQ( config.getOutputFeatures(), 1 );
        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( LinearConfigTests, EdgeCases_AsymmetricFeatures ) {
        LinearConfig small_to_large( 8, 1024 );
        LinearConfig large_to_small( 2048, 16 );

        EXPECT_EQ( small_to_large.getInputFeatures(), 8 );
        EXPECT_EQ( small_to_large.getOutputFeatures(), 1024 );

        EXPECT_EQ( large_to_small.getInputFeatures(), 2048 );
        EXPECT_EQ( large_to_small.getOutputFeatures(), 16 );

        EXPECT_NO_THROW( small_to_large.validate() );
        EXPECT_NO_THROW( large_to_small.validate() );
    }

    TEST_F( LinearConfigTests, MultipleBiasToggle ) {
        LinearConfig config( 128, 64 );

        EXPECT_TRUE( config.hasBias() );

        config.withBias( false );
        EXPECT_FALSE( config.hasBias() );

        config.withBias( true );
        EXPECT_TRUE( config.hasBias() );

        config.withBias( false );
        EXPECT_FALSE( config.hasBias() );
    }

    TEST_F( LinearConfigTests, TypicalNeuralNetworkConfigurations ) {
        LinearConfig hidden_layer( 768, 3072 );
        hidden_layer.withName( "transformer_ffn_hidden" )
            .withBias( true )
            .withTraining( true );

        EXPECT_EQ( hidden_layer.getInputFeatures(), 768 );
        EXPECT_EQ( hidden_layer.getOutputFeatures(), 3072 );
        EXPECT_TRUE( hidden_layer.hasBias() );
        EXPECT_EQ( hidden_layer.getName(), "transformer_ffn_hidden" );
        EXPECT_TRUE( hidden_layer.isTraining() );
        EXPECT_NO_THROW( hidden_layer.validate() );

        LinearConfig output_layer( 3072, 768 );
        output_layer.withName( "transformer_ffn_output" )
            .withBias( true )
            .withTraining( true );

        EXPECT_EQ( output_layer.getInputFeatures(), 3072 );
        EXPECT_EQ( output_layer.getOutputFeatures(), 768 );
        EXPECT_TRUE( output_layer.hasBias() );
        EXPECT_EQ( output_layer.getName(), "transformer_ffn_output" );
        EXPECT_TRUE( output_layer.isTraining() );
        EXPECT_NO_THROW( output_layer.validate() );
    }

    TEST_F( LinearConfigTests, ClassificationHeadConfiguration ) {
        LinearConfig classification_head( 768, 1000 );
        classification_head.withName( "classification_head" )
            .withBias( false )
            .withPrecisionPolicy( ComputePrecision::Policy::Accuracy )
            .withTraining( false );

        EXPECT_EQ( classification_head.getInputFeatures(), 768 );
        EXPECT_EQ( classification_head.getOutputFeatures(), 1000 );
        EXPECT_FALSE( classification_head.hasBias() );
        EXPECT_EQ( classification_head.getName(), "classification_head" );
        EXPECT_EQ( classification_head.getPrecisionPolicy(), ComputePrecision::Policy::Accuracy );
        EXPECT_FALSE( classification_head.isTraining() );
        EXPECT_NO_THROW( classification_head.validate() );
    }

    TEST_F( LinearConfigTests, EmbeddingProjectionConfiguration ) {
        LinearConfig embedding_projection( 50257, 768 );
        embedding_projection.withName( "token_embedding_projection" )
            .withBias( false )
            .withPrecisionPolicy( ComputePrecision::Policy::Performance );

        EXPECT_EQ( embedding_projection.getInputFeatures(), 50257 );
        EXPECT_EQ( embedding_projection.getOutputFeatures(), 768 );
        EXPECT_FALSE( embedding_projection.hasBias() );
        EXPECT_EQ( embedding_projection.getName(), "token_embedding_projection" );
        EXPECT_EQ( embedding_projection.getPrecisionPolicy(), ComputePrecision::Policy::Performance );
        EXPECT_NO_THROW( embedding_projection.validate() );
    }

    TEST_F( LinearConfigTests, MethodChainPreservation ) {
        LinearConfig config( 256, 128 );
        size_t original_input = config.getInputFeatures();
        size_t original_output = config.getOutputFeatures();

        config.withBias( false )
            .withName( "chained_config" );

        config.withBias( true );

        EXPECT_EQ( config.getInputFeatures(), original_input );
        EXPECT_EQ( config.getOutputFeatures(), original_output );
        EXPECT_TRUE( config.hasBias() );
        EXPECT_EQ( config.getName(), "chained_config" );
    }

    TEST_F( LinearConfigTests, DefaultValidation ) {
        LinearConfig config( 128, 64 );

        EXPECT_NO_THROW( config.validate() );
    }
}