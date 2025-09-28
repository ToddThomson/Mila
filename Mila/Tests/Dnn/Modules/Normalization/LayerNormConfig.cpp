#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <vector>

import Mila;

namespace Modules::Normalization::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class LayerNormConfigTests : public ::testing::Test {
    protected:
        void SetUp() override {}
    };

    TEST_F( LayerNormConfigTests, DefaultConstructor ) {
        LayerNormConfig config;

        EXPECT_TRUE( config.getInputShape().empty() );
        EXPECT_EQ( config.getAxis(), -1 );
        EXPECT_TRUE( config.hasBias() );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-5f );
    }

    TEST_F( LayerNormConfigTests, NormalizedDimensionConstructor ) {
        size_t normalized_dim = 768;
        LayerNormConfig config( normalized_dim );

        std::vector<size_t> expected_shape = { 1, 1, normalized_dim };
        EXPECT_EQ( config.getInputShape(), expected_shape );
        EXPECT_EQ( config.getAxis(), -1 );
        EXPECT_TRUE( config.hasBias() );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-5f );
    }

    TEST_F( LayerNormConfigTests, WithInputShape ) {
        LayerNormConfig config;
        std::vector<size_t> input_shape = { 32, 128, 768 };

        auto& result = config.withInputShape( input_shape );

        EXPECT_EQ( &result, &config );
        EXPECT_EQ( config.getInputShape(), input_shape );
    }

    TEST_F( LayerNormConfigTests, WithAxis ) {
        LayerNormConfig config;

        auto& result = config.withAxis( 2 );

        EXPECT_EQ( &result, &config );
        EXPECT_EQ( config.getAxis(), 2 );

        config.withAxis( -2 );
        EXPECT_EQ( config.getAxis(), -2 );

        config.withAxis( 0 );
        EXPECT_EQ( config.getAxis(), 0 );
    }

    TEST_F( LayerNormConfigTests, WithBias ) {
        LayerNormConfig config;

        auto& result = config.withBias( false );

        EXPECT_EQ( &result, &config );
        EXPECT_FALSE( config.hasBias() );

        config.withBias( true );
        EXPECT_TRUE( config.hasBias() );
    }

    TEST_F( LayerNormConfigTests, WithEpsilon ) {
        LayerNormConfig config;

        auto& result = config.withEpsilon( 1e-6f );

        EXPECT_EQ( &result, &config );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-6f );

        config.withEpsilon( 1e-3f );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-3f );
    }

    TEST_F( LayerNormConfigTests, FluentInterfaceChaining ) {
        LayerNormConfig config;
        std::vector<size_t> input_shape = { 16, 64, 512 };

        auto& result = config
            .withInputShape( input_shape )
            .withAxis( 2 )
            .withBias( false )
            .withEpsilon( 1e-4f )
            .withName( "test_layernorm" )
            .withTraining( true );

        EXPECT_EQ( &result, &config );
        EXPECT_EQ( config.getInputShape(), input_shape );
        EXPECT_EQ( config.getAxis(), 2 );
        EXPECT_FALSE( config.hasBias() );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-4f );
        EXPECT_EQ( config.getName(), "test_layernorm" );
        EXPECT_TRUE( config.isTraining() );
    }

    TEST_F( LayerNormConfigTests, ValidationSuccess ) {
        LayerNormConfig config;
        config.withInputShape( { 32, 128, 768 } )
            .withEpsilon( 1e-5f )
            .withName( "valid_config" );

        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( LayerNormConfigTests, ValidationFailure_EmptyInputShape ) {
        LayerNormConfig config;
        config.withName( "test_config" );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( LayerNormConfigTests, ValidationFailure_ZeroEpsilon ) {
        LayerNormConfig config;
        config.withInputShape( { 32, 128, 768 } )
            .withEpsilon( 0.0f )
            .withName( "test_config" );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( LayerNormConfigTests, ValidationFailure_NegativeEpsilon ) {
        LayerNormConfig config;
        config.withInputShape( { 32, 128, 768 } )
            .withEpsilon( -1e-5f )
            .withName( "test_config" );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( LayerNormConfigTests, ValidationErrorMessage_EmptyShape ) {
        LayerNormConfig config;
        config.withName( "test_config" );

        try {
            config.validate();
            FAIL() << "Expected std::invalid_argument to be thrown";
        }
        catch ( const std::invalid_argument& e ) {
            std::string error_msg = e.what();
            EXPECT_TRUE( error_msg.find( "Input shape cannot be empty" ) != std::string::npos );
        }
    }

    TEST_F( LayerNormConfigTests, ValidationErrorMessage_InvalidEpsilon ) {
        LayerNormConfig config;
        config.withInputShape( { 32, 128, 768 } )
            .withEpsilon( 0.0f )
            .withName( "test_config" );

        try {
            config.validate();
            FAIL() << "Expected std::invalid_argument to be thrown";
        }
        catch ( const std::invalid_argument& e ) {
            std::string error_msg = e.what();
            EXPECT_TRUE( error_msg.find( "Epsilon must be a positive value" ) != std::string::npos );
        }
    }

    TEST_F( LayerNormConfigTests, BaseClassInteraction ) {
        LayerNormConfig config;

        config.withInputShape( { 32, 128, 768 } )
            .withAxis( -1 )
            .withBias( true )
            .withEpsilon( 1e-5f )
            .withName( "test_layernorm" )
            .withPrecisionPolicy( ComputePrecision::Policy::Performance )
            .withTraining( false );

        EXPECT_EQ( config.getInputShape(), std::vector<size_t>( { 32, 128, 768 } ) );
        EXPECT_EQ( config.getAxis(), -1 );
        EXPECT_TRUE( config.hasBias() );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-5f );
        EXPECT_EQ( config.getName(), "test_layernorm" );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Performance );
        EXPECT_FALSE( config.isTraining() );
    }

    TEST_F( LayerNormConfigTests, ConfigurationPersistence ) {
        LayerNormConfig config;
        std::vector<size_t> input_shape = { 8, 32, 256 };

        config.withInputShape( input_shape )
            .withAxis( 2 )
            .withBias( false )
            .withEpsilon( 1e-6f )
            .withName( "persistent_layernorm" );

        LayerNormConfig copied_config = config;

        EXPECT_EQ( copied_config.getInputShape(), input_shape );
        EXPECT_EQ( copied_config.getAxis(), 2 );
        EXPECT_FALSE( copied_config.hasBias() );
        EXPECT_FLOAT_EQ( copied_config.getEpsilon(), 1e-6f );
        EXPECT_EQ( copied_config.getName(), "persistent_layernorm" );
    }

    TEST_F( LayerNormConfigTests, EdgeCases_LargeInputShape ) {
        LayerNormConfig config;
        std::vector<size_t> large_shape = { 1024, 2048, 4096 };

        config.withInputShape( large_shape );

        EXPECT_EQ( config.getInputShape(), large_shape );
    }

    TEST_F( LayerNormConfigTests, EdgeCases_SingleElementShape ) {
        LayerNormConfig config;
        std::vector<size_t> single_shape = { 1 };

        config.withInputShape( single_shape );

        EXPECT_EQ( config.getInputShape(), single_shape );
    }

    TEST_F( LayerNormConfigTests, EdgeCases_HighDimensionalShape ) {
        LayerNormConfig config;
        std::vector<size_t> high_dim_shape = { 2, 4, 8, 16, 32 };

        config.withInputShape( high_dim_shape );

        EXPECT_EQ( config.getInputShape(), high_dim_shape );
    }

    TEST_F( LayerNormConfigTests, EdgeCases_ExtremeAxisValues ) {
        LayerNormConfig config;

        config.withAxis( -100 );
        EXPECT_EQ( config.getAxis(), -100 );

        config.withAxis( 100 );
        EXPECT_EQ( config.getAxis(), 100 );

        config.withAxis( 0 );
        EXPECT_EQ( config.getAxis(), 0 );
    }

    TEST_F( LayerNormConfigTests, EdgeCases_VerySmallEpsilon ) {
        LayerNormConfig config;
        config.withInputShape( { 1, 1, 1 } )
            .withEpsilon( 1e-20f );

        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-20f );
        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( LayerNormConfigTests, EdgeCases_VeryLargeEpsilon ) {
        LayerNormConfig config;
        config.withInputShape( { 1, 1, 1 } )
            .withEpsilon( 1.0f );

        EXPECT_FLOAT_EQ( config.getEpsilon(), 1.0f );
        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( LayerNormConfigTests, DefaultValidation ) {
        LayerNormConfig config( 768 );

        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( LayerNormConfigTests, MethodChainPreservation ) {
        LayerNormConfig config;
        std::vector<size_t> original_shape = { 16, 32, 64 };

        config.withInputShape( original_shape )
            .withAxis( 1 )
            .withBias( false )
            .withEpsilon( 1e-4f );

        config.withName( "chained_config" );

        EXPECT_EQ( config.getInputShape(), original_shape );
        EXPECT_EQ( config.getAxis(), 1 );
        EXPECT_FALSE( config.hasBias() );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-4f );
        EXPECT_EQ( config.getName(), "chained_config" );
    }

    TEST_F( LayerNormConfigTests, MultipleBiasToggle ) {
        LayerNormConfig config;

        EXPECT_TRUE( config.hasBias() );

        config.withBias( false );
        EXPECT_FALSE( config.hasBias() );

        config.withBias( true );
        EXPECT_TRUE( config.hasBias() );

        config.withBias( false );
        EXPECT_FALSE( config.hasBias() );
    }

    TEST_F( LayerNormConfigTests, MultipleEpsilonUpdates ) {
        LayerNormConfig config;

        config.withEpsilon( 1e-3f );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-3f );

        config.withEpsilon( 1e-6f );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-6f );

        config.withEpsilon( 1e-8f );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-8f );
    }

    TEST_F( LayerNormConfigTests, TypicalTransformerConfiguration ) {
        LayerNormConfig config;

        config.withInputShape( { 32, 512, 768 } )
            .withAxis( -1 )
            .withBias( true )
            .withEpsilon( 1e-12f )
            .withName( "transformer_layernorm" )
            .withTraining( true );

        EXPECT_EQ( config.getInputShape(), std::vector<size_t>( { 32, 512, 768 } ) );
        EXPECT_EQ( config.getAxis(), -1 );
        EXPECT_TRUE( config.hasBias() );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-12f );
        EXPECT_EQ( config.getName(), "transformer_layernorm" );
        EXPECT_TRUE( config.isTraining() );

        EXPECT_NO_THROW( config.validate() );
    }
}