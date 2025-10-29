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

        // Axis is optional by default (not set)
        EXPECT_FALSE( config.getAxis().has_value() );
        EXPECT_TRUE( config.hasBias() );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-5f );
    }

    TEST_F( LayerNormConfigTests, WithAxis ) {
        LayerNormConfig config;

        auto& result = config.withAxis( 2 );

        EXPECT_EQ( &result, &config );
        ASSERT_TRUE( config.getAxis().has_value() );
        EXPECT_EQ( config.getAxis().value(), 2 );

        config.withAxis( -2 );
        ASSERT_TRUE( config.getAxis().has_value() );
        EXPECT_EQ( config.getAxis().value(), -2 );

        config.withAxis( 0 );
        ASSERT_TRUE( config.getAxis().has_value() );
        EXPECT_EQ( config.getAxis().value(), 0 );
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

    TEST_F( LayerNormConfigTests, FluentInterfaceChaining_NoInputShape ) {
        LayerNormConfig config;

        auto& result = config
            .withAxis( 2 )
            .withBias( false )
            .withEpsilon( 1e-4f )
            .withName( "test_layernorm" )
            .withTraining( true );

        EXPECT_EQ( &result, &config );
        ASSERT_TRUE( config.getAxis().has_value() );
        EXPECT_EQ( config.getAxis().value(), 2 );
        EXPECT_FALSE( config.hasBias() );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-4f );
        EXPECT_EQ( config.getName(), "test_layernorm" );
        EXPECT_TRUE( config.isTraining() );
    }

    TEST_F( LayerNormConfigTests, Validate_Throws_WhenInputShapeMissing ) {
        LayerNormConfig config;
        config.withEpsilon( 1e-5f )
            .withName( "valid_config" );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( LayerNormConfigTests, ValidationErrorMessage_Default ) {
        LayerNormConfig config;
        config.withName( "test_config" );

        try {
            config.validate();
            FAIL() << "Expected std::invalid_argument to be thrown";
        }
        catch ( const std::invalid_argument& e ) {
            std::string error_msg = e.what();
            // Updated validation message references normalized_shape or axis requirement
            EXPECT_TRUE( error_msg.find( "normalized_shape" ) != std::string::npos ||
                         error_msg.find( "axis" ) != std::string::npos );
        }
    }

    TEST_F( LayerNormConfigTests, BaseClassInteraction_NoInputShape ) {
        LayerNormConfig config;

        config.withAxis( -1 )
            .withBias( true )
            .withEpsilon( 1e-5f )
            .withName( "test_layernorm" )
            .withPrecisionPolicy( ComputePrecision::Policy::Performance )
            .withTraining( false );

        ASSERT_TRUE( config.getAxis().has_value() );
        EXPECT_EQ( config.getAxis().value(), -1 );
        EXPECT_TRUE( config.hasBias() );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-5f );
        EXPECT_EQ( config.getName(), "test_layernorm" );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Performance );
        EXPECT_FALSE( config.isTraining() );
    }

    TEST_F( LayerNormConfigTests, ConfigurationPersistence_NoInputShape ) {
        LayerNormConfig config;

        config.withAxis( 2 )
            .withBias( false )
            .withEpsilon( 1e-6f )
            .withName( "persistent_layernorm" );

        LayerNormConfig copied_config = config;

        ASSERT_TRUE( copied_config.getAxis().has_value() );
        EXPECT_EQ( copied_config.getAxis().value(), 2 );
        EXPECT_FALSE( copied_config.hasBias() );
        EXPECT_FLOAT_EQ( copied_config.getEpsilon(), 1e-6f );
        EXPECT_EQ( copied_config.getName(), "persistent_layernorm" );
    }

    TEST_F( LayerNormConfigTests, EdgeCases_ExtremeAxisValues ) {
        LayerNormConfig config;

        config.withAxis( -100 );
        ASSERT_TRUE( config.getAxis().has_value() );
        EXPECT_EQ( config.getAxis().value(), -100 );

        config.withAxis( 100 );
        ASSERT_TRUE( config.getAxis().has_value() );
        EXPECT_EQ( config.getAxis().value(), 100 );

        config.withAxis( 0 );
        ASSERT_TRUE( config.getAxis().has_value() );
        EXPECT_EQ( config.getAxis().value(), 0 );
    }

    TEST_F( LayerNormConfigTests, EdgeCases_VerySmallAndLargeEpsilon ) {
        LayerNormConfig config;

        config.withEpsilon( 1e-20f );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-20f );

        config.withEpsilon( 1.0f );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1.0f );
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

    TEST_F( LayerNormConfigTests, TypicalTransformerConfiguration_NoInputShape ) {
        LayerNormConfig config;

        config.withAxis( -1 )
            .withBias( true )
            .withEpsilon( 1e-12f )
            .withName( "transformer_layernorm" )
            .withTraining( true );

        ASSERT_TRUE( config.getAxis().has_value() );
        EXPECT_EQ( config.getAxis().value(), -1 );
        EXPECT_TRUE( config.hasBias() );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-12f );
        EXPECT_EQ( config.getName(), "transformer_layernorm" );
        EXPECT_TRUE( config.isTraining() );

        // With axis explicitly set, validate should succeed (axis is considered specified)
        EXPECT_NO_THROW( config.validate() );
    }
}