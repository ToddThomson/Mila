#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

import Mila;

namespace Dnn::Components::Normalization::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class LayerNormConfigTests : public ::testing::Test {
    protected:
        void SetUp() override {}
    };

    // ========================================================================
    // Constructor — shape mode
    // ========================================================================

    TEST_F( LayerNormConfigTests, ShapeMode_Constructor_SetsNormalizedShape )
    {
        LayerNormConfig config( shape_t{ 768 } );

        EXPECT_TRUE( config.hasNormalizedShape() );
        ASSERT_EQ( config.getNormalizedShape().size(), 1u );
        EXPECT_EQ( config.getNormalizedShape()[ 0 ], 768 );
        EXPECT_FALSE( config.getAxis().has_value() );
    }

    TEST_F( LayerNormConfigTests, ShapeMode_Constructor_Defaults )
    {
        LayerNormConfig config( shape_t{ 512 } );

        EXPECT_TRUE( config.hasBias() );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-5f );
    }

    TEST_F( LayerNormConfigTests, ShapeMode_MultiDimensional )
    {
        LayerNormConfig config( shape_t{ 32, 64 } );

        EXPECT_TRUE( config.hasNormalizedShape() );
        ASSERT_EQ( config.getNormalizedShape().size(), 2u );
        EXPECT_EQ( config.getNormalizedShape()[ 0 ], 32 );
        EXPECT_EQ( config.getNormalizedShape()[ 1 ], 64 );
    }

    // ========================================================================
    // Constructor — axis mode
    // ========================================================================

    TEST_F( LayerNormConfigTests, AxisMode_Constructor_SetsAxis )
    {
        LayerNormConfig config( int64_t{ -1 } );

        ASSERT_TRUE( config.getAxis().has_value() );
        EXPECT_EQ( config.getAxis().value(), -1 );
        EXPECT_FALSE( config.hasNormalizedShape() );
    }

    TEST_F( LayerNormConfigTests, AxisMode_Constructor_Defaults )
    {
        LayerNormConfig config( int64_t{ 2 } );

        EXPECT_TRUE( config.hasBias() );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-5f );
    }

    TEST_F( LayerNormConfigTests, AxisMode_Constructor_NegativeAxis )
    {
        LayerNormConfig config( int64_t{ -2 } );

        ASSERT_TRUE( config.getAxis().has_value() );
        EXPECT_EQ( config.getAxis().value(), -2 );
    }

    TEST_F( LayerNormConfigTests, AxisMode_Constructor_ZeroAxis )
    {
        LayerNormConfig config( int64_t{ 0 } );

        ASSERT_TRUE( config.getAxis().has_value() );
        EXPECT_EQ( config.getAxis().value(), 0 );
    }

    // ========================================================================
    // Fluent setters — withBias
    // ========================================================================

    TEST_F( LayerNormConfigTests, WithBias_ReturnsRef_AndSetsValue )
    {
        LayerNormConfig config( int64_t{ -1 } );

        auto& result = config.withBias( false );

        EXPECT_EQ( &result, &config );
        EXPECT_FALSE( config.hasBias() );
    }

    TEST_F( LayerNormConfigTests, WithBias_Toggle )
    {
        LayerNormConfig config( int64_t{ -1 } );

        EXPECT_TRUE( config.hasBias() );

        config.withBias( false );
        EXPECT_FALSE( config.hasBias() );

        config.withBias( true );
        EXPECT_TRUE( config.hasBias() );

        config.withBias( false );
        EXPECT_FALSE( config.hasBias() );
    }

    // ========================================================================
    // Fluent setters — withEpsilon
    // ========================================================================

    TEST_F( LayerNormConfigTests, WithEpsilon_ReturnsRef_AndSetsValue )
    {
        LayerNormConfig config( int64_t{ -1 } );

        auto& result = config.withEpsilon( 1e-6f );

        EXPECT_EQ( &result, &config );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-6f );
    }

    TEST_F( LayerNormConfigTests, WithEpsilon_MultipleUpdates )
    {
        LayerNormConfig config( int64_t{ -1 } );

        config.withEpsilon( 1e-3f );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-3f );

        config.withEpsilon( 1e-6f );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-6f );

        config.withEpsilon( 1e-8f );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-8f );
    }

    // ========================================================================
    // Fluent chaining
    // ========================================================================

    TEST_F( LayerNormConfigTests, FluentChaining_AxisMode )
    {
        LayerNormConfig config( int64_t{ 2 } );

        config.withBias( false )
            .withEpsilon( 1e-4f )
            .withPrecisionPolicy( ComputePrecision::Policy::Performance );

        ASSERT_TRUE( config.getAxis().has_value() );
        EXPECT_EQ( config.getAxis().value(), 2 );
        EXPECT_FALSE( config.hasBias() );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-4f );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Performance );
    }

    TEST_F( LayerNormConfigTests, FluentChaining_ShapeMode )
    {
        LayerNormConfig config( shape_t{ 768 } );

        config.withBias( false )
            .withEpsilon( 1e-5f )
            .withPrecisionPolicy( ComputePrecision::Policy::Accuracy );

        EXPECT_TRUE( config.hasNormalizedShape() );
        EXPECT_FALSE( config.hasBias() );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-5f );
        EXPECT_EQ( config.getPrecisionPolicy(), ComputePrecision::Policy::Accuracy );
    }

    TEST_F( LayerNormConfigTests, FluentChaining_RvalueProducesValidObject )
    {
        auto config = LayerNormConfig( shape_t{ 768 } )
            .withBias( false )
            .withEpsilon( 1e-5f );

        EXPECT_TRUE( config.hasNormalizedShape() );
        EXPECT_FALSE( config.hasBias() );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-5f );
    }

    // ========================================================================
    // Validation
    // ========================================================================

    TEST_F( LayerNormConfigTests, Validate_ShapeMode_Valid )
    {
        LayerNormConfig config( shape_t{ 768 } );

        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( LayerNormConfigTests, Validate_AxisMode_Valid )
    {
        LayerNormConfig config( int64_t{ -1 } );

        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( LayerNormConfigTests, Validate_ShapeMode_ZeroDimension_Throws )
    {
        LayerNormConfig config( shape_t{ 0 } );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( LayerNormConfigTests, Validate_ShapeMode_NegativeDimension_Throws )
    {
        LayerNormConfig config( shape_t{ -1 } );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( LayerNormConfigTests, Validate_ZeroEpsilon_Throws )
    {
        LayerNormConfig config( int64_t{ -1 } );
        config.withEpsilon( 0.0f );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( LayerNormConfigTests, Validate_NegativeEpsilon_Throws )
    {
        LayerNormConfig config( int64_t{ -1 } );
        config.withEpsilon( -1e-5f );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( LayerNormConfigTests, Validate_ErrorMessage_MentionsEpsilon )
    {
        LayerNormConfig config( int64_t{ -1 } );
        config.withEpsilon( 0.0f );

        try {
            config.validate();
            FAIL() << "Expected std::invalid_argument";
        }
        catch ( const std::invalid_argument& e ) {
            EXPECT_NE( std::string( e.what() ).find( "epsilon" ), std::string::npos );
        }
    }

    TEST_F( LayerNormConfigTests, Validate_ErrorMessage_MentionsInvalidShapeIndex )
    {
        LayerNormConfig config( shape_t{ 64, 0, 32 } );

        try {
            config.validate();
            FAIL() << "Expected std::invalid_argument";
        }
        catch ( const std::invalid_argument& e ) {
            std::string msg = e.what();
            EXPECT_NE( msg.find( "index" ), std::string::npos );
        }
    }

    // ========================================================================
    // Copy semantics
    // ========================================================================

    TEST_F( LayerNormConfigTests, Copy_PreservesAllFields_ShapeMode )
    {
        LayerNormConfig config( shape_t{ 512 } );
        config.withBias( false ).withEpsilon( 1e-6f ).withPrecisionPolicy( ComputePrecision::Policy::Accuracy );

        LayerNormConfig copy = config;

        EXPECT_TRUE( copy.hasNormalizedShape() );
        ASSERT_EQ( copy.getNormalizedShape().size(), 1u );
        EXPECT_EQ( copy.getNormalizedShape()[ 0 ], 512 );
        EXPECT_FALSE( copy.hasBias() );
        EXPECT_FLOAT_EQ( copy.getEpsilon(), 1e-6f );
        EXPECT_EQ( copy.getPrecisionPolicy(), ComputePrecision::Policy::Accuracy );
    }

    TEST_F( LayerNormConfigTests, Copy_PreservesAllFields_AxisMode )
    {
        LayerNormConfig config( int64_t{ 2 } );
        config.withBias( false ).withEpsilon( 1e-6f ).withPrecisionPolicy( ComputePrecision::Policy::Accuracy );

        LayerNormConfig copy = config;

        ASSERT_TRUE( copy.getAxis().has_value() );
        EXPECT_EQ( copy.getAxis().value(), 2 );
        EXPECT_FALSE( copy.hasBias() );
        EXPECT_FLOAT_EQ( copy.getEpsilon(), 1e-6f );
        EXPECT_EQ( copy.getPrecisionPolicy(), ComputePrecision::Policy::Accuracy );
    }

    // ========================================================================
    // Edge cases
    // ========================================================================

    TEST_F( LayerNormConfigTests, EdgeCase_VerySmallEpsilon )
    {
        LayerNormConfig config( int64_t{ -1 } );
        config.withEpsilon( 1e-20f );

        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-20f );
        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( LayerNormConfigTests, EdgeCase_LargeEpsilon )
    {
        LayerNormConfig config( int64_t{ -1 } );
        config.withEpsilon( 1.0f );

        EXPECT_FLOAT_EQ( config.getEpsilon(), 1.0f );
        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( LayerNormConfigTests, EdgeCase_ExtremeNegativeAxis )
    {
        LayerNormConfig config( int64_t{ -100 } );

        ASSERT_TRUE( config.getAxis().has_value() );
        EXPECT_EQ( config.getAxis().value(), -100 );
    }

    TEST_F( LayerNormConfigTests, EdgeCase_LargePositiveAxis )
    {
        LayerNormConfig config( int64_t{ 100 } );

        ASSERT_TRUE( config.getAxis().has_value() );
        EXPECT_EQ( config.getAxis().value(), 100 );
    }

    // ========================================================================
    // Typical usage patterns
    // ========================================================================

    TEST_F( LayerNormConfigTests, TypicalTransformerConfig_ShapeMode )
    {
        auto config = LayerNormConfig( shape_t{ 768 } )
            .withEpsilon( 1e-5f )
            .withBias( true );

        EXPECT_TRUE( config.hasNormalizedShape() );
        EXPECT_EQ( config.getNormalizedShape()[ 0 ], 768 );
        EXPECT_TRUE( config.hasBias() );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-5f );
        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( LayerNormConfigTests, TypicalTransformerConfig_AxisMode )
    {
        auto config = LayerNormConfig( int64_t{ -1 } )
            .withEpsilon( 1e-12f )
            .withBias( false );

        ASSERT_TRUE( config.getAxis().has_value() );
        EXPECT_EQ( config.getAxis().value(), -1 );
        EXPECT_FALSE( config.hasBias() );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-12f );
        EXPECT_NO_THROW( config.validate() );
    }
}