/**
 * @file LayerNormConfig.cpp
 * @brief Config-archetype tests for LayerNormConfig.
 *
 * Reference instance of the config archetype (see Specifications/Testing.md):
 * the two construction modes (shape / axis), fluent setters (withBias /
 * withEpsilon), validate(), and the metadata round-trip. Device-agnostic, so it
 * rides the MILA_ENABLE_CUDA=OFF CI gate.
 */

#include <gtest/gtest.h>
#include <string>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Components::Normalization::LayerNorm
{
    using namespace Mila::Dnn;
    using Mila::Dnn::Serialization::SerializationMetadata;

    class LayerNormConfigTests : public ::testing::Test
    {
    };

    // ====================================================================
    // A. Construction & Defaults
    // ====================================================================

    TEST_F( LayerNormConfigTests, ShapeMode_StoresNormalizedShape )
    {
        LayerNormConfig config( shape_t{ 768 } );

        EXPECT_TRUE( config.hasNormalizedShape() );
        EXPECT_EQ( config.getNormalizedShape(), ( shape_t{ 768 } ) );
        EXPECT_FALSE( config.getAxis().has_value() );
    }

    TEST_F( LayerNormConfigTests, AxisMode_StoresAxis )
    {
        LayerNormConfig config( int64_t{ -1 } );

        EXPECT_FALSE( config.hasNormalizedShape() );
        ASSERT_TRUE( config.getAxis().has_value() );
        EXPECT_EQ( config.getAxis().value(), -1 );
    }

    TEST_F( LayerNormConfigTests, Defaults_BiasTrueEpsilon1e5 )
    {
        LayerNormConfig config( shape_t{ 16 } );

        EXPECT_TRUE( config.hasBias() );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-5f );
    }

    // ====================================================================
    // Fluent interface
    // ====================================================================

    TEST_F( LayerNormConfigTests, WithBias_SetsValue )
    {
        auto config = LayerNormConfig( shape_t{ 16 } ).withBias( false );

        EXPECT_FALSE( config.hasBias() );
    }

    TEST_F( LayerNormConfigTests, WithEpsilon_SetsValue )
    {
        auto config = LayerNormConfig( shape_t{ 16 } ).withEpsilon( 1e-6f );

        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-6f );
    }

    // ====================================================================
    // A. Validation
    // ====================================================================

    TEST_F( LayerNormConfigTests, Validate_PassesForShapeMode )
    {
        EXPECT_NO_THROW( LayerNormConfig( shape_t{ 16 } ).validate() );
    }

    TEST_F( LayerNormConfigTests, Validate_PassesForAxisMode )
    {
        EXPECT_NO_THROW( LayerNormConfig( int64_t{ -1 } ).validate() );
    }

    TEST_F( LayerNormConfigTests, Validate_ThrowsForNonPositiveEpsilon )
    {
        EXPECT_THROW( LayerNormConfig( shape_t{ 16 } ).withEpsilon( 0.0f ).validate(), std::invalid_argument );
    }

    TEST_F( LayerNormConfigTests, Validate_ThrowsForZeroNormalizedDim )
    {
        EXPECT_THROW( LayerNormConfig( shape_t{ 0 } ).validate(), std::invalid_argument );
    }

    // ====================================================================
    // H. Serialization round-trip
    // ====================================================================

    TEST_F( LayerNormConfigTests, Metadata_RoundTripShapeMode )
    {
        LayerNormConfig source( shape_t{ 768 } );
        source.withBias( false ).withEpsilon( 1e-6f );

        SerializationMetadata meta = source.toMetadata();

        LayerNormConfig loaded( shape_t{ 1 } );
        loaded.fromMetadata( meta );

        EXPECT_EQ( loaded.getNormalizedShape(), ( shape_t{ 768 } ) );
        EXPECT_FALSE( loaded.hasBias() );
        EXPECT_FLOAT_EQ( loaded.getEpsilon(), 1e-6f );
    }

    // ====================================================================
    // I. Diagnostics
    // ====================================================================

    TEST_F( LayerNormConfigTests, ToString_NamesConfig )
    {
        LayerNormConfig config( shape_t{ 16 } );

        EXPECT_NE( config.toString().find( "LayerNormConfig" ), std::string::npos );
    }
}
