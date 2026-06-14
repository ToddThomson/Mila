/**
 * @file RmsNormConfig.cpp
 * @brief Config-archetype tests for RmsNormConfig.
 *
 * Reference instance of the config archetype (see Specifications/Testing.md):
 * the two construction modes (shape / axis), fluent setters (withBias /
 * withEpsilon), validate(), and the metadata round-trip.
 *
 * Device-agnostic, so this rides the MILA_ENABLE_CUDA=OFF CI gate even though the
 * RmsNorm component is CUDA-only (no CPU RmsNormOp specialization).
 */

#include <gtest/gtest.h>
#include <string>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Components::Normalization::RmsNorm
{
    using namespace Mila::Dnn;
    using Mila::Dnn::Serialization::SerializationMetadata;

    class RmsNormConfigTests : public ::testing::Test
    {
    };

    // ====================================================================
    // A. Construction & Defaults
    // ====================================================================

    TEST_F( RmsNormConfigTests, ShapeMode_StoresNormalizedShape )
    {
        RmsNormConfig config( shape_t{ 768 } );

        EXPECT_TRUE( config.hasNormalizedShape() );
        EXPECT_EQ( config.getNormalizedShape(), ( shape_t{ 768 } ) );
        EXPECT_FALSE( config.getAxis().has_value() );
    }

    TEST_F( RmsNormConfigTests, AxisMode_StoresAxis )
    {
        RmsNormConfig config( int64_t{ -1 } );

        EXPECT_FALSE( config.hasNormalizedShape() );
        ASSERT_TRUE( config.getAxis().has_value() );
        EXPECT_EQ( config.getAxis().value(), -1 );
    }

    TEST_F( RmsNormConfigTests, Defaults_BiasTrueEpsilon1e5 )
    {
        RmsNormConfig config( shape_t{ 16 } );

        EXPECT_TRUE( config.hasBias() );
        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-5f );
    }

    // ====================================================================
    // Fluent interface
    // ====================================================================

    TEST_F( RmsNormConfigTests, WithBias_SetsValue )
    {
        auto config = RmsNormConfig( shape_t{ 16 } ).withBias( false );

        EXPECT_FALSE( config.hasBias() );
    }

    TEST_F( RmsNormConfigTests, WithEpsilon_SetsValue )
    {
        auto config = RmsNormConfig( shape_t{ 16 } ).withEpsilon( 1e-6f );

        EXPECT_FLOAT_EQ( config.getEpsilon(), 1e-6f );
    }

    // ====================================================================
    // A. Validation
    // ====================================================================

    TEST_F( RmsNormConfigTests, Validate_PassesForShapeMode )
    {
        EXPECT_NO_THROW( RmsNormConfig( shape_t{ 16 } ).validate() );
    }

    TEST_F( RmsNormConfigTests, Validate_PassesForAxisMode )
    {
        EXPECT_NO_THROW( RmsNormConfig( int64_t{ -1 } ).validate() );
    }

    TEST_F( RmsNormConfigTests, Validate_ThrowsForNonPositiveEpsilon )
    {
        EXPECT_THROW( RmsNormConfig( shape_t{ 16 } ).withEpsilon( 0.0f ).validate(), std::invalid_argument );
    }

    TEST_F( RmsNormConfigTests, Validate_ThrowsForZeroNormalizedDim )
    {
        EXPECT_THROW( RmsNormConfig( shape_t{ 0 } ).validate(), std::invalid_argument );
    }

    // ====================================================================
    // H. Serialization round-trip
    // ====================================================================

    TEST_F( RmsNormConfigTests, Metadata_RoundTripShapeMode )
    {
        RmsNormConfig source( shape_t{ 768 } );
        source.withBias( false ).withEpsilon( 1e-6f );

        SerializationMetadata meta = source.toMetadata();

        RmsNormConfig loaded( shape_t{ 1 } );
        loaded.fromMetadata( meta );

        EXPECT_EQ( loaded.getNormalizedShape(), ( shape_t{ 768 } ) );
        EXPECT_FALSE( loaded.hasBias() );
        EXPECT_FLOAT_EQ( loaded.getEpsilon(), 1e-6f );
    }

    // ====================================================================
    // I. Diagnostics
    // ====================================================================

    TEST_F( RmsNormConfigTests, ToString_NamesConfig )
    {
        RmsNormConfig config( shape_t{ 16 } );

        EXPECT_NE( config.toString().find( "RmsNormConfig" ), std::string::npos );
    }
}
