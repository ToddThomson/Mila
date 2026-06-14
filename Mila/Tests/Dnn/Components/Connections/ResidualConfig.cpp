/**
 * @file ResidualConfig.cpp
 * @brief Config-archetype tests for ResidualConfig.
 *
 * Reference instance of the config archetype (see Specifications/Testing.md):
 * defaults, fluent setters (withConnectionType / withScalingFactor), validate(),
 * and the metadata round-trip. Device-agnostic, rides the CPU-only CI gate.
 */

#include <gtest/gtest.h>
#include <string>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Components::Connections
{
    using namespace Mila::Dnn;
    using Mila::Dnn::Serialization::SerializationMetadata;

    class ResidualConfigTests : public ::testing::Test
    {
    };

    // ====================================================================
    // A. Construction & Defaults
    // ====================================================================

    TEST_F( ResidualConfigTests, Defaults_AdditionScaleOne )
    {
        ResidualConfig config;

        EXPECT_EQ( config.getConnectionType(), ConnectionType::Addition );
        EXPECT_FLOAT_EQ( config.getScalingFactor(), 1.0f );
    }

    // ====================================================================
    // Fluent interface
    // ====================================================================

    TEST_F( ResidualConfigTests, WithConnectionType_SetsValue )
    {
        auto config = ResidualConfig().withConnectionType( ConnectionType::Addition );

        EXPECT_EQ( config.getConnectionType(), ConnectionType::Addition );
    }

    TEST_F( ResidualConfigTests, WithScalingFactor_SetsValue )
    {
        auto config = ResidualConfig().withScalingFactor( 0.5f );

        EXPECT_FLOAT_EQ( config.getScalingFactor(), 0.5f );
    }

    // ====================================================================
    // A. Validation
    // ====================================================================

    TEST_F( ResidualConfigTests, Validate_PassesForDefault )
    {
        EXPECT_NO_THROW( ResidualConfig().validate() );
    }

    TEST_F( ResidualConfigTests, Validate_ThrowsForNonPositiveScale )
    {
        EXPECT_THROW( ResidualConfig().withScalingFactor( 0.0f ).validate(), std::invalid_argument );
    }

    // ====================================================================
    // H. Serialization round-trip
    // ====================================================================

    TEST_F( ResidualConfigTests, Metadata_RoundTripPreservesFields )
    {
        ResidualConfig source;
        source.withScalingFactor( 0.25f );

        SerializationMetadata meta = source.toMetadata();

        ResidualConfig loaded;
        loaded.fromMetadata( meta );

        EXPECT_FLOAT_EQ( loaded.getScalingFactor(), 0.25f );
        EXPECT_EQ( loaded.getConnectionType(), ConnectionType::Addition );
    }

    // ====================================================================
    // I. Diagnostics
    // ====================================================================

    TEST_F( ResidualConfigTests, ToString_NamesConfig )
    {
        ResidualConfig config;

        EXPECT_NE( config.toString().find( "ResidualConfig" ), std::string::npos );
    }
}
