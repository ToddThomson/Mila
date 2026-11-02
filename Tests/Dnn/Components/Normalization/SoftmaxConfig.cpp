/**
 * @file SoftmaxConfig.cpp
 * @brief Config-archetype tests for SoftmaxConfig.
 *
 * Reference instance of the config archetype (see Specifications/Testing.md):
 * the axis fluent setter, the default (-1 = last dimension), validate() (a no-op
 * for Softmax), and the metadata round-trip. Device-agnostic, rides the CPU gate.
 *
 * Re-greened from the pre-methodology version (dropped the retired
 * withPrecisionPolicy axis, removed from component configs).
 */

#include <gtest/gtest.h>
#include <string>

import Mila;

namespace Mila::Tests::Dnn::Components::Normalization
{
    using namespace Mila::Dnn;
    using Mila::Dnn::Serialization::SerializationMetadata;

    class SoftmaxConfigTests : public ::testing::Test
    {
    };

    // ====================================================================
    // A. Construction & Defaults
    // ====================================================================

    TEST_F( SoftmaxConfigTests, Defaults_AxisMinusOne )
    {
        SoftmaxConfig config;

        EXPECT_EQ( config.getAxis(), -1 );
    }

    // ====================================================================
    // Fluent interface
    // ====================================================================

    TEST_F( SoftmaxConfigTests, WithAxis_SetsValue )
    {
        auto config = SoftmaxConfig().withAxis( 2 );

        EXPECT_EQ( config.getAxis(), 2 );
    }

    // ====================================================================
    // A. Validation (no-op contract — accepts any axis)
    // ====================================================================

    TEST_F( SoftmaxConfigTests, Validate_PassesAlways )
    {
        EXPECT_NO_THROW( SoftmaxConfig().withAxis( 7 ).validate() );
    }

    // ====================================================================
    // H. Serialization round-trip
    // ====================================================================

    TEST_F( SoftmaxConfigTests, Metadata_RoundTripPreservesAxis )
    {
        SoftmaxConfig source;
        source.withAxis( 1 );

        SerializationMetadata meta = source.toMetadata();

        SoftmaxConfig loaded;
        loaded.fromMetadata( meta );

        EXPECT_EQ( loaded.getAxis(), 1 );
    }

    // ====================================================================
    // I. Diagnostics
    // ====================================================================

    TEST_F( SoftmaxConfigTests, ToString_NamesConfig )
    {
        SoftmaxConfig config;

        EXPECT_NE( config.toString().find( "SoftmaxConfig" ), std::string::npos );
    }
}
