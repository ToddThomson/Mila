/**
 * @file SwigluConfig.cpp
 * @brief Config-archetype tests for SwigluConfig.
 *
 * Reference instance of the config archetype (see Specifications/Testing.md).
 * SwigluConfig is parameterless -- hidden_dim is derived from the input shape at
 * build time -- so the surface is just default construction, the (no-op)
 * validate(), and the metadata round-trip. There are no fluent setters to cover.
 *
 * Device-agnostic, so this rides the MILA_ENABLE_CUDA=OFF CI gate even though the
 * Swiglu component itself is CUDA-only (no CPU SwigluOp specialization).
 */

#include <gtest/gtest.h>
#include <string>

import Mila;

namespace Mila::Tests::Dnn::Components::Activations::Swiglu
{
    using namespace Mila::Dnn;
    using Mila::Dnn::Serialization::SerializationMetadata;

    class SwigluConfigTests : public ::testing::Test
    {
    };

    // ====================================================================
    // A. Construction & Validation
    // ====================================================================

    TEST_F( SwigluConfigTests, Default_ConstructsAndValidates )
    {
        SwigluConfig config;

        EXPECT_NO_THROW( config.validate() );
    }

    // ====================================================================
    // H. Serialization round-trip
    // ====================================================================

    TEST_F( SwigluConfigTests, Metadata_RoundTripStaysValid )
    {
        SwigluConfig source;

        SerializationMetadata meta = source.toMetadata();

        SwigluConfig loaded;
        loaded.fromMetadata( meta );

        EXPECT_NO_THROW( loaded.validate() );
    }

    // ====================================================================
    // I. Diagnostics
    // ====================================================================

    TEST_F( SwigluConfigTests, ToString_NamesConfig )
    {
        SwigluConfig config;

        EXPECT_NE( config.toString().find( "SwigluConfig" ), std::string::npos );
    }
}
