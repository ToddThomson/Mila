/**
 * @file GatedMLPConfig.cpp
 * @brief Config-archetype tests for GatedMLPConfig.
 *
 * Defaults, fluent setters, validation, and the metadata round-trip
 * (Specifications/Testing.md config archetype). GatedMLP's forward/backward needs
 * the CUDA-only Swiglu gate, so the functional surface lives in GatedMLP.Cuda.cpp;
 * this CPU-gated file covers the pure-config contract.
 */

#include <gtest/gtest.h>
#include <string>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Components::FFN::GatedMLP
{
    using namespace Mila::Dnn;
    using Mila::Dnn::Serialization::SerializationMetadata;

    class GatedMLPConfigTests : public ::testing::Test
    {
    };

    // ====================================================================
    // A. Construction & Defaults
    // ====================================================================

    TEST_F( GatedMLPConfigTests, Construct_StoresDimensions )
    {
        GatedMLPConfig config( 512, 1376 );

        EXPECT_EQ( config.getInputFeatures(), 512 );
        EXPECT_EQ( config.getHiddenSize(), 1376 );
    }

    TEST_F( GatedMLPConfigTests, Default_IsBiasFreeSilu )
    {
        GatedMLPConfig config( 8, 16 );

        // Gated FFNs are typically bias-free.
        EXPECT_FALSE( config.hasBias() );
        EXPECT_EQ( config.getGateActivation(), ActivationType::Silu );
    }

    // ====================================================================
    // Fluent interface
    // ====================================================================

    TEST_F( GatedMLPConfigTests, WithBias_SetsValue )
    {
        auto config = GatedMLPConfig( 8, 16 ).withBias( true );

        EXPECT_TRUE( config.hasBias() );
    }

    TEST_F( GatedMLPConfigTests, WithGateActivation_SetsValue )
    {
        auto config = GatedMLPConfig( 8, 16 ).withGateActivation( ActivationType::Gelu );

        EXPECT_EQ( config.getGateActivation(), ActivationType::Gelu );
    }

    // ====================================================================
    // A. Validation
    // ====================================================================

    TEST_F( GatedMLPConfigTests, Validate_PassesForPositiveDims )
    {
        GatedMLPConfig config( 8, 16 );

        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( GatedMLPConfigTests, Validate_ThrowsForZeroInputFeatures )
    {
        GatedMLPConfig config( 0, 16 );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( GatedMLPConfigTests, Validate_ThrowsForZeroHiddenSize )
    {
        GatedMLPConfig config( 8, 0 );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    // ====================================================================
    // H. Serialization round-trip
    // ====================================================================

    TEST_F( GatedMLPConfigTests, Metadata_RoundTripPreservesFields )
    {
        auto source = GatedMLPConfig( 512, 1376 )
            .withBias( true )
            .withGateActivation( ActivationType::Gelu );

        SerializationMetadata meta = source.toMetadata();

        GatedMLPConfig loaded( 1, 1 );
        loaded.fromMetadata( meta );

        EXPECT_EQ( loaded.getInputFeatures(), 512 );
        EXPECT_EQ( loaded.getHiddenSize(), 1376 );
        EXPECT_TRUE( loaded.hasBias() );
        EXPECT_EQ( loaded.getGateActivation(), ActivationType::Gelu );
    }

    // ====================================================================
    // I. Diagnostics
    // ====================================================================

    TEST_F( GatedMLPConfigTests, ToString_NonEmpty )
    {
        GatedMLPConfig config( 8, 16 );

        EXPECT_FALSE( config.toString().empty() );
    }
}
