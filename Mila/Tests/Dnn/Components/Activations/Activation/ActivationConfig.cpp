/**
 * @file ActivationConfig.cpp
 * @brief Config-archetype tests for ActivationConfig.
 *
 * Defaults, fluent setters, the elementwise-only validation gate, and the
 * metadata round-trip (Specifications/Testing.md config archetype). No
 * forward/backward -- that is Activation.Cpu.cpp.
 */

#include <gtest/gtest.h>
#include <string>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Components::Activations::Activation
{
    using namespace Mila::Dnn;
    using Mila::Dnn::Serialization::SerializationMetadata;

    class ActivationConfigTests : public ::testing::Test
    {
    };

    // ====================================================================
    // A. Construction & Defaults
    // ====================================================================

    TEST_F( ActivationConfigTests, Default_IsGelu )
    {
        ActivationConfig config;

        EXPECT_EQ( config.getActivationType(), ActivationType::Gelu );
        EXPECT_FLOAT_EQ( config.getLeakyReluAlpha(), 0.01f );
        EXPECT_EQ( config.getGeluApproximation(), ApproximationMethod::Tanh );
    }

    TEST_F( ActivationConfigTests, ExplicitType_IsStored )
    {
        ActivationConfig config( ActivationType::Mish );

        EXPECT_EQ( config.getActivationType(), ActivationType::Mish );
    }

    // ====================================================================
    // Fluent interface
    // ====================================================================

    TEST_F( ActivationConfigTests, WithLeakyReluAlpha_SetsValue )
    {
        auto config = ActivationConfig( ActivationType::LeakyRelu ).withLeakyReluAlpha( 0.2f );

        EXPECT_FLOAT_EQ( config.getLeakyReluAlpha(), 0.2f );
    }

    TEST_F( ActivationConfigTests, WithActivationType_ChainsOnRvalue )
    {
        auto config = ActivationConfig().withActivationType( ActivationType::Silu );

        EXPECT_EQ( config.getActivationType(), ActivationType::Silu );
    }

    // ====================================================================
    // A. Validation -- elementwise only
    // ====================================================================

    TEST_F( ActivationConfigTests, Validate_PassesForElementwise )
    {
        EXPECT_NO_THROW( ActivationConfig( ActivationType::Gelu ).validate() );
        EXPECT_NO_THROW( ActivationConfig( ActivationType::Silu ).validate() );
        EXPECT_NO_THROW( ActivationConfig( ActivationType::Relu ).validate() );
        EXPECT_NO_THROW( ActivationConfig( ActivationType::None ).validate() );
        EXPECT_NO_THROW( ActivationConfig( ActivationType::Mish ).validate() );
    }

    TEST_F( ActivationConfigTests, Validate_ThrowsForSwiglu )
    {
        ActivationConfig config( ActivationType::Swiglu );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( ActivationConfigTests, IsElementwise_RejectsSwiglu )
    {
        EXPECT_TRUE( isElementwiseActivation( ActivationType::Gelu ) );
        EXPECT_TRUE( isElementwiseActivation( ActivationType::LeakyRelu ) );
        EXPECT_FALSE( isElementwiseActivation( ActivationType::Swiglu ) );
    }

    // ====================================================================
    // H. Serialization round-trip
    // ====================================================================

    TEST_F( ActivationConfigTests, Metadata_RoundTripPreservesFields )
    {
        auto source = ActivationConfig( ActivationType::LeakyRelu ).withLeakyReluAlpha( 0.05f );

        SerializationMetadata meta = source.toMetadata();

        ActivationConfig loaded;
        loaded.fromMetadata( meta );

        EXPECT_EQ( loaded.getActivationType(), ActivationType::LeakyRelu );
        EXPECT_FLOAT_EQ( loaded.getLeakyReluAlpha(), 0.05f );
    }

    TEST_F( ActivationConfigTests, FromMetadata_IgnoresMissingKeys )
    {
        ActivationConfig config( ActivationType::Silu );

        SerializationMetadata empty;
        config.fromMetadata( empty );

        EXPECT_EQ( config.getActivationType(), ActivationType::Silu );
    }

    // ====================================================================
    // I. Diagnostics
    // ====================================================================

    TEST_F( ActivationConfigTests, ToString_NonEmpty )
    {
        ActivationConfig config( ActivationType::Tanh );

        EXPECT_FALSE( config.toString().empty() );
    }
}
