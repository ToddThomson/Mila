/**
 * @file MLPConfig.cpp
 * @brief Config-archetype tests for MLPConfig.
 *
 * Reference for the config test archetype (see Specifications/Testing.md):
 * construction defaults, fluent setters, validation throws, and the metadata
 * round-trip. No forward/backward -- that is the concrete-component archetype
 * (MLP.Cpu.cpp / MLP.Cuda.cpp).
 */

#include <gtest/gtest.h>
#include <string>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Components::FFN
{
    using namespace Mila::Dnn;
    using Mila::Dnn::Serialization::SerializationMetadata;

    class MLPConfigTests : public ::testing::Test
    {
    };

    // ====================================================================
    // A. Construction & Defaults
    // ====================================================================

    TEST_F( MLPConfigTests, Construct_StoresFeatureDimensions )
    {
        MLPConfig config( 768, 3072 );

        EXPECT_EQ( config.getInputFeatures(), 768 );
        EXPECT_EQ( config.getHiddenSize(), 3072 );
    }

    TEST_F( MLPConfigTests, Default_BiasEnabled )
    {
        MLPConfig config( 256, 1024 );

        EXPECT_TRUE( config.hasBias() );
    }

    TEST_F( MLPConfigTests, Default_ActivationIsGelu )
    {
        MLPConfig config( 256, 1024 );

        EXPECT_EQ( config.getActivationType(), ActivationType::Gelu );
    }

    // ====================================================================
    // Fluent interface
    // ====================================================================

    TEST_F( MLPConfigTests, WithBias_SetsValue )
    {
        MLPConfig config( 256, 1024 );
        config.withBias( false );

        EXPECT_FALSE( config.hasBias() );
    }

    TEST_F( MLPConfigTests, WithBias_ReturnsSameObjectOnLvalue )
    {
        MLPConfig config( 256, 1024 );
        auto&& result = config.withBias( false );

        EXPECT_EQ( &result, &config );
    }

    TEST_F( MLPConfigTests, WithActivation_SetsValue )
    {
        MLPConfig config( 256, 1024 );
        config.withActivation( ActivationType::Swiglu );

        EXPECT_EQ( config.getActivationType(), ActivationType::Swiglu );
    }

    TEST_F( MLPConfigTests, FluentChain_AppliesAllSetters )
    {
        auto config = MLPConfig( 512, 2048 )
            .withBias( false )
            .withActivation( ActivationType::Gelu );

        EXPECT_EQ( config.getInputFeatures(), 512 );
        EXPECT_EQ( config.getHiddenSize(), 2048 );
        EXPECT_FALSE( config.hasBias() );
        EXPECT_EQ( config.getActivationType(), ActivationType::Gelu );
    }

    TEST_F( MLPConfigTests, WithBias_MultipleCallsTakeLast )
    {
        MLPConfig config( 512, 2048 );
        config.withBias( false );
        config.withBias( true );

        EXPECT_TRUE( config.hasBias() );
    }

    // ====================================================================
    // A. Validation
    // ====================================================================

    TEST_F( MLPConfigTests, Validate_PassesForPositiveDimensions )
    {
        MLPConfig config( 512, 2048 );

        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( MLPConfigTests, Validate_PassesForSingleFeature )
    {
        MLPConfig config( 1, 1 );

        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( MLPConfigTests, Validate_ThrowsForZeroInputFeatures )
    {
        MLPConfig config( 0, 1024 );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( MLPConfigTests, Validate_ThrowsForNegativeInputFeatures )
    {
        MLPConfig config( -1, 1024 );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( MLPConfigTests, Validate_ThrowsForZeroHiddenSize )
    {
        MLPConfig config( 512, 0 );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( MLPConfigTests, Validate_ThrowsForNegativeHiddenSize )
    {
        MLPConfig config( 512, -1 );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( MLPConfigTests, Validate_ErrorMentionsInputFeatures )
    {
        MLPConfig config( 0, 1024 );

        try
        {
            config.validate();
            FAIL() << "expected std::invalid_argument";
        }
        catch ( const std::invalid_argument& e )
        {
            EXPECT_NE( std::string( e.what() ).find( "Input features" ), std::string::npos );
        }
    }

    TEST_F( MLPConfigTests, Validate_ErrorMentionsHiddenSize )
    {
        MLPConfig config( 512, 0 );

        try
        {
            config.validate();
            FAIL() << "expected std::invalid_argument";
        }
        catch ( const std::invalid_argument& e )
        {
            EXPECT_NE( std::string( e.what() ).find( "Hidden size" ), std::string::npos );
        }
    }

    TEST_F( MLPConfigTests, Validate_RejectsNonGelu )
    {
        // MLP is the dense GELU FFN; gated activations live in GatedMLP. The config
        // setter and metadata accept any enum value, but validate() is the honest
        // gate that rejects an activation MLP does not implement.
        MLPConfig config( 512, 2048 );
        config.withActivation( ActivationType::Swiglu );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( MLPConfigTests, Validate_PassesForGelu )
    {
        MLPConfig config( 512, 2048 );
        config.withActivation( ActivationType::Gelu );

        EXPECT_NO_THROW( config.validate() );
    }

    // ====================================================================
    // H. Serialization round-trip
    // ====================================================================

    TEST_F( MLPConfigTests, Metadata_RoundTripPreservesFields )
    {
        MLPConfig source( 768, 3072 );
        source.withBias( false ).withActivation( ActivationType::Swiglu );

        SerializationMetadata meta = source.toMetadata();

        MLPConfig loaded( 1, 1 );
        loaded.fromMetadata( meta );

        EXPECT_EQ( loaded.getInputFeatures(), 768 );
        EXPECT_EQ( loaded.getHiddenSize(), 3072 );
        EXPECT_FALSE( loaded.hasBias() );
        EXPECT_EQ( loaded.getActivationType(), ActivationType::Swiglu );
    }

    TEST_F( MLPConfigTests, FromMetadata_IgnoresMissingKeys )
    {
        MLPConfig config( 256, 1024 );
        config.withBias( false );

        SerializationMetadata empty;
        config.fromMetadata( empty );

        // Untouched fields retain their prior values.
        EXPECT_EQ( config.getInputFeatures(), 256 );
        EXPECT_EQ( config.getHiddenSize(), 1024 );
        EXPECT_FALSE( config.hasBias() );
    }

    // ====================================================================
    // I. Diagnostics
    // ====================================================================

    TEST_F( MLPConfigTests, ToString_NonEmpty )
    {
        MLPConfig config( 768, 3072 );

        EXPECT_FALSE( config.toString().empty() );
    }

    // ====================================================================
    // J. Accessor traits
    // ====================================================================

    TEST_F( MLPConfigTests, Getters_AreNoexcept )
    {
        MLPConfig config( 512, 2048 );

        EXPECT_TRUE( noexcept( config.getInputFeatures() ) );
        EXPECT_TRUE( noexcept( config.getHiddenSize() ) );
        EXPECT_TRUE( noexcept( config.hasBias() ) );
        EXPECT_TRUE( noexcept( config.getActivationType() ) );
    }
}
