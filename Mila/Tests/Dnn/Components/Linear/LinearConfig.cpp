/**
 * @file LinearConfig.cpp
 * @brief Config-archetype tests for LinearConfig.
 *
 * Reference instance of the config test archetype (see Specifications/Testing.md):
 * defaults, fluent setters, validation, and the metadata round-trip. No
 * forward/backward -- that is the concrete-component archetype (Linear.Cpu.cpp).
 *
 * Device-agnostic, so this rides the MILA_ENABLE_CUDA=OFF CI gate.
 */

#include <gtest/gtest.h>
#include <string>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Components::Linear
{
    using namespace Mila::Dnn;
    using Mila::Dnn::Serialization::SerializationMetadata;

    class LinearConfigTests : public ::testing::Test
    {
    };

    // ====================================================================
    // A. Construction & Defaults
    // ====================================================================

    TEST_F( LinearConfigTests, Construct_StoresFeatureDimensions )
    {
        LinearConfig config( 128, 64 );

        EXPECT_EQ( config.getInputFeatures(), 128 );
        EXPECT_EQ( config.getOutputFeatures(), 64 );
    }

    TEST_F( LinearConfigTests, Construct_BiasDefaultsTrue )
    {
        LinearConfig config( 128, 64 );

        EXPECT_TRUE( config.hasBias() );
    }

    // ====================================================================
    // Fluent interface
    // ====================================================================

    TEST_F( LinearConfigTests, WithBias_SetsValue )
    {
        LinearConfig config( 128, 64 );
        config.withBias( false );

        EXPECT_FALSE( config.hasBias() );
    }

    TEST_F( LinearConfigTests, WithBias_ReturnsSameObjectOnLvalue )
    {
        LinearConfig config( 128, 64 );
        auto&& result = config.withBias( false );

        EXPECT_EQ( &result, &config );
    }

    TEST_F( LinearConfigTests, WithBias_MultipleCallsTakeLast )
    {
        LinearConfig config( 128, 64 );
        config.withBias( false );
        config.withBias( true );

        EXPECT_TRUE( config.hasBias() );
    }

    TEST_F( LinearConfigTests, WithInputFeatures_SetsValue )
    {
        LinearConfig config( 128, 64 );
        config.withInputFeatures( 256 );

        EXPECT_EQ( config.getInputFeatures(), 256 );
    }

    TEST_F( LinearConfigTests, WithOutputFeatures_SetsValue )
    {
        LinearConfig config( 128, 64 );
        config.withOutputFeatures( 512 );

        EXPECT_EQ( config.getOutputFeatures(), 512 );
    }

    TEST_F( LinearConfigTests, Fluent_ChainsOnRvalue )
    {
        auto config = LinearConfig( 8, 4 )
            .withInputFeatures( 256 )
            .withOutputFeatures( 128 )
            .withBias( false );

        EXPECT_EQ( config.getInputFeatures(), 256 );
        EXPECT_EQ( config.getOutputFeatures(), 128 );
        EXPECT_FALSE( config.hasBias() );
    }

    // ====================================================================
    // A. Validation
    // ====================================================================

    TEST_F( LinearConfigTests, Validate_PassesForPositiveFeatures )
    {
        LinearConfig config( 128, 64 );

        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( LinearConfigTests, Validate_PassesForSingleFeature )
    {
        LinearConfig config( 1, 1 );

        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( LinearConfigTests, Validate_ThrowsForZeroInputFeatures )
    {
        LinearConfig config( 0, 64 );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( LinearConfigTests, Validate_ThrowsForZeroOutputFeatures )
    {
        LinearConfig config( 128, 0 );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( LinearConfigTests, Validate_ThrowsForBothZero )
    {
        LinearConfig config( 0, 0 );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( LinearConfigTests, Validate_ErrorMentionsGreaterThanZero )
    {
        LinearConfig config( 0, 64 );

        try
        {
            config.validate();
            FAIL() << "expected std::invalid_argument";
        }
        catch ( const std::invalid_argument& e )
        {
            EXPECT_NE( std::string( e.what() ).find( "greater than zero" ), std::string::npos );
        }
    }

    // ====================================================================
    // H. Serialization round-trip
    // ====================================================================

    TEST_F( LinearConfigTests, Metadata_RoundTripPreservesFields )
    {
        LinearConfig source( 768, 3072 );
        source.withBias( false );

        SerializationMetadata meta = source.toMetadata();

        LinearConfig loaded( 1, 1 );
        loaded.fromMetadata( meta );

        EXPECT_EQ( loaded.getInputFeatures(), 768 );
        EXPECT_EQ( loaded.getOutputFeatures(), 3072 );
        EXPECT_FALSE( loaded.hasBias() );
    }

    TEST_F( LinearConfigTests, FromMetadata_IgnoresMissingKeys )
    {
        LinearConfig config( 256, 128 );
        config.withBias( false );

        SerializationMetadata empty;
        config.fromMetadata( empty );

        EXPECT_EQ( config.getInputFeatures(), 256 );
        EXPECT_EQ( config.getOutputFeatures(), 128 );
        EXPECT_FALSE( config.hasBias() );
    }

    // ====================================================================
    // I. Diagnostics
    // ====================================================================

    TEST_F( LinearConfigTests, ToString_ContainsFeatureDimensions )
    {
        LinearConfig config( 256, 128 );

        const std::string text = config.toString();

        EXPECT_NE( text.find( "LinearConfig" ), std::string::npos );
        EXPECT_NE( text.find( "256" ), std::string::npos );
        EXPECT_NE( text.find( "128" ), std::string::npos );
    }
}
