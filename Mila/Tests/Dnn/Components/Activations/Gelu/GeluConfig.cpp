/**
 * @file GeluConfig.cpp
 * @brief Config-archetype tests for GeluConfig.
 *
 * Reference instance of the config test archetype (see Specifications/Testing.md):
 * defaults, fluent setters, validation, and the metadata round-trip. No
 * forward/backward — that is the concrete-component archetype (Gelu.Cpu.cpp).
 */

#include <gtest/gtest.h>
#include <string>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Components::Activations::Gelu
{
    using namespace Mila::Dnn;
    using Mila::Dnn::Serialization::SerializationMetadata;

    class GeluConfigTests : public ::testing::Test
    {
    };

    // ====================================================================
    // A. Construction & Defaults
    // ====================================================================

    TEST_F( GeluConfigTests, Default_IsTanh )
    {
        GeluConfig config;

        EXPECT_EQ( config.getApproximationMethod(), ApproximationMethod::Tanh );
    }

    TEST_F( GeluConfigTests, EnumValues_AreStable )
    {
        // The numeric values are part of the serialized wire format.
        EXPECT_EQ( static_cast<int>( ApproximationMethod::Exact ), 0 );
        EXPECT_EQ( static_cast<int>( ApproximationMethod::Tanh ), 1 );
        EXPECT_EQ( static_cast<int>( ApproximationMethod::Sigmoid ), 2 );
    }

    // ====================================================================
    // Fluent interface
    // ====================================================================

    TEST_F( GeluConfigTests, WithApproximationMethod_SetsValue )
    {
        GeluConfig config;
        config.withApproximationMethod( ApproximationMethod::Exact );

        EXPECT_EQ( config.getApproximationMethod(), ApproximationMethod::Exact );
    }

    TEST_F( GeluConfigTests, WithApproximationMethod_ReturnsSameObjectOnLvalue )
    {
        GeluConfig config;
        auto&& result = config.withApproximationMethod( ApproximationMethod::Sigmoid );

        EXPECT_EQ( &result, &config );
    }

    TEST_F( GeluConfigTests, WithApproximationMethod_ChainsOnRvalue )
    {
        auto config = GeluConfig().withApproximationMethod( ApproximationMethod::Exact );

        EXPECT_EQ( config.getApproximationMethod(), ApproximationMethod::Exact );
    }

    TEST_F( GeluConfigTests, WithApproximationMethod_MultipleCallsTakeLast )
    {
        GeluConfig config;
        config.withApproximationMethod( ApproximationMethod::Exact );
        config.withApproximationMethod( ApproximationMethod::Tanh );

        EXPECT_EQ( config.getApproximationMethod(), ApproximationMethod::Tanh );
    }

    // ====================================================================
    // A. Validation (only Tanh is currently supported)
    // ====================================================================

    TEST_F( GeluConfigTests, Validate_PassesForTanh )
    {
        GeluConfig config;

        EXPECT_NO_THROW( config.validate() );
    }

    TEST_F( GeluConfigTests, Validate_ThrowsForExact )
    {
        GeluConfig config;
        config.withApproximationMethod( ApproximationMethod::Exact );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( GeluConfigTests, Validate_ThrowsForSigmoid )
    {
        GeluConfig config;
        config.withApproximationMethod( ApproximationMethod::Sigmoid );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST_F( GeluConfigTests, Validate_ErrorMentionsTanh )
    {
        GeluConfig config;
        config.withApproximationMethod( ApproximationMethod::Exact );

        try
        {
            config.validate();
            FAIL() << "expected std::invalid_argument";
        }
        catch ( const std::invalid_argument& e )
        {
            EXPECT_NE( std::string( e.what() ).find( "Tanh" ), std::string::npos );
        }
    }

    // ====================================================================
    // H. Serialization round-trip
    // ====================================================================

    TEST_F( GeluConfigTests, Metadata_RoundTripPreservesMethod )
    {
        GeluConfig source;
        source.withApproximationMethod( ApproximationMethod::Sigmoid );

        SerializationMetadata meta = source.toMetadata();

        GeluConfig loaded;
        loaded.fromMetadata( meta );

        EXPECT_EQ( loaded.getApproximationMethod(), ApproximationMethod::Sigmoid );
    }

    TEST_F( GeluConfigTests, FromMetadata_IgnoresMissingKeys )
    {
        GeluConfig config;
        config.withApproximationMethod( ApproximationMethod::Exact );

        SerializationMetadata empty;
        config.fromMetadata( empty );

        EXPECT_EQ( config.getApproximationMethod(), ApproximationMethod::Exact );
    }

    TEST_F( GeluConfigTests, FromMetadata_UnknownMethodThrows )
    {
        SerializationMetadata meta;
        meta.set( "approximation_method", std::string( "Bogus" ) );

        GeluConfig config;

        EXPECT_THROW( config.fromMetadata( meta ), std::invalid_argument );
    }

    // ====================================================================
    // I. Diagnostics
    // ====================================================================

    TEST_F( GeluConfigTests, ToString_NonEmpty )
    {
        GeluConfig config;

        EXPECT_FALSE( config.toString().empty() );
    }
}
