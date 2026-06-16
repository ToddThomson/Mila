/**
 * @file GptBlock.Config.cpp
 * @brief Config-archetype tests for GptBlockConfig.
 *
 * Construction defaults, fluent setters, validation throws (constructor-eager and
 * via validate()), and the metadata round-trip. No forward/backward -- that is the
 * concrete-component archetype (GptBlock.Cpu.cpp / GptBlock.Cuda.cpp).
 */

#include <gtest/gtest.h>
#include <string>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Components::Transformers::Gpt
{
    using namespace Mila::Dnn;
    using Mila::Dnn::Serialization::SerializationMetadata;

    class GptBlockConfigTests : public ::testing::Test
    {
    };

    // ====================================================================
    // A. Construction & Defaults
    // ====================================================================

    TEST_F( GptBlockConfigTests, Construct_StoresModelDimAndHeads )
    {
        GptBlockConfig cfg( 64, 8 );

        EXPECT_EQ( cfg.getModelDim(), 64 );
        EXPECT_EQ( cfg.getNumHeads(), 8 );
    }

    TEST_F( GptBlockConfigTests, Default_BiasDisabled )
    {
        GptBlockConfig cfg( 64, 8 );

        EXPECT_FALSE( cfg.useBias() );
    }

    TEST_F( GptBlockConfigTests, Default_ActivationIsGelu )
    {
        GptBlockConfig cfg( 64, 8 );

        EXPECT_EQ( cfg.getActivationType(), ActivationType::Gelu );
    }

    TEST_F( GptBlockConfigTests, Default_HiddenSizeUnset )
    {
        GptBlockConfig cfg( 64, 8 );

        EXPECT_EQ( cfg.getHiddenSize(), 0 );
    }

    TEST_F( GptBlockConfigTests, Default_ResidualScaleIsOne )
    {
        GptBlockConfig cfg( 64, 8 );

        EXPECT_FLOAT_EQ( cfg.getResidualScale(), 1.0f );
    }

    TEST_F( GptBlockConfigTests, EffectiveHiddenDimension_DefaultsToFourTimesModelDim )
    {
        GptBlockConfig cfg( 64, 8 );

        EXPECT_EQ( cfg.getEffectiveHiddenDimension(), 256 );
    }

    TEST_F( GptBlockConfigTests, EffectiveHiddenDimension_UsesExplicitWhenSet )
    {
        GptBlockConfig cfg( 64, 8 );
        cfg.withHiddenSize( 128 );

        EXPECT_EQ( cfg.getEffectiveHiddenDimension(), 128 );
    }

    // ====================================================================
    // Fluent interface
    // ====================================================================

    TEST_F( GptBlockConfigTests, WithHiddenSize_SetsValue )
    {
        GptBlockConfig cfg( 32, 4 );
        cfg.withHiddenSize( 128 );

        EXPECT_EQ( cfg.getHiddenSize(), 128 );
    }

    TEST_F( GptBlockConfigTests, WithBias_SetsValue )
    {
        GptBlockConfig cfg( 32, 4 );
        cfg.withBias( true );

        EXPECT_TRUE( cfg.useBias() );
    }

    TEST_F( GptBlockConfigTests, WithActivation_SetsValue )
    {
        GptBlockConfig cfg( 32, 4 );
        cfg.withActivation( ActivationType::Relu );

        EXPECT_EQ( cfg.getActivationType(), ActivationType::Relu );
    }

    TEST_F( GptBlockConfigTests, WithResidualScale_SetsValue )
    {
        GptBlockConfig cfg( 32, 4 );
        cfg.withResidualScale( 0.5f );

        EXPECT_FLOAT_EQ( cfg.getResidualScale(), 0.5f );
    }

    TEST_F( GptBlockConfigTests, WithMaxSequenceLength_AcceptsPositive )
    {
        GptBlockConfig cfg( 32, 4 );

        EXPECT_NO_THROW( cfg.withMaxSequenceLength( 1024 ) );
    }

    TEST_F( GptBlockConfigTests, FluentChain_AppliesAllSetters )
    {
        GptBlockConfig cfg( 32, 4 );
        cfg.withHiddenSize( 128 )
            .withBias( true )
            .withActivation( ActivationType::Gelu )
            .withResidualScale( 0.25f );

        EXPECT_EQ( cfg.getHiddenSize(), 128 );
        EXPECT_TRUE( cfg.useBias() );
        EXPECT_EQ( cfg.getActivationType(), ActivationType::Gelu );
        EXPECT_FLOAT_EQ( cfg.getResidualScale(), 0.25f );
    }

    TEST_F( GptBlockConfigTests, WithHiddenSize_ReturnsSameObjectOnLvalue )
    {
        GptBlockConfig cfg( 32, 4 );
        auto&& result = cfg.withHiddenSize( 128 );

        EXPECT_EQ( &result, &cfg );
    }

    // ====================================================================
    // A. Validation
    // ====================================================================

    TEST_F( GptBlockConfigTests, Validate_PassesForValidConfig )
    {
        GptBlockConfig cfg( 64, 8 );

        EXPECT_NO_THROW( cfg.validate() );
    }

    TEST_F( GptBlockConfigTests, Construct_ThrowsWhenModelDimZero )
    {
        EXPECT_THROW( GptBlockConfig( 0, 4 ), std::invalid_argument );
    }

    TEST_F( GptBlockConfigTests, Construct_ThrowsWhenNumHeadsZero )
    {
        EXPECT_THROW( GptBlockConfig( 64, 0 ), std::invalid_argument );
    }

    TEST_F( GptBlockConfigTests, Construct_ThrowsWhenModelDimNotDivisibleByHeads )
    {
        EXPECT_THROW( GptBlockConfig( 65, 8 ), std::invalid_argument );
    }

    TEST_F( GptBlockConfigTests, WithMaxSequenceLength_ThrowsForNonPositive )
    {
        GptBlockConfig cfg( 32, 4 );

        EXPECT_THROW( cfg.withMaxSequenceLength( 0 ), std::invalid_argument );
    }

    // ====================================================================
    // H. Serialization round-trip
    // ====================================================================

    TEST_F( GptBlockConfigTests, Metadata_RoundTripPreservesFields )
    {
        GptBlockConfig source( 128, 8 );
        source.withHiddenSize( 512 )
            .withBias( true )
            .withActivation( ActivationType::Gelu )
            .withResidualScale( 0.5f )
            .withMaxSequenceLength( 1024 );

        SerializationMetadata meta = source.toMetadata();

        GptBlockConfig loaded( 64, 8 );
        loaded.fromMetadata( meta );

        EXPECT_EQ( loaded.getModelDim(), 128 );
        EXPECT_EQ( loaded.getNumHeads(), 8 );
        EXPECT_EQ( loaded.getHiddenSize(), 512 );
        EXPECT_TRUE( loaded.useBias() );
        EXPECT_EQ( loaded.getActivationType(), ActivationType::Gelu );
        EXPECT_FLOAT_EQ( loaded.getResidualScale(), 0.5f );
    }

    // ====================================================================
    // I. Diagnostics
    // ====================================================================

    TEST_F( GptBlockConfigTests, ToString_NonEmpty )
    {
        GptBlockConfig cfg( 64, 8 );

        EXPECT_FALSE( cfg.toString().empty() );
    }

    // ====================================================================
    // J. Accessor traits
    // ====================================================================

    TEST_F( GptBlockConfigTests, Getters_AreNoexcept )
    {
        GptBlockConfig cfg( 64, 8 );

        EXPECT_TRUE( noexcept( cfg.getModelDim() ) );
        EXPECT_TRUE( noexcept( cfg.getNumHeads() ) );
        EXPECT_TRUE( noexcept( cfg.getHiddenSize() ) );
        EXPECT_TRUE( noexcept( cfg.useBias() ) );
        EXPECT_TRUE( noexcept( cfg.getActivationType() ) );
        EXPECT_TRUE( noexcept( cfg.getResidualScale() ) );
        EXPECT_TRUE( noexcept( cfg.getEffectiveHiddenDimension() ) );
    }
}
