/**
 * @file GqaConfig.cpp
 * @brief Config-archetype tests for GqaConfig.
 *
 * Reference instance of the config archetype (see Specifications/Testing.md):
 * required-constructor fields (model_dim, num_heads, num_kv_heads), the derived
 * head_dim / group_size accessors, validate() (the full documented @throws set),
 * and the metadata round-trip. Device-agnostic, rides the CPU gate.
 */

#include <gtest/gtest.h>
#include <string>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Components::Attention::GQA
{
    using namespace Mila::Dnn;
    using Mila::Dnn::Serialization::SerializationMetadata;

    class GqaConfigTests : public ::testing::Test
    {
    };

    // ====================================================================
    // A. Construction & Derived accessors
    // ====================================================================

    TEST_F( GqaConfigTests, Construct_StoresFields )
    {
        GqaConfig config( 64, 8, 2 );

        EXPECT_EQ( config.getModelDim(), 64 );
        EXPECT_EQ( config.getNumHeads(), 8 );
        EXPECT_EQ( config.getNumKvHeads(), 2 );
    }

    TEST_F( GqaConfigTests, DerivedHeadDimAndGroupSize )
    {
        GqaConfig config( 64, 8, 2 );

        EXPECT_EQ( config.getHeadDim(), 8 );    // 64 / 8
        EXPECT_EQ( config.getGroupSize(), 4 );  // 8 / 2
    }

    // ====================================================================
    // Fluent interface
    // ====================================================================

    TEST_F( GqaConfigTests, FluentSetters_UpdateValues )
    {
        auto config = GqaConfig( 64, 8, 2 ).withModelDim( 128 ).withNumHeads( 16 ).withNumKvHeads( 4 );

        EXPECT_EQ( config.getModelDim(), 128 );
        EXPECT_EQ( config.getNumHeads(), 16 );
        EXPECT_EQ( config.getNumKvHeads(), 4 );
    }

    // ====================================================================
    // A. Validation (each documented @throws is one negative test)
    // ====================================================================

    TEST_F( GqaConfigTests, Validate_PassesForValid )
    {
        EXPECT_NO_THROW( GqaConfig( 64, 8, 2 ).validate() );
    }

    TEST_F( GqaConfigTests, Validate_PassesForMqa )
    {
        // num_kv_heads == 1 recovers Multi-Query Attention.
        EXPECT_NO_THROW( GqaConfig( 64, 8, 1 ).validate() );
    }

    TEST_F( GqaConfigTests, Validate_ThrowsForNonPositiveModelDim )
    {
        EXPECT_THROW( GqaConfig( 0, 8, 2 ).validate(), std::invalid_argument );
    }

    TEST_F( GqaConfigTests, Validate_ThrowsForTooFewHeads )
    {
        EXPECT_THROW( GqaConfig( 64, 1, 1 ).validate(), std::invalid_argument );
    }

    TEST_F( GqaConfigTests, Validate_ThrowsWhenModelDimNotDivisibleByHeads )
    {
        EXPECT_THROW( GqaConfig( 60, 8, 2 ).validate(), std::invalid_argument );
    }

    TEST_F( GqaConfigTests, Validate_ThrowsForZeroKvHeads )
    {
        EXPECT_THROW( GqaConfig( 64, 8, 0 ).validate(), std::invalid_argument );
    }

    TEST_F( GqaConfigTests, Validate_ThrowsWhenKvHeadsExceedHeads )
    {
        EXPECT_THROW( GqaConfig( 64, 8, 16 ).validate(), std::invalid_argument );
    }

    TEST_F( GqaConfigTests, Validate_ThrowsWhenHeadsNotDivisibleByKvHeads )
    {
        EXPECT_THROW( GqaConfig( 64, 8, 3 ).validate(), std::invalid_argument );
    }

    // ====================================================================
    // H. Serialization round-trip
    // ====================================================================

    TEST_F( GqaConfigTests, Metadata_RoundTripPreservesFields )
    {
        GqaConfig source( 128, 16, 4 );

        SerializationMetadata meta = source.toMetadata();

        GqaConfig loaded( 1, 1, 1 );
        loaded.fromMetadata( meta );

        EXPECT_EQ( loaded.getModelDim(), 128 );
        EXPECT_EQ( loaded.getNumHeads(), 16 );
        EXPECT_EQ( loaded.getNumKvHeads(), 4 );
    }

    // ====================================================================
    // I. Diagnostics
    // ====================================================================

    TEST_F( GqaConfigTests, ToString_NamesConfig )
    {
        GqaConfig config( 64, 8, 2 );

        EXPECT_NE( config.toString().find( "GQA Config" ), std::string::npos );
    }
}
