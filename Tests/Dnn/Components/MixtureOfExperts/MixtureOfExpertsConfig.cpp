/**
 * @file MixtureOfExpertsConfig.cpp
 * @brief Config-archetype tests for MixtureOfExpertsConfig (see Specifications/Testing.md).
 */

#include <gtest/gtest.h>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Components::MixtureOfExperts
{
    using namespace Mila::Dnn;

    TEST( MixtureOfExpertsConfigTests, Construct_StoresDimensions )
    {
        const MixtureOfExpertsConfig config( 2816, 704, 128, 8 );

        EXPECT_EQ( config.getHiddenSize(), 2816 );
        EXPECT_EQ( config.getExpertIntermediateSize(), 704 );
        EXPECT_EQ( config.getNumExperts(), 128 );
        EXPECT_EQ( config.getTopK(), 8 );
    }

    TEST( MixtureOfExpertsConfigTests, Construct_ThrowsForNonPositiveWidths )
    {
        EXPECT_THROW( MixtureOfExpertsConfig( 0, 704, 128, 8 ), std::invalid_argument );
        EXPECT_THROW( MixtureOfExpertsConfig( 2816, 0, 128, 8 ), std::invalid_argument );
    }

    TEST( MixtureOfExpertsConfigTests, Construct_ThrowsForNonPositiveExpertCount )
    {
        EXPECT_THROW( MixtureOfExpertsConfig( 2816, 704, 0, 1 ), std::invalid_argument );
    }

    TEST( MixtureOfExpertsConfigTests, Construct_ThrowsForTopKOutsideExpertRange )
    {
        EXPECT_THROW( MixtureOfExpertsConfig( 2816, 704, 128, 0 ), std::invalid_argument );
        EXPECT_THROW( MixtureOfExpertsConfig( 2816, 704, 128, 129 ), std::invalid_argument );
    }

    TEST( MixtureOfExpertsConfigTests, Metadata_RoundTripPreservesFields )
    {
        const MixtureOfExpertsConfig source( 2816, 704, 128, 8 );
        MixtureOfExpertsConfig restored( 1, 1, 1, 1 );

        restored.fromMetadata( source.toMetadata() );

        EXPECT_EQ( restored.getHiddenSize(), 2816 );
        EXPECT_EQ( restored.getExpertIntermediateSize(), 704 );
        EXPECT_EQ( restored.getNumExperts(), 128 );
        EXPECT_EQ( restored.getTopK(), 8 );
    }

    TEST( MixtureOfExpertsConfigTests, ToString_NonEmpty )
    {
        EXPECT_FALSE( MixtureOfExpertsConfig( 2816, 704, 128, 8 ).toString().empty() );
    }
}
