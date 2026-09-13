/**
 * @file RouterConfig.cpp
 * @brief Config-archetype tests for RouterConfig (see Specifications/Testing.md).
 */

#include <gtest/gtest.h>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Components::MixtureOfExperts
{
    using namespace Mila::Dnn;

    TEST( RouterConfigTests, Construct_StoresDimensions )
    {
        const RouterConfig config( 2816, 128, 8 );

        EXPECT_EQ( config.getHiddenSize(), 2816 );
        EXPECT_EQ( config.getNumExperts(), 128 );
        EXPECT_EQ( config.getTopK(), 8 );
    }

    TEST( RouterConfigTests, Default_EpsilonIsGemmaRmsNormEpsilon )
    {
        EXPECT_FLOAT_EQ( RouterConfig( 2816, 128, 8 ).getEpsilon(), 1e-6f );
    }

    TEST( RouterConfigTests, WithEpsilon_SetsValue )
    {
        EXPECT_FLOAT_EQ( RouterConfig( 2816, 128, 8 ).withEpsilon( 1e-5f ).getEpsilon(), 1e-5f );
    }

    TEST( RouterConfigTests, Construct_ThrowsForNonPositiveHiddenSize )
    {
        EXPECT_THROW( RouterConfig( 0, 128, 8 ), std::invalid_argument );
    }

    TEST( RouterConfigTests, Construct_ThrowsForNonPositiveExpertCount )
    {
        EXPECT_THROW( RouterConfig( 2816, 0, 1 ), std::invalid_argument );
    }

    TEST( RouterConfigTests, Construct_ThrowsForTopKOutsideExpertRange )
    {
        EXPECT_THROW( RouterConfig( 2816, 128, 0 ), std::invalid_argument );
        EXPECT_THROW( RouterConfig( 2816, 128, 129 ), std::invalid_argument );
    }

    TEST( RouterConfigTests, WithEpsilon_ThrowsForNegative )
    {
        EXPECT_THROW( RouterConfig( 2816, 128, 8 ).withEpsilon( -1.0f ), std::invalid_argument );
    }

    TEST( RouterConfigTests, Metadata_RoundTripPreservesFields )
    {
        const RouterConfig source = RouterConfig( 2816, 128, 8 ).withEpsilon( 1e-5f );
        RouterConfig restored( 1, 1, 1 );

        restored.fromMetadata( source.toMetadata() );

        EXPECT_EQ( restored.getHiddenSize(), 2816 );
        EXPECT_EQ( restored.getNumExperts(), 128 );
        EXPECT_EQ( restored.getTopK(), 8 );
        EXPECT_FLOAT_EQ( restored.getEpsilon(), 1e-5f );
    }

    TEST( RouterConfigTests, ToString_NonEmpty )
    {
        EXPECT_FALSE( RouterConfig( 2816, 128, 8 ).toString().empty() );
    }
}
