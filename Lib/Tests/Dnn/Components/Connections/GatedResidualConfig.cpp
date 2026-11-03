/**
 * @file GatedResidualConfig.cpp
 * @brief Unit tests for GatedResidualConfig: geometry, validation, serialization.
 */

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

import Mila;

namespace Mila::Tests::Dnn::Components::Connections
{
    using namespace Mila::Dnn;

    TEST( GatedResidualConfigTests, Geometry_StreamWidthIsStreamsTimesModelDim )
    {
        GatedResidualConfig config( 256, 4, 32 );

        EXPECT_EQ( config.getStreamWidth(), 1024 );
        EXPECT_TRUE( config.hasInjection() );
    }

    TEST( GatedResidualConfigTests, Validate_RefusesASingleStream )
    {
        EXPECT_THROW( GatedResidualConfig( 256, 1, 32 ), std::invalid_argument );
    }

    TEST( GatedResidualConfigTests, Validate_RefusesNonPositiveRank )
    {
        EXPECT_THROW( GatedResidualConfig( 256, 4, 0 ), std::invalid_argument );
    }

    TEST( GatedResidualConfigTests, Metadata_RoundTrips )
    {
        auto source = GatedResidualConfig( 256, 4, 32 ).withInjection( false ).withEpsilon( 1e-5f );

        GatedResidualConfig loaded( 1, 2, 1 );
        loaded.fromMetadata( source.toMetadata() );

        EXPECT_EQ( loaded.getModelDim(), 256 );
        EXPECT_EQ( loaded.getStreams(), 4 );
        EXPECT_EQ( loaded.getRank(), 32 );
        EXPECT_FALSE( loaded.hasInjection() );
        EXPECT_FLOAT_EQ( loaded.getEpsilon(), 1e-5f );
    }
}
