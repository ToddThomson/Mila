/**
 * @file CausalConv1dConfig.cpp
 * @brief Unit tests for CausalConv1dConfig: defaults, dilation, the state-depth bound, serialization.
 */

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

import Mila;

namespace Mila::Tests::Dnn::Components::Convolutions
{
    using namespace Mila::Dnn;

    TEST( CausalConv1dConfigTests, Defaults_DilationOneNoBias )
    {
        CausalConv1dConfig config( 8, 4 );

        EXPECT_EQ( config.getDilation(), 1 );
        EXPECT_FALSE( config.hasBias() );
        EXPECT_EQ( config.getStateRows(), 3 );
    }

    TEST( CausalConv1dConfigTests, WithDilation_ScalesTheStateDepth )
    {
        auto config = CausalConv1dConfig( 8, 4 ).withDilation( 3 );

        EXPECT_EQ( config.getDilation(), 3 );
        EXPECT_EQ( config.getStateRows(), 9 );
        EXPECT_NO_THROW( config.validate() );
    }

    TEST( CausalConv1dConfigTests, Validate_ThrowsForNonPositiveDilation )
    {
        EXPECT_THROW( CausalConv1dConfig( 8, 4 ).withDilation( 0 ).validate(), std::invalid_argument );
    }

    TEST( CausalConv1dConfigTests, Validate_ThrowsWhenTheStateExceedsTheKernelBound )
    {
        // ( 4 - 1 ) * 6 = 18 retained rows, past the 16 the kernels stage in registers.
        EXPECT_THROW( CausalConv1dConfig( 8, 4 ).withDilation( 6 ).validate(), std::invalid_argument );
        EXPECT_NO_THROW( CausalConv1dConfig( 8, 9 ).withDilation( 2 ).validate() );
    }

    TEST( CausalConv1dConfigTests, Metadata_RoundTripsDilation )
    {
        auto source = CausalConv1dConfig( 8, 4 ).withDilation( 3 ).withBias( true );

        CausalConv1dConfig loaded( 1, 1 );
        loaded.fromMetadata( source.toMetadata() );

        EXPECT_EQ( loaded.getChannels(), 8 );
        EXPECT_EQ( loaded.getKernelWidth(), 4 );
        EXPECT_EQ( loaded.getDilation(), 3 );
        EXPECT_TRUE( loaded.hasBias() );
    }

    TEST( CausalConv1dConfigTests, ToString_NamesDilation )
    {
        auto config = CausalConv1dConfig( 8, 4 ).withDilation( 3 );

        EXPECT_NE( config.toString().find( "dilation=3" ), std::string::npos );
    }
}
