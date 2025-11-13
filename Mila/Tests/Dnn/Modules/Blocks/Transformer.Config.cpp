#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

import Mila;

namespace Modules::Blocks::Tests
{
    using namespace Mila::Dnn;

    TEST( TransformerConfigTests, DefaultsAndAccessors )
    {
        // embedding_dim = 64, num_heads = 8
        TransformerConfig cfg( /*embedding_dim=*/64, /*num_heads=*/8 );

        EXPECT_EQ( cfg.getEmbeddingDim(), 64 );
        EXPECT_EQ( cfg.getNumHeads(), 8 );
        EXPECT_EQ( cfg.getHiddenDimension(), 0 ); // not set -> 0
        EXPECT_TRUE( cfg.useBias() );
        EXPECT_EQ( cfg.getActivationType(), ActivationType::Gelu );

        // Default name comes from ConfigurationBase ("unnamed") so validate should succeed
        EXPECT_NO_THROW( cfg.validate() );
    }

    TEST( TransformerConfigTests, FluentSettersAndGetters )
    {
        TransformerConfig cfg( /*embedding_dim=*/32, /*num_heads=*/4 );

        // Chain fluent setters
        cfg.withHiddenDimension( 128 )
            .withBias( false )
            .withActivation( ActivationType::Relu )
            .withName( "transformer_A" );

        EXPECT_EQ( cfg.getHiddenDimension(), 128 );
        EXPECT_FALSE( cfg.useBias() );
        EXPECT_EQ( cfg.getActivationType(), ActivationType::Relu );
        EXPECT_EQ( cfg.getName(), "transformer_A" );

        EXPECT_NO_THROW( cfg.validate() );
    }

    TEST( TransformerConfigTests, Constructor_ThrowsWhenNumHeadsZero )
    {
        // Constructor performs immediate validation; expect it to throw.
        EXPECT_THROW( TransformerConfig( /*embedding_dim=*/64, /*num_heads=*/0 ), std::invalid_argument );
    }

    TEST( TransformerConfigTests, Constructor_ThrowsWhenEmbeddingDimZero )
    {
        // Constructor performs immediate validation; expect it to throw.
        EXPECT_THROW( TransformerConfig( /*embedding_dim=*/0, /*num_heads=*/4 ), std::invalid_argument );
    }

    TEST( TransformerConfigTests, Constructor_ThrowsWhenEmbeddingNotDivisibleByHeads )
    {
        // embedding 65 not divisible by 8 heads -> constructor should throw
        EXPECT_THROW( TransformerConfig( /*embedding_dim=*/65, /*num_heads=*/8 ), std::invalid_argument );
    }

    TEST( TransformerConfigTests, WithName_IsChainable )
    {
        TransformerConfig cfg( /*embedding_dim=*/16, /*num_heads=*/2 );

        auto& ref = cfg.withName( "named_transformer" );
        EXPECT_EQ( &ref, &cfg );
        EXPECT_EQ( cfg.getName(), "named_transformer" );

        EXPECT_NO_THROW( cfg.validate() );
    }
}