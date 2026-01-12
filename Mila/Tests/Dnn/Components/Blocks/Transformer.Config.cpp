#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

import Mila;

namespace CompositeComponents_Tests
{
    using namespace Mila::Dnn;

    TEST( TransformerConfigTests, DefaultsAndAccessors )
    {
        // embedding_dim = 64, num_heads = 8
        TransformerConfig cfg( /*embedding_dim=*/64, /*num_heads=*/8 );

        EXPECT_EQ( cfg.getEmbeddingDim(), 64 );
        EXPECT_EQ( cfg.getNumHeads(), 8 );
        EXPECT_EQ( cfg.getHiddenDimension(), 0 ); // not set -> 0
        EXPECT_FALSE( cfg.useBias() );
        EXPECT_EQ( cfg.getActivationType(), ActivationType::Gelu );

        // Default name comes from ComponentConfig ("unnamed") so validate should succeed
        EXPECT_NO_THROW( cfg.validate() );
    }

    TEST( TransformerConfigTests, FluentSettersAndGetters )
    {
        TransformerConfig cfg( /*embedding_dim=*/32, /*num_heads=*/4 );

        // Chain fluent setters (name is no longer set via the config)
        cfg.withHiddenDimension( 128 )
            .withBias( false )
            .withActivation( ActivationType::Relu );

        EXPECT_EQ( cfg.getHiddenDimension(), 128 );
        EXPECT_FALSE( cfg.useBias() );
        EXPECT_EQ( cfg.getActivationType(), ActivationType::Relu );

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
}