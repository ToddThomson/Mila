#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <string>

import Mila;

namespace Modules::Layers::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class AttentionConfigTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
        }
    };

    TEST_F( AttentionConfigTests, ConstructorSetsValues )
    {
        int64_t embedding_dim = 64;
        int64_t num_heads = 8;

        AttentionConfig cfg( embedding_dim, num_heads );

        EXPECT_EQ( cfg.getEmbeddingDim(), embedding_dim );
        EXPECT_EQ( cfg.getNumHeads(), num_heads );
    }

    TEST_F( AttentionConfigTests, FluentBaseSettersWork )
    {
        AttentionConfig cfg( 128, 8 );

        cfg.withName( "test_attention" )
            .withPrecisionPolicy( ComputePrecision::Policy::Performance );

        EXPECT_EQ( cfg.getName(), "test_attention" );
        EXPECT_EQ( cfg.getPrecisionPolicy(), ComputePrecision::Policy::Performance );
    }

    TEST_F( AttentionConfigTests, ValidationSuccess )
    {
        AttentionConfig cfg( 768, 12 );
        cfg.withName( "valid_attention" );

        EXPECT_NO_THROW( cfg.validate() );
    }

    TEST_F( AttentionConfigTests, ValidationFailure_ZeroEmbeddingDim )
    {
        AttentionConfig cfg( 0, 8 );
        cfg.withName( "bad_embedding" );

        EXPECT_THROW( cfg.validate(), std::invalid_argument );
    }

    TEST_F( AttentionConfigTests, ValidationFailure_ZeroNumHeads )
    {
        AttentionConfig cfg( 64, 0 );
        cfg.withName( "bad_heads" );

        EXPECT_THROW( cfg.validate(), std::invalid_argument );
    }

    TEST_F( AttentionConfigTests, ValidationFailure_NotDivisible )
    {
        AttentionConfig cfg( 65, 8 );
        cfg.withName( "not_divisible" );

        try
        {
            cfg.validate();
            FAIL() << "Expected std::invalid_argument due to embedding_dim % num_heads != 0";
        }
        catch (const std::invalid_argument& e)
        {
            std::string msg = e.what();
            EXPECT_NE( msg.find( "divisible" ), std::string::npos );
        }
    }

    TEST_F( AttentionConfigTests, CopyPreservesValues )
    {
        AttentionConfig cfg( 256, 8 );
        cfg.withName( "persistent_attention" )
            .withPrecisionPolicy( ComputePrecision::Policy::Accuracy );

        AttentionConfig copy = cfg;

        EXPECT_EQ( copy.getEmbeddingDim(), cfg.getEmbeddingDim() );
        EXPECT_EQ( copy.getNumHeads(), cfg.getNumHeads() );
        EXPECT_EQ( copy.getName(), cfg.getName() );
        EXPECT_EQ( copy.getPrecisionPolicy(), cfg.getPrecisionPolicy() );
    }

    TEST_F( AttentionConfigTests, EdgeCases_MinimalValid )
    {
        AttentionConfig cfg( 1, 1 );
        cfg.withName( "minimal_attention" );

        EXPECT_NO_THROW( cfg.validate() );
        EXPECT_EQ( cfg.getEmbeddingDim(), 1 );
        EXPECT_EQ( cfg.getNumHeads(), 1 );
    }

    TEST_F( AttentionConfigTests, EdgeCases_LargeValues )
    {
        AttentionConfig cfg( 4096, 16 );
        cfg.withName( "large_attention" );

        EXPECT_NO_THROW( cfg.validate() );
        EXPECT_EQ( cfg.getEmbeddingDim(), 4096 );
        EXPECT_EQ( cfg.getNumHeads(), 16 );
    }
}