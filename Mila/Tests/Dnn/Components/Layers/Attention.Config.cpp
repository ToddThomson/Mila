#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <optional>

import Mila;

namespace Components::Layers::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    class AttentionConfigTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {}
    };

    TEST_F( AttentionConfigTests, ConstructorSetsValues )
    {
        int64_t embedding_dim = 64;
        int64_t num_heads = 8;

        AttentionConfig cfg( embedding_dim, num_heads );

        EXPECT_EQ( cfg.getModelDim(), embedding_dim );
        EXPECT_EQ( cfg.getNumHeads(), num_heads );
    }

    TEST_F( AttentionConfigTests, FluentBaseSettersWork )
    {
        AttentionConfig cfg( 128, 8 );

        // New fluent setters in the updated interface
        cfg.withModelDim( 256 )
            .withNumHeads( 16 );

        EXPECT_EQ( cfg.getModelDim(), 256 );
        EXPECT_EQ( cfg.getNumHeads(), 16 );
    }

    TEST_F( AttentionConfigTests, ValidationSuccess )
    {
        AttentionConfig cfg( 768, 12 );

        EXPECT_NO_THROW( cfg.validate() );
    }

    TEST_F( AttentionConfigTests, ValidationFailure_ZeroEmbeddingDim )
    {
        AttentionConfig cfg( 0, 8 );

        EXPECT_THROW( cfg.validate(), std::invalid_argument );
    }

    TEST_F( AttentionConfigTests, ValidationFailure_ZeroNumHeads )
    {
        AttentionConfig cfg( 64, 0 );

        EXPECT_THROW( cfg.validate(), std::invalid_argument );
    }

    TEST_F( AttentionConfigTests, ValidationFailure_NotDivisible )
    {
        AttentionConfig cfg( 65, 8 );

        try
        {
            cfg.validate();
            FAIL() << "Expected std::invalid_argument due to embedding_dim % num_heads != 0";
        }
        catch ( const std::invalid_argument& e )
        {
            std::string msg = e.what();
            EXPECT_NE( msg.find( "divisible" ), std::string::npos );
        }
    }

    TEST_F( AttentionConfigTests, CopyPreservesValues )
    {
        AttentionConfig cfg( 256, 8 );

        AttentionConfig copy = cfg;

        EXPECT_EQ( copy.getModelDim(), cfg.getModelDim() );
        EXPECT_EQ( copy.getNumHeads(), cfg.getNumHeads() );
    }

    TEST_F( AttentionConfigTests, MetadataRoundTrip )
    {
        AttentionConfig cfg( 512, 8 );

        auto meta = cfg.toMetadata();

        // Ensure metadata contains expected integer fields
        auto ed = meta.tryGetInt( "model_dim" );
        auto nh = meta.tryGetInt( "num_heads" );

        ASSERT_TRUE( ed.has_value() );
        ASSERT_TRUE( nh.has_value() );

        EXPECT_EQ( static_cast<int64_t>(*ed), 512 );
        EXPECT_EQ( static_cast<int64_t>(*nh), 8 );

        // Verify we can populate another config from the metadata
        AttentionConfig cfg2( 1, 1 );
        cfg2.fromMetadata( meta );

        EXPECT_EQ( cfg2.getModelDim(), 512 );
        EXPECT_EQ( cfg2.getNumHeads(), 8 );
    }

    TEST_F( AttentionConfigTests, ToStringContainsFields )
    {
        AttentionConfig cfg( 768, 12 );

        std::string s = cfg.toString();

        EXPECT_NE( s.find( "model_dim=768" ), std::string::npos );
        EXPECT_NE( s.find( "num_heads=12" ), std::string::npos );
    }

    TEST_F( AttentionConfigTests, EdgeCases_MinimalValid )
    {
        AttentionConfig cfg( 1, 1 );

        EXPECT_NO_THROW( cfg.validate() );
        EXPECT_EQ( cfg.getModelDim(), 1 );
        EXPECT_EQ( cfg.getNumHeads(), 1 );
    }

    TEST_F( AttentionConfigTests, EdgeCases_LargeValues )
    {
        AttentionConfig cfg( 4096, 16 );

        EXPECT_NO_THROW( cfg.validate() );
        EXPECT_EQ( cfg.getModelDim(), 4096 );
        EXPECT_EQ( cfg.getNumHeads(), 16 );
    }
}