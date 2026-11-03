/**
 * @file Gpt.Presets.cpp
 * @brief Tests for the GPT-2 preset configuration factories.
 *
 * Each preset returns a canonical GptConfig (Small/Medium/Large/XL). Verifies the
 * published dimensions, the 4x-embedding hidden size, head divisibility, and that
 * each preset validates.
 */

#include <gtest/gtest.h>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Components::Transformers::Gpt
{
    using namespace Mila::Dnn;

    class GptPresetsTests : public ::testing::Test
    {
    };

    TEST_F( GptPresetsTests, GPT2Small_Dimensions )
    {
        auto cfg = GPT2_Small();

        EXPECT_EQ( cfg.getEmbeddingSize(), 768 );
        EXPECT_EQ( cfg.getNumLayers(), 12 );
        EXPECT_EQ( cfg.getNumHeads(), 12 );
        EXPECT_EQ( cfg.getHiddenSize(), 3072 );
        EXPECT_TRUE( cfg.getUseBias() );
        EXPECT_EQ( cfg.getMaxSequenceLength(), static_cast<dim_t>( 1024 ) );
        EXPECT_NO_THROW( cfg.validate() );
    }

    TEST_F( GptPresetsTests, GPT2Medium_Dimensions )
    {
        auto cfg = GPT2_Medium();

        EXPECT_EQ( cfg.getEmbeddingSize(), 1024 );
        EXPECT_EQ( cfg.getNumLayers(), 24 );
        EXPECT_EQ( cfg.getNumHeads(), 16 );
        EXPECT_EQ( cfg.getHiddenSize(), 4096 );
        EXPECT_TRUE( cfg.getUseBias() );
        EXPECT_NO_THROW( cfg.validate() );
    }

    TEST_F( GptPresetsTests, GPT2Large_Dimensions )
    {
        auto cfg = GPT2_Large();

        EXPECT_EQ( cfg.getEmbeddingSize(), 1280 );
        EXPECT_EQ( cfg.getNumLayers(), 36 );
        EXPECT_EQ( cfg.getNumHeads(), 20 );
        EXPECT_EQ( cfg.getHiddenSize(), 5120 );
        EXPECT_TRUE( cfg.getUseBias() );
        EXPECT_NO_THROW( cfg.validate() );
    }

    TEST_F( GptPresetsTests, GPT2XL_Dimensions )
    {
        auto cfg = GPT2_XL();

        EXPECT_EQ( cfg.getEmbeddingSize(), 1600 );
        EXPECT_EQ( cfg.getNumLayers(), 48 );
        EXPECT_EQ( cfg.getNumHeads(), 25 );
        EXPECT_EQ( cfg.getHiddenSize(), 6400 );
        EXPECT_TRUE( cfg.getUseBias() );
        EXPECT_NO_THROW( cfg.validate() );
    }

    TEST_F( GptPresetsTests, AllPresets_HiddenIsFourTimesEmbedding )
    {
        EXPECT_EQ( GPT2_Small().getHiddenSize(), 4 * GPT2_Small().getEmbeddingSize() );
        EXPECT_EQ( GPT2_Medium().getHiddenSize(), 4 * GPT2_Medium().getEmbeddingSize() );
        EXPECT_EQ( GPT2_Large().getHiddenSize(), 4 * GPT2_Large().getEmbeddingSize() );
        EXPECT_EQ( GPT2_XL().getHiddenSize(), 4 * GPT2_XL().getEmbeddingSize() );
    }

    TEST_F( GptPresetsTests, AllPresets_HeadsDivideEmbedding )
    {
        EXPECT_EQ( GPT2_Small().getEmbeddingSize() % GPT2_Small().getNumHeads(), 0 );
        EXPECT_EQ( GPT2_Medium().getEmbeddingSize() % GPT2_Medium().getNumHeads(), 0 );
        EXPECT_EQ( GPT2_Large().getEmbeddingSize() % GPT2_Large().getNumHeads(), 0 );
        EXPECT_EQ( GPT2_XL().getEmbeddingSize() % GPT2_XL().getNumHeads(), 0 );
    }
}
