/**
 * @file Llama.Presets.cpp
 * @brief Tests for the LLaMA preset configuration factories.
 *
 * Each preset returns a canonical LlamaConfig (3.2 1B/3B, 3.1 8B, 3 8B, 2 7B).
 * Verifies the published dimensions, the GQA KV-head ratio, RoPE theta/scaling,
 * the bias-free default, and that each preset validates and respects the
 * head-divisibility invariants.
 */

#include <gtest/gtest.h>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Components::Transformers::Llama
{
    using namespace Mila::Dnn;

    class LlamaPresetsTests : public ::testing::Test
    {
    };

    TEST_F( LlamaPresetsTests, Llama3_2_1B_Dimensions )
    {
        auto cfg = Llama3_2_1B();

        EXPECT_EQ( cfg.getModelDim(), 2048 );
        EXPECT_EQ( cfg.getNumLayers(), 16 );
        EXPECT_EQ( cfg.getNumHeads(), 32 );
        EXPECT_EQ( cfg.getNumKVHeads(), 8 );
        EXPECT_EQ( cfg.getHiddenDimension(), 8192 );
        EXPECT_FLOAT_EQ( cfg.getRoPETheta(), 500000.0f );
        EXPECT_FLOAT_EQ( cfg.getRoPEScalingFactor(), 32.0f );
        EXPECT_FALSE( cfg.useBias() );
        EXPECT_NO_THROW( cfg.validate() );
    }

    TEST_F( LlamaPresetsTests, Llama3_2_3B_Dimensions )
    {
        auto cfg = Llama3_2_3B();

        EXPECT_EQ( cfg.getModelDim(), 3072 );
        EXPECT_EQ( cfg.getNumLayers(), 28 );
        EXPECT_EQ( cfg.getNumHeads(), 24 );
        EXPECT_EQ( cfg.getNumKVHeads(), 8 );
        EXPECT_EQ( cfg.getHiddenDimension(), 8192 );
        EXPECT_FLOAT_EQ( cfg.getRoPETheta(), 500000.0f );
        EXPECT_FLOAT_EQ( cfg.getRoPEScalingFactor(), 32.0f );
        EXPECT_FALSE( cfg.useBias() );
        EXPECT_NO_THROW( cfg.validate() );
    }

    TEST_F( LlamaPresetsTests, Llama3_1_8B_Dimensions )
    {
        auto cfg = Llama3_1_8B();

        EXPECT_EQ( cfg.getModelDim(), 4096 );
        EXPECT_EQ( cfg.getNumLayers(), 32 );
        EXPECT_EQ( cfg.getNumHeads(), 32 );
        EXPECT_EQ( cfg.getNumKVHeads(), 8 );
        EXPECT_EQ( cfg.getHiddenDimension(), 14336 );
        EXPECT_FLOAT_EQ( cfg.getRoPETheta(), 500000.0f );
        EXPECT_FLOAT_EQ( cfg.getRoPEScalingFactor(), 8.0f );
        EXPECT_FALSE( cfg.useBias() );
        EXPECT_NO_THROW( cfg.validate() );
    }

    TEST_F( LlamaPresetsTests, Llama3_8B_Dimensions )
    {
        // Llama 3 (original 8B): same shape as 3.1 8B but native 8k context, no RoPE scaling.
        auto cfg = Llama3_8B();

        EXPECT_EQ( cfg.getModelDim(), 4096 );
        EXPECT_EQ( cfg.getNumLayers(), 32 );
        EXPECT_EQ( cfg.getNumHeads(), 32 );
        EXPECT_EQ( cfg.getNumKVHeads(), 8 );
        EXPECT_EQ( cfg.getHiddenDimension(), 14336 );
        EXPECT_FLOAT_EQ( cfg.getRoPETheta(), 500000.0f );
        EXPECT_EQ( cfg.getMaxSequenceLength(), static_cast<dim_t>( 8192 ) );
        EXPECT_FALSE( cfg.useBias() );
        EXPECT_NO_THROW( cfg.validate() );
    }

    TEST_F( LlamaPresetsTests, Llama2_7B_UsesMHAAndOriginalTheta )
    {
        // Llama 2 7B sets num_kv_heads == num_heads (effectively MHA) and theta=10000.
        auto cfg = Llama2_7B();

        EXPECT_EQ( cfg.getModelDim(), 4096 );
        EXPECT_EQ( cfg.getNumLayers(), 32 );
        EXPECT_EQ( cfg.getNumHeads(), 32 );
        EXPECT_EQ( cfg.getNumKVHeads(), 32 );
        EXPECT_EQ( cfg.getHiddenDimension(), 11008 );
        EXPECT_FLOAT_EQ( cfg.getRoPETheta(), 10000.0f );
        EXPECT_FALSE( cfg.useBias() );
        EXPECT_NO_THROW( cfg.validate() );
    }

    TEST_F( LlamaPresetsTests, AllPresets_HeadsDivideEmbedding )
    {
        EXPECT_EQ( Llama3_2_1B().getModelDim() % Llama3_2_1B().getNumHeads(), 0 );
        EXPECT_EQ( Llama3_2_3B().getModelDim() % Llama3_2_3B().getNumHeads(), 0 );
        EXPECT_EQ( Llama3_1_8B().getModelDim() % Llama3_1_8B().getNumHeads(), 0 );
        EXPECT_EQ( Llama3_8B().getModelDim() % Llama3_8B().getNumHeads(), 0 );
        EXPECT_EQ( Llama2_7B().getModelDim() % Llama2_7B().getNumHeads(), 0 );
    }

    TEST_F( LlamaPresetsTests, AllPresets_NumHeadsDivisibleByKVHeads )
    {
        EXPECT_EQ( Llama3_2_1B().getNumHeads() % Llama3_2_1B().getNumKVHeads(), 0 );
        EXPECT_EQ( Llama3_2_3B().getNumHeads() % Llama3_2_3B().getNumKVHeads(), 0 );
        EXPECT_EQ( Llama3_1_8B().getNumHeads() % Llama3_1_8B().getNumKVHeads(), 0 );
        EXPECT_EQ( Llama3_8B().getNumHeads() % Llama3_8B().getNumKVHeads(), 0 );
        EXPECT_EQ( Llama2_7B().getNumHeads() % Llama2_7B().getNumKVHeads(), 0 );
    }
}
