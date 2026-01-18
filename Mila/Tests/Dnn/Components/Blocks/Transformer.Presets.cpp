#include <gtest/gtest.h>

import Mila;

namespace CompositeComponents_Tests
{
    using namespace Mila::Dnn;

    TEST( TransformerPresets, GPT2SmallProperties )
    {
        // Verify primary dimensions and basic options exposed by the GPT-2 small preset.
        auto cfg = GPT2_Small();

        EXPECT_EQ( cfg.getEmbeddingDim(), 768 );
        EXPECT_EQ( cfg.getNumHeads(), 12 );

        // Hidden dimension explicitly set in preset.
        EXPECT_EQ( cfg.getHiddenDimension(), 3072 );

        EXPECT_TRUE( cfg.useBias() );

        EXPECT_EQ( cfg.getActivationType(), ActivationType::Gelu );

        // Defaults for GPT-2 family
        EXPECT_EQ( cfg.getNormType(), NormType::LayerNorm );
        EXPECT_EQ( cfg.getAttentionType(), AttentionType::Standard );
        EXPECT_EQ( cfg.getEncodingType(), EncodingType::Learned );

        // Preset should validate without throwing.
        EXPECT_NO_THROW( cfg.validate() );
    }

    TEST( TransformerPresets, Llama3_8BProperties )
    {
        // Llama 3 (8B) preset uses GQA, RMSNorm, RoPE and no bias.
        auto cfg = Llama3_8B();

        EXPECT_EQ( cfg.getEmbeddingDim(), 4096 );
        EXPECT_EQ( cfg.getNumHeads(), 32 );

        EXPECT_EQ( cfg.getHiddenDimension(), 14336 );

        EXPECT_FALSE( cfg.useBias() );

        EXPECT_EQ( cfg.getActivationType(), ActivationType::Swiglu );

        EXPECT_EQ( cfg.getNormType(), NormType::RMSNorm );
        EXPECT_EQ( cfg.getAttentionType(), AttentionType::GroupedQuery );

        // KV heads configured for GQA
        EXPECT_EQ( cfg.getNumKVHeads(), 8 );

        EXPECT_EQ( cfg.getEncodingType(), EncodingType::RoPE );

        // RoPE theta explicitly set in preset
        EXPECT_FLOAT_EQ( cfg.getRoPETheta(), 500000.0f );

        // Max sequence length set
        EXPECT_EQ( cfg.getMaxSequenceLength(), static_cast<Mila::Dnn::dim_t>(8192) );

        EXPECT_NO_THROW( cfg.validate() );
    }

    TEST( TransformerPresets, Llama3_2_1BProperties )
    {
        // Lightweight Llama 3.2 1B preset checks.
        auto cfg = Llama3_2_1B();

        EXPECT_EQ( cfg.getEmbeddingDim(), 2048 );
        EXPECT_EQ( cfg.getNumHeads(), 32 );

        EXPECT_EQ( cfg.getHiddenDimension(), 8192 );

        EXPECT_FALSE( cfg.useBias() );

        EXPECT_EQ( cfg.getActivationType(), ActivationType::Swiglu );

        EXPECT_NO_THROW( cfg.validate() );
    }
}