#include <gtest/gtest.h>

import Mila;

namespace Networks_Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Networks;

    TEST( LlamaPresets, Llama3_8B_Properties )
    {
        // Llama 3 (8B) preset uses GQA, RMSNorm, RoPE and no bias.
        auto cfg = Llama3_8B();

        EXPECT_EQ( cfg.getEmbeddingSize(), 4096 );
        EXPECT_EQ( cfg.getNumHeads(), 32 );

        EXPECT_EQ( cfg.getHiddenDimension(), 14336 );

        EXPECT_FALSE( cfg.useBias() );

        // KV heads configured for GQA
        EXPECT_EQ( cfg.getNumKVHeads(), 8 );

        // RoPE theta explicitly set in preset
        EXPECT_FLOAT_EQ( cfg.getRoPETheta(), 500000.0f );

        // Max sequence length set
        EXPECT_EQ( cfg.getMaxSequenceLength(), static_cast<Mila::Dnn::dim_t>(8192) );

        EXPECT_NO_THROW( cfg.validate() );
    }

    TEST( LlamaPresets, Llama3_2_1B_Properties )
    {
        // Lightweight Llama 3.2 1B preset checks.
        auto cfg = Llama3_2_1B();

        EXPECT_EQ( cfg.getEmbeddingSize(), 2048 );
        EXPECT_EQ( cfg.getNumHeads(), 32 );

        EXPECT_EQ( cfg.getHiddenDimension(), 8192 );

        EXPECT_FALSE( cfg.useBias() );

        EXPECT_NO_THROW( cfg.validate() );
    }
}