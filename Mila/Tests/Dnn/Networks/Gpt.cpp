#include <gtest/gtest.h>

import Mila;

namespace CompositeComponents_Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Networks;
    using namespace Mila::Dnn::Compute;

    TEST( GptPresetsCreation, ConstructAndBuild_GPT2Small_Standalone )
    {
        // Create config from preset
        GptConfig cfg = GPT2_Small();

        // Basic preset sanity
        dim_t embedding_dim = cfg.getEmbeddingSize();
        dim_t num_heads = cfg.getNumHeads();

        ASSERT_GT( embedding_dim, 0 );
        ASSERT_GT( num_heads, 0 );

        // Construct GPT network bound to CPU device
        std::shared_ptr<Gpt<DeviceType::Cpu, TensorDataType::FP32>> gpt;

        EXPECT_NO_THROW(
            ( gpt = std::make_shared<Gpt<DeviceType::Cpu, TensorDataType::FP32>>(
                "gpt2_small_preset",
                cfg,
                Device::Cpu()
            ) )
        );

        ASSERT_NE( gpt, nullptr );

        // Build with a small test shape: {batch, seq_len, embedding_dim}
        shape_t shape = { 2, 8, embedding_dim };

        EXPECT_NO_THROW( gpt->build( shape ) );

        // After build the component should report built
        EXPECT_TRUE( gpt->isBuilt() );

        // Device checks
        EXPECT_EQ( gpt->getDeviceType(), DeviceType::Cpu );
        EXPECT_EQ( gpt->getDeviceId().type, DeviceType::Cpu );
    }

    TEST( GptPresetsCreation, ConstructAndBuild_GPT2Small_NoDeviceId_SharedContext )
    {
        // Construct GPT without owning execution context (shared context mode)
        GptConfig cfg = GPT2_Small();

        std::shared_ptr<Gpt<DeviceType::Cpu, TensorDataType::FP32>> gpt;
        EXPECT_NO_THROW(
            ( gpt = std::make_shared<Gpt<DeviceType::Cpu, TensorDataType::FP32>>(
                "gpt2_small_preset",
                cfg,
                Device::Cpu()
            ) )
        );

        ASSERT_NE( gpt, nullptr );

        // Building without an execution context should throw since no context is set
        shape_t shape = { 1, 4, cfg.getEmbeddingSize() };
        EXPECT_THROW( gpt->build( shape ), std::runtime_error );
    }
}