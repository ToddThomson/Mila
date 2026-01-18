#include <gtest/gtest.h>

import Mila;

namespace CompositeComponents_Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    TEST( TransformerPresetsCreation, ConstructAndBuild_GPT2Small_Standalone )
    {
        // Create config from preset
        TransformerConfig cfg = GPT2_Small();

        // Basic preset sanity
        dim_t embedding_dim = cfg.getEmbeddingDim();
        dim_t num_heads = cfg.getNumHeads();

        ASSERT_GT( embedding_dim, 0 );
        ASSERT_GT( num_heads, 0 );
        ASSERT_EQ( cfg.getActivationType(), ActivationType::Gelu );

        // Construct standalone Transformer bound to CPU device
        std::shared_ptr<Transformer<DeviceType::Cpu, TensorDataType::FP32>> transformer;
        EXPECT_NO_THROW(
            ( transformer = std::make_shared<Transformer<DeviceType::Cpu, TensorDataType::FP32>>(
                "gpt2_preset_transformer",
                cfg,
                Device::Cpu()
            ) )
        );

        ASSERT_NE( transformer, nullptr );

        // Build with a small test shape: {batch, seq_len, embedding_dim}
        shape_t shape = { 2, 8, embedding_dim };

        EXPECT_NO_THROW( transformer->build( shape ) );

        // After build the component should report built
        EXPECT_TRUE( transformer->isBuilt() );

        // Device checks
        EXPECT_EQ( transformer->getDeviceType(), DeviceType::Cpu );
        EXPECT_EQ( transformer->getDeviceId().type, DeviceType::Cpu );
    }

    TEST( TransformerPresetsCreation, ConstructAndBuild_GPT2Small_NoDeviceId_SharedContext )
    {
        // Construct transformer without owning execution context (shared context mode)
        TransformerConfig cfg = GPT2_Small();

        std::shared_ptr<Transformer<DeviceType::Cpu, TensorDataType::FP32>> transformer;
        EXPECT_NO_THROW(
            ( transformer = std::make_shared<Transformer<DeviceType::Cpu, TensorDataType::FP32>>(
                "gpt2_preset_transformer_shared",
                cfg
            ) )
        );

        ASSERT_NE( transformer, nullptr );

        // Building without an execution context should throw since no context is set
        shape_t shape = { 1, 4, cfg.getEmbeddingDim() };
        EXPECT_THROW( transformer->build( shape ), std::runtime_error );
    }
}