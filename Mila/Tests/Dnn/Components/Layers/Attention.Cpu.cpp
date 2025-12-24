#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>

import Mila;

namespace Components::Layers::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    template<TensorDataType TPrecision>
    using CpuTensor = Tensor<TPrecision, CpuMemoryResource>;

    class AttentionCpuTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
        }
    };

    TEST_F( AttentionCpuTests, BuildWithoutExecutionContext_Throws )
    {
        AttentionConfig cfg( 64, 8 );

        // Construct in deferred/shared mode (no DeviceId) and attempt to build
        auto module = std::make_shared<Attention<DeviceType::Cpu, TensorDataType::FP32>>( "attn_null", cfg );

        // Building without an execution context must fail per Component contract
        shape_t input_shape = { 1, 1, static_cast<int64_t>(3 * 64) };
        EXPECT_THROW( module->build( input_shape ), std::runtime_error );
    }

    TEST_F( AttentionCpuTests, ParameterCount_IsZero )
    {
        AttentionConfig cfg( 64, 8 );

        // Provide DeviceId to construct standalone component that creates its own ExecutionContext
        DeviceId dev{ DeviceType::Cpu, 0 };
        auto module = std::make_shared<Attention<DeviceType::Cpu, TensorDataType::FP32>>( "attn_params", cfg, dev );

        EXPECT_EQ( module->parameterCount(), 0u );
    }

    TEST_F( AttentionCpuTests, BuildAndForward )
    {
        const int64_t B = 2;
        const int64_t T = 4;
        const int64_t C = 64; // embedding dim
        const int64_t heads = 8;

        AttentionConfig cfg( C, heads );

        DeviceId dev{ DeviceType::Cpu, 0 };
        auto module = std::make_shared<Attention<DeviceType::Cpu, TensorDataType::FP32>>( "attn_forward", cfg, dev );

        // Model-layout shapes: input is concatenated Q||K||V -> [B, T, 3 * C], output -> [B, T, C]
        shape_t input_shape = { B, T, static_cast<int64_t>(3 * C) };
        shape_t output_shape = { B, T, static_cast<int64_t>(C) };

        // Build using model-layout shapes
        EXPECT_NO_THROW( module->build( input_shape ) );
        EXPECT_TRUE( module->isBuilt() );

        // Prepare concatenated QKV input and output tensors in model-layout
        CpuTensor<TensorDataType::FP32> input( dev, input_shape );
        CpuTensor<TensorDataType::FP32> output( dev, output_shape );

        // Fill deterministic values on CPU
        for ( size_t i = 0; i < input.size(); ++i )
        {
            input.data()[ i ] = static_cast<float>( (i % 100) ) * 0.01f;
        }

        // Forward should accept single concatenated input and produce model-layout output
        EXPECT_NO_THROW( module->forward( input, output ) );

        // Output shape / size checks
        EXPECT_EQ( output.shape(), output_shape );
        EXPECT_EQ( output.size(), static_cast<size_t>( B * T * C ) );
    }

    TEST_F( AttentionCpuTests, ToStringContainsInfo )
    {
        AttentionConfig cfg( 32, 4 );

        DeviceId dev{ DeviceType::Cpu, 0 };
        auto module = std::make_shared<Attention<DeviceType::Cpu, TensorDataType::FP32>>( "attn_info", cfg, dev );

        // Build with model-layout shape: input last-dim = 3 * embedding_dim (3*32 = 96)
        module->build( shape_t{ 1, 1, 96 } );

        auto s = module->toString();
        EXPECT_NE( s.find( "Attention" ), std::string::npos );
        EXPECT_NE( s.find( "Embedding dimension" ), std::string::npos );
        EXPECT_NE( s.find( "Number of heads" ), std::string::npos );
    }

    TEST_F( AttentionCpuTests, TrainingModeToggle )
    {
        AttentionConfig cfg( 64, 8 );

        DeviceId dev{ DeviceType::Cpu, 0 };
        auto module = std::make_shared<Attention<DeviceType::Cpu, TensorDataType::FP32>>( "attn_train_toggle", cfg, dev );
        module->build( shape_t{ 1, 1, 192 } );

        EXPECT_FALSE( module->isTraining() );

        module->setTraining( true );
        EXPECT_TRUE( module->isTraining() );

        module->setTraining( false );
        EXPECT_FALSE( module->isTraining() );
    }
}