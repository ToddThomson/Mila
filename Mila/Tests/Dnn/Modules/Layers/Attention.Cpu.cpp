#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdint>

import Mila;

namespace Modules::Layers::Tests
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
            exec_ctx_ = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        }

        std::shared_ptr<ExecutionContext<DeviceType::Cpu>> exec_ctx_;
    };

    TEST_F( AttentionCpuTests, Constructor_NullContext_Throws )
    {
        AttentionConfig cfg( 64, 8 );
        cfg.withName( "attn_null" );

        std::shared_ptr<ExecutionContext<DeviceType::Cpu>> null_ctx;
        EXPECT_THROW(
            (std::make_shared<Attention<DeviceType::Cpu, TensorDataType::FP32>>( null_ctx, cfg )),
            std::invalid_argument );
    }

    TEST_F( AttentionCpuTests, ParameterCount_IsZero )
    {
        AttentionConfig cfg( 64, 8 );
        cfg.withName( "attn_params" );

        auto module = std::make_shared<Attention<DeviceType::Cpu, TensorDataType::FP32>>( exec_ctx_, cfg );
        EXPECT_EQ( module->parameterCount(), 0u );
    }

    TEST_F( AttentionCpuTests, BuildAndForward )
    {
        const int64_t B = 2;
        const int64_t T = 4;
        const int64_t C = 64; // embedding dim
        const int64_t heads = 8;
        const int64_t hs = C / heads;

        AttentionConfig cfg( C, heads );
        cfg.withName( "attn_forward" );

        auto module = std::make_shared<Attention<DeviceType::Cpu, TensorDataType::FP32>>( exec_ctx_, cfg );

        // Head-major shapes: [B, NH, T, hs]
        shape_t input_shape = { B, heads, T, hs };
        shape_t output_shape = { B, heads, T, hs };

        // Build
        EXPECT_NO_THROW( module->build( input_shape ) );
        EXPECT_TRUE( module->isBuilt() );

        // Prepare Q/K/V and output tensors in head-major layout
        CpuTensor<TensorDataType::FP32> Q( exec_ctx_->getDevice(), input_shape );
        CpuTensor<TensorDataType::FP32> K( exec_ctx_->getDevice(), input_shape );
        CpuTensor<TensorDataType::FP32> V( exec_ctx_->getDevice(), input_shape );
        CpuTensor<TensorDataType::FP32> output( exec_ctx_->getDevice(), output_shape );

        // Fill deterministic values on CPU
        for (size_t i = 0; i < Q.size(); ++i)
        {
            Q.data()[i] = static_cast<float>( (i % 100) ) * 0.01f;
            K.data()[i] = static_cast<float>( (i % 97) ) * 0.02f;
            V.data()[i] = static_cast<float>( (i % 89) ) * 0.03f;
        }

        // Forward should succeed with head-major inputs
        EXPECT_NO_THROW( module->forward( Q, K, V, output ) );

        // Output shape / size checks
        EXPECT_EQ( output.shape(), output_shape );
        EXPECT_EQ( output.size(), static_cast<size_t>(B * heads * T * hs) );
    }

    TEST_F( AttentionCpuTests, ToStringContainsInfo )
    {
        AttentionConfig cfg( 32, 4 );
        cfg.withName( "attn_info" );

        auto module = std::make_shared<Attention<DeviceType::Cpu, TensorDataType::FP32>>( exec_ctx_, cfg );

        // Build with head-major shape: embedding 32 with 4 heads -> hs = 8
        module->build( shape_t{ 1, 4, 1, 8 } );

        auto s = module->toString();
        EXPECT_NE( s.find( "Attention" ), std::string::npos );
        EXPECT_NE( s.find( "Embedding dimension" ), std::string::npos );
        EXPECT_NE( s.find( "Number of heads" ), std::string::npos );
    }

    TEST_F( AttentionCpuTests, TrainingModeToggle )
    {
        AttentionConfig cfg( 64, 8 );
        cfg.withName( "attn_train_toggle" );

        auto module = std::make_shared<Attention<DeviceType::Cpu, TensorDataType::FP32>>( exec_ctx_, cfg );

        EXPECT_FALSE( module->isTraining() );

        module->setTraining( true );
        EXPECT_TRUE( module->isTraining() );

        module->setTraining( false );
        EXPECT_FALSE( module->isTraining() );
    }
}