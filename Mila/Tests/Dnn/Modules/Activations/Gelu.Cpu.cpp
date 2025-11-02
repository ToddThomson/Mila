/**
 * @file Gelu.Cpu.cpp
 * @brief Unit tests for GELU activation module on CPU device.
 *
 * Verifies basic API, forward/backward invocation, config, and execution-context
 * constructor behavior for the CPU-specialized Gelu module.
 *
 * Tests assert behavior for both registered and unregistered backend operation:
 * - If the backend operation is registered the module should construct and operate.
 * - If the backend operation is not registered the Gelu constructor must throw.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <stdexcept>

import Mila;

namespace Modules::Activations::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    using MR = CpuMemoryResource;
    using GeluCpuModule = Gelu<DeviceType::Cpu, dtype_t::FP32>;

    struct GeluCpuTestData {
        std::vector<int64_t> shape;
        std::shared_ptr<GeluCpuModule> gelu_module;

        static GeluCpuTestData CreateWithExecutionContext( std::shared_ptr<ExecutionContext<DeviceType::Cpu>> ctx, int64_t batch, int64_t seq, int64_t chan ) {
            GeluCpuTestData d;
            d.shape = { batch, seq, chan };
            GeluConfig config;

            d.gelu_module = std::make_shared<GeluCpuModule>( ctx, config );

            return d;
        }
    };

    class GeluCpuTests : public ::testing::Test {
    protected:
        void SetUp() override {
            batch_ = 2;
            seq_ = 4;
            chan_ = 8;
        }

        void TearDown() override {
            data_by_ctx_.gelu_module.reset();
        }

        dim_t batch_{ 0 }, seq_{ 0 }, chan_{ 0 };
        GeluCpuTestData data_by_ctx_;
    };

    TEST_F( GeluCpuTests, Construct_WithExecutionContext_ThrowsWhenOpNotRegistered ) {
        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();

        // Operation name used by Gelu::createOperation is "GeluOp"
        if (!isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ))
        {
            EXPECT_THROW( GeluCpuModule( ctx, GeluConfig() ), std::runtime_error );
        }
        else
        {
            EXPECT_NO_THROW( GeluCpuModule( ctx, GeluConfig() ) );
        }
    }

    TEST_F( GeluCpuTests, Forward_BehaviorDependsOnRegistration ) {
        // If not registered, constructor is expected to throw.
        if (!isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ))
        {
            auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
            EXPECT_THROW( GeluCpuModule( ctx, GeluConfig() ), std::runtime_error );
            return;
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto d = GeluCpuTestData::CreateWithExecutionContext( ctx, batch_, seq_, chan_ );

        auto device = ctx->getDevice();

        Tensor<dtype_t::FP32, MR> input( device, d.shape );
        Tensor<dtype_t::FP32, MR> output( device, d.shape );

        for (size_t i = 0; i < input.size(); ++i)
        {
            input.data()[i] = static_cast<float>( i ) / input.size() * 4.0f - 2.0f;
        }

		// All modules must be built before use
		d.gelu_module->build( d.shape );

        EXPECT_NO_THROW( d.gelu_module->forward( input, output ) );
        EXPECT_EQ( output.size(), input.size() );
    }

    TEST_F( GeluCpuTests, ToString_ContainsGeluOrConstructorThrows ) {
        if (!isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ))
        {
            auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
            
            EXPECT_THROW( GeluCpuModule( ctx, GeluConfig() ), std::runtime_error );
            
            return;
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto d = GeluCpuTestData::CreateWithExecutionContext( ctx, batch_, seq_, chan_ );
        
        auto s = d.gelu_module->toString();
        
        EXPECT_FALSE( s.empty() );
        EXPECT_NE( s.find( "Gelu" ), std::string::npos );
    }

    TEST_F( GeluCpuTests, DefaultApproximationMethod_IsTanhOrConstructorThrows ) {
        if (!isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ))
        {
            auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
            EXPECT_THROW( GeluCpuModule( ctx, GeluConfig() ), std::runtime_error );
            return;
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto d = GeluCpuTestData::CreateWithExecutionContext( ctx, batch_, seq_, chan_ );
        EXPECT_EQ( d.gelu_module->getApproximationMethod(), GeluConfig::ApproximationMethod::Tanh );
    }

    TEST_F( GeluCpuTests, ExecutionContext_DeviceTypeMatchesOrConstructorThrows ) {
        if (!isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ))
        {
            auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
            EXPECT_THROW( GeluCpuModule( ctx, GeluConfig() ), std::runtime_error );
            return;
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto d = GeluCpuTestData::CreateWithExecutionContext( ctx, batch_, seq_, chan_ );
        
        auto dev = ctx->getDevice();
        
        ASSERT_NE( dev, nullptr );
        EXPECT_EQ( dev->getDeviceType(), DeviceType::Cpu );
    }

    TEST_F( GeluCpuTests, Construct_WithExecutionContext_WorksOrThrows ) {
        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();

        if (!isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ))
        {
            EXPECT_THROW( GeluCpuModule( ctx, GeluConfig() ), std::runtime_error );
            return;
        }

        auto d = GeluCpuTestData::CreateWithExecutionContext( ctx, batch_, seq_, chan_ );
        EXPECT_EQ( ctx->getDevice()->getDeviceType(), DeviceType::Cpu );
    }

    TEST_F( GeluCpuTests, Constructor_NullExecutionContext_ThrowsInvalidArgument )
    {
        std::shared_ptr<ExecutionContext<DeviceType::Cpu>> null_ctx = nullptr;
        EXPECT_THROW( GeluCpuModule( null_ctx, GeluConfig() ), std::invalid_argument );
    }

    TEST_F( GeluCpuTests, Constructor_InvalidConfig_ThrowsInvalidArgument )
    {
        // Validation happens before backend creation so this should throw regardless of registry state
        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        GeluConfig bad_cfg = GeluConfig().withApproximationMethod( GeluConfig::ApproximationMethod::Exact );

        EXPECT_THROW( GeluCpuModule( ctx, bad_cfg ), std::invalid_argument );
    }

    TEST_F( GeluCpuTests, Backward_IsNoOp )
    {
        if (!isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ))
        {
            GTEST_SKIP() << "GeluOp not registered; skipping backward no-op check.";
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto d = GeluCpuTestData::CreateWithExecutionContext( ctx, batch_, seq_, chan_ );

        Tensor<dtype_t::FP32, MR> input( ctx->getDevice(), d.shape );
        Tensor<dtype_t::FP32, MR> output_grad( ctx->getDevice(), d.shape );
        Tensor<dtype_t::FP32, MR> input_grad( ctx->getDevice(), d.shape );

        // Initialize input and gradients
        for (size_t i = 0; i < input.size(); ++i)
        {
            input.data()[i] = 0.1f * static_cast<float>( i );
            output_grad.data()[i] = 1.0f;
            input_grad.data()[i] = 0.0f;
        }

        // Build then invoke backward — current Gelu::backward is a no-op
        d.gelu_module->build( d.shape );

        EXPECT_NO_THROW( d.gelu_module->backward( input, output_grad, input_grad ) );

        // Since Gelu::backward does not call the backend, input_grad must remain unchanged (all zeros)
        for (size_t i = 0; i < input_grad.size(); ++i)
        {
            EXPECT_EQ( input_grad.data()[i], 0.0f );
        }
    }

    TEST_F( GeluCpuTests, Synchronize_ParameterCount_SetTraining )
    {
        if (!isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ))
        {
            GTEST_SKIP() << "GeluOp not registered; skipping sync/parameter/training checks.";
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto d = GeluCpuTestData::CreateWithExecutionContext( ctx, batch_, seq_, chan_ );

        // synchronize should forward to exec_context_->synchronize and not throw
        EXPECT_NO_THROW( d.gelu_module->synchronize() );

        // module has no parameters
        EXPECT_EQ( d.gelu_module->parameterCount(), 0u );

        // training flag toggles
        d.gelu_module->setTraining( true );
        EXPECT_TRUE( d.gelu_module->isTraining() );

        d.gelu_module->setTraining( false );
        EXPECT_FALSE( d.gelu_module->isTraining() );
    }
}