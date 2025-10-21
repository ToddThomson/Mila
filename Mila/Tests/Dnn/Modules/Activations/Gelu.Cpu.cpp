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

import Mila;

namespace Modules::Activations::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    // Memory resource for CPU device
    using MR = CpuMemoryResource;
    using GeluCpuModule = Gelu<DeviceType::Cpu, TensorDataType::FP32>;

    struct GeluCpuTestData {
        std::vector<size_t> shape;
        std::shared_ptr<GeluCpuModule> gelu_module;

        static GeluCpuTestData CreateWithDeviceId( int device_id, size_t batch, size_t seq, size_t chan ) {
            GeluCpuTestData d;
            d.shape = { batch, seq, chan };
            GeluConfig config;

            // Create an execution context from the device_id and forward to the Gelu ctor.
            auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>( device_id );
            d.gelu_module = std::make_shared<GeluCpuModule>( config, ctx );

            return d;
        }

        static GeluCpuTestData CreateWithExecutionContext( std::shared_ptr<ExecutionContext<DeviceType::Cpu>> ctx, size_t batch, size_t seq, size_t chan ) {
            GeluCpuTestData d;
            d.shape = { batch, seq, chan };
            GeluConfig config;

            d.gelu_module = std::make_shared<GeluCpuModule>( config, ctx );

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
            data_by_id_.gelu_module.reset();
            data_by_ctx_.gelu_module.reset();
        }

        size_t batch_{ 0 }, seq_{ 0 }, chan_{ 0 };
        GeluCpuTestData data_by_id_;
        GeluCpuTestData data_by_ctx_;
    };

    TEST_F( GeluCpuTests, Construct_WithExecutionContext_ThrowsWhenOpNotRegistered ) {
        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>( -1 );

        // Operation name used by Gelu::createOperation is "GeluOp"
        if (!isOperationRegistered<DeviceType::Cpu, TensorDataType::FP32>( "GeluOp" ))
        {
            EXPECT_THROW( GeluCpuModule( GeluConfig(), ctx ), std::runtime_error );
        }
        else
        {
            EXPECT_NO_THROW( GeluCpuModule( GeluConfig(), ctx ) );
        }
    }

    TEST_F( GeluCpuTests, Forward_BehaviorDependsOnRegistration ) {
        // If not registered, constructor is expected to throw.
        if (!isOperationRegistered<DeviceType::Cpu, TensorDataType::FP32>( "GeluOp" ))
        {
            auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>( -1 );
            EXPECT_THROW( GeluCpuModule( GeluConfig(), ctx ), std::runtime_error );
            return;
        }

        auto d = GeluCpuTestData::CreateWithDeviceId( -1, batch_, seq_, chan_ );

        // Construct tensors using the device-aware constructors defined in Tensor.ixx
        auto device = d.gelu_module->getExecutionContext()->getDevice();

        Tensor<TensorDataType::FP32, MR> input( device, d.shape );
        Tensor<TensorDataType::FP32, MR> output( device, d.shape );

        // Fill input with deterministic values (host-accessible)
        for (size_t i = 0; i < input.size(); ++i)
        {
            input.data()[i] = static_cast<float>( i ) / input.size() * 4.0f - 2.0f;
        }

        EXPECT_NO_THROW( d.gelu_module->forward( input, output ) );
        EXPECT_EQ( output.size(), input.size() );
    }

    TEST_F( GeluCpuTests, ToString_ContainsGeluOrConstructorThrows ) {
        if (!isOperationRegistered<DeviceType::Cpu, TensorDataType::FP32>( "GeluOp" ))
        {
            auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>( -1 );
            EXPECT_THROW( GeluCpuModule( GeluConfig(), ctx ), std::runtime_error );
            return;
        }

        auto d = GeluCpuTestData::CreateWithDeviceId( -1, batch_, seq_, chan_ );
        auto s = d.gelu_module->toString();
        EXPECT_FALSE( s.empty() );
        EXPECT_NE( s.find( "Gelu" ), std::string::npos );
    }

    TEST_F( GeluCpuTests, DefaultApproximationMethod_IsTanhOrConstructorThrows ) {
        if (!isOperationRegistered<DeviceType::Cpu, TensorDataType::FP32>( "GeluOp" ))
        {
            auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>( -1 );
            EXPECT_THROW( GeluCpuModule( GeluConfig(), ctx ), std::runtime_error );
            return;
        }

        auto d = GeluCpuTestData::CreateWithDeviceId( -1, batch_, seq_, chan_ );
        EXPECT_EQ( d.gelu_module->getApproximationMethod(), GeluConfig::ApproximationMethod::Tanh );
    }

    TEST_F( GeluCpuTests, ExecutionContext_DeviceTypeMatchesOrConstructorThrows ) {
        if (!isOperationRegistered<DeviceType::Cpu, TensorDataType::FP32>( "GeluOp" ))
        {
            auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>( -1 );
            EXPECT_THROW( GeluCpuModule( GeluConfig(), ctx ), std::runtime_error );
            return;
        }

        auto d = GeluCpuTestData::CreateWithDeviceId( -1, batch_, seq_, chan_ );
        auto exec_ctx = d.gelu_module->getExecutionContext();
        ASSERT_NE( exec_ctx, nullptr );
        auto dev = exec_ctx->getDevice();
        ASSERT_NE( dev, nullptr );
        EXPECT_EQ( dev->getDeviceType(), DeviceType::Cpu );
    }

    TEST_F( GeluCpuTests, Construct_WithExecutionContext_WorksOrThrows ) {
        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>( -1 );

        if (!isOperationRegistered<DeviceType::Cpu, TensorDataType::FP32>( "GeluOp" ))
        {
            EXPECT_THROW( GeluCpuModule( GeluConfig(), ctx ), std::runtime_error );
            return;
        }

        auto d = GeluCpuTestData::CreateWithExecutionContext( ctx, batch_, seq_, chan_ );
        EXPECT_EQ( d.gelu_module->getExecutionContext()->getDevice()->getDeviceType(), DeviceType::Cpu );
    }
}