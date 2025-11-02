/**
 * @file Gelu.Cuda.cpp
 * @brief Unit tests for GELU activation module on CUDA device.
 *
 * Tests assert behavior for both registered and unregistered backend operation:
 * - If the backend operation is registered the module should construct and operate.
 * - If the backend operation is not registered the Gelu constructor must throw.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>

import Mila;

namespace Modules::Activations::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    // Memory resource for CUDA device
    using MR = CudaDeviceMemoryResource;

    // Gelu module aliases with explicit precision to match Gelu.ixx
    using GeluCudaModule = Gelu<DeviceType::Cuda, TensorDataType::FP32>;
    using GeluCpuModule = Gelu<DeviceType::Cpu, TensorDataType::FP32>;

    struct GeluCudaTestData {
        std::vector<int64_t> shape;
        std::shared_ptr<GeluCudaModule> gelu_module;

        static GeluCudaTestData CreateWithExecutionContext( std::shared_ptr<ExecutionContext<DeviceType::Cuda>> ctx, int64_t batch, int64_t seq, int64_t chan ) {
            GeluCudaTestData d;
            d.shape = { batch, seq, chan };
            GeluConfig config;
            d.gelu_module = std::make_shared<GeluCudaModule>( ctx, config );
            return d;
        }
    };

    class GeluCudaTests : public ::testing::Test {
    protected:
        void SetUp() override {
            batch_ = 2;
            seq_ = 4;
            chan_ = 8;
        }

        void TearDown() override {
            data_by_ctx_.gelu_module.reset();
        }

        size_t batch_{ 0 }, seq_{ 0 }, chan_{ 0 };
        GeluCudaTestData data_by_ctx_;
    };

    TEST_F( GeluCudaTests, Construct_WithDeviceId_BehaviorDependsOnRegistration ) {
        // If CUDA GELU op is registered construction should succeed; otherwise it must throw.
        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

        if (isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" ))
        {
            EXPECT_NO_THROW( (GeluCudaModule( ctx, GeluConfig() )) );
        }
        else
        {
            EXPECT_THROW( (GeluCudaModule( ctx, GeluConfig() )), std::runtime_error );
        }
    }

    TEST_F( GeluCudaTests, Construct_WithExecutionContext_BehaviorDependsOnRegistration ) {
        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

        if (isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" ))
        {
            EXPECT_NO_THROW( (GeluCudaModule( ctx, GeluConfig() )) );
        }
        else
        {
            EXPECT_THROW( (GeluCudaModule( ctx, GeluConfig() )), std::runtime_error );
        }
    }

    TEST_F( GeluCudaTests, Forward_BehaviorDependsOnRegistration ) {
        if (!isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" ))
        {
            // Expect constructor to fail when op is not registered.
            auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
            EXPECT_THROW( (GeluCudaModule( ctx, GeluConfig() )), std::runtime_error );
            return;
        }

        // Create an execution context up-front and pass it to the factory
        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
        auto d = GeluCudaTestData::CreateWithExecutionContext( ctx, batch_, seq_, chan_ );

        // Use the same execution context (ctx) to obtain device for tensor construction
        auto cuda_device = ctx->getDevice();

        Tensor<TensorDataType::FP32, MR> input( cuda_device, d.shape );
        Tensor<TensorDataType::FP32, MR> output( cuda_device, d.shape );

        // Initialize on host then copy to device using a host tensor constructed with a CPU device
        auto cpu_ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>( -1 );
        auto cpu_device = cpu_ctx->getDevice();
        Tensor<TensorDataType::FP32, CpuMemoryResource> host_input( cpu_device, d.shape );

        for (size_t i = 0; i < host_input.size(); ++i)
            host_input.data()[i] = static_cast<float>( i ) / host_input.size() * 4.0f - 2.0f;

        copy( host_input, input );

        ASSERT_NO_THROW( d.gelu_module->forward( input, output ) );
        EXPECT_EQ( output.size(), input.size() );
    }

    TEST_F( GeluCudaTests, ToString_ContainsGeluOrConstructorThrows ) {
        if (!isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" ))
        {
            auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
            EXPECT_THROW( (GeluCudaModule( ctx, GeluConfig() )), std::runtime_error );
            return;
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
        auto d = GeluCudaTestData::CreateWithExecutionContext( ctx, batch_, seq_, chan_ );
        auto s = d.gelu_module->toString();
        
        EXPECT_FALSE( s.empty() );
        EXPECT_NE( s.find( "Gelu" ), std::string::npos );
    }

    TEST_F( GeluCudaTests, Construct_WithExecutionContext_WorksOrThrows ) {
        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

        if (!isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" ))
        {
            EXPECT_THROW( (GeluCudaModule( ctx, GeluConfig() )), std::runtime_error );
            return;
        }

        auto d = GeluCudaTestData::CreateWithExecutionContext( ctx, batch_, seq_, chan_ );
        EXPECT_EQ( ctx->getDevice()->getDeviceType(), DeviceType::Cuda );
    }

    TEST_F( GeluCudaTests, CpuCuda_Equivalence_OrConstructorThrows ) {
        bool cpu_registered = isOperationRegistered<DeviceType::Cpu, TensorDataType::FP32>( "GeluOp" );
        bool cuda_registered = isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" );

        if (!cpu_registered || !cuda_registered)
        {
            // If either backend is missing, at least one constructor must throw.
            if (!cpu_registered)
            {
                auto cpu_ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>( -1 );
                EXPECT_THROW( (GeluCpuModule( cpu_ctx, GeluConfig() )), std::runtime_error );
            }
            if (!cuda_registered)
            {
                auto cuda_ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
                EXPECT_THROW( (GeluCudaModule( cuda_ctx, GeluConfig() )), std::runtime_error );
            }

            return;
        }

        // Both registered: validate numerical equivalence.
        std::vector<int64_t> shape = { 2, 2, 4 };
        GeluConfig config;
        auto cpu_ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto cuda_ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
        
        auto cpu_gelu = std::make_shared<GeluCpuModule>( cpu_ctx, config );
        auto cuda_gelu = std::make_shared<GeluCudaModule>( cuda_ctx, config );

        // Construct CPU tensors using the cpu_ctx device
        auto cpu_device = cpu_ctx->getDevice();
        Tensor<TensorDataType::FP32, CpuMemoryResource> cpu_input( cpu_device, shape );
        Tensor<TensorDataType::FP32, CpuMemoryResource> cpu_output( cpu_device, shape );

        for (size_t i = 0; i < cpu_input.size(); ++i)
            cpu_input.data()[i] = static_cast<float>( i ) / cpu_input.size() * 4.0f - 2.0f;

		cpu_gelu->build( cpu_input.shape() );
        cpu_gelu->forward( cpu_input, cpu_output );

        // Move to device and run CUDA module using cuda_ctx device
        Tensor<TensorDataType::FP32, MR> cuda_input( cuda_ctx->getDevice(), shape );

        copy( cpu_input, cuda_input );

        Tensor<TensorDataType::FP32, MR> cuda_output( cuda_ctx->getDevice(), shape );

		cuda_gelu->build( cuda_input.shape() );
        cuda_gelu->forward( cuda_input, cuda_output );

        auto cuda_output_host = toHost<TensorDataType::FP32>( cuda_output );

        const float epsilon = 1e-3f;

        for (size_t i = 0; i < cpu_output.size(); ++i)
        {
            float cpu_val = cpu_output.data()[i];
            float gpu_val = cuda_output_host.data()[i];
            float diff = std::abs( cpu_val - gpu_val );
            EXPECT_LT( diff, epsilon ) << "Mismatch at index " << i
                << ": CPU=" << cpu_val
                << ", CUDA=" << gpu_val;
        }
    }
}