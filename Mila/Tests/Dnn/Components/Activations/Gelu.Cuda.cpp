/**
 * @file Gelu.Cuda.cpp
 * @brief Unit tests for GELU activation module on CUDA device.
 *
 * Tests assert behavior for both registered and unregistered backend operation:
 * - If the backend operation is registered the module should construct and operate.
 * - If the backend operation is not registered the Gelu constructor must throw.
 * - Forward and backward passes are validated for correctness.
 * - CPU-CUDA equivalence is verified for both forward and backward passes.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstdlib>

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

    struct GeluCudaTestData
    {
        std::vector<int64_t> shape;
        std::shared_ptr<GeluCudaModule> gelu_module;

        static GeluCudaTestData CreateWithExecutionContext(
            std::shared_ptr<ExecutionContext<DeviceType::Cuda>> ctx,
            int64_t batch, int64_t seq, int64_t chan )
        {
            GeluCudaTestData d;
            d.shape = { batch, seq, chan };
            GeluConfig config;
            d.gelu_module = std::make_shared<GeluCudaModule>( ctx, config );
            return d;
        }
    };

    class GeluCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            batch_ = 2;
            seq_ = 4;
            chan_ = 8;
        }

        void TearDown() override
        {
            data_by_ctx_.gelu_module.reset();
        }

        // Helper: Compute GELU(x) = x * ?(x) where ? is the cumulative distribution function of the standard Gaussian
        // Using tanh approximation: GELU(x) ? 0.5 * x * (1 + tanh(?(2/?) * (x + 0.044715 * x^3)))
        float geluReference( float x )
        {
            constexpr float sqrt_2_over_pi = 0.7978845608f;  // sqrt(2/?)
            constexpr float coeff = 0.044715f;
            float x_cubed = x * x * x;
            float tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);
            return 0.5f * x * (1.0f + std::tanh( tanh_arg ));
        }

        // Helper: Compute GELU gradient
        // d/dx GELU(x) ? ?(x) + x * ?(x) where ? is the Gaussian PDF
        float geluGradientReference( float x )
        {
            constexpr float sqrt_2_over_pi = 0.7978845608f;
            constexpr float coeff = 0.044715f;

            float x_squared = x * x;
            float x_cubed = x * x_squared;

            float tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);
            float tanh_val = std::tanh( tanh_arg );
            float sech_squared = 1.0f - tanh_val * tanh_val;  // sech²(x) = 1 - tanh²(x)

            float d_tanh_arg = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x_squared);

            return 0.5f * (1.0f + tanh_val) + 0.5f * x * sech_squared * d_tanh_arg;
        }

        size_t batch_{ 0 }, seq_{ 0 }, chan_{ 0 };
        GeluCudaTestData data_by_ctx_;
    };

    // ========================================================================
    // Construction Tests
    // ========================================================================

    TEST_F( GeluCudaTests, Construct_WithDeviceId_BehaviorDependsOnRegistration )
    {
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

    TEST_F( GeluCudaTests, Construct_WithExecutionContext_BehaviorDependsOnRegistration )
    {
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

    // ========================================================================
    // Forward Pass Tests
    // ========================================================================

    TEST_F( GeluCudaTests, Forward_BehaviorDependsOnRegistration )
    {
        if (!isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" ))
        {
            auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
            EXPECT_THROW( (GeluCudaModule( ctx, GeluConfig() )), std::runtime_error );
            return;
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
        auto d = GeluCudaTestData::CreateWithExecutionContext( ctx, batch_, seq_, chan_ );

        auto cuda_device = ctx->getDevice();

        Tensor<TensorDataType::FP32, MR> input( cuda_device, d.shape );
        Tensor<TensorDataType::FP32, MR> output( cuda_device, d.shape );

        // Initialize input
        auto cpu_ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto cpu_device = cpu_ctx->getDevice();
        Tensor<TensorDataType::FP32, CpuMemoryResource> host_input( cpu_device, d.shape );

        for (size_t i = 0; i < host_input.size(); ++i)
            host_input.data()[i] = static_cast<float>( i ) / host_input.size() * 4.0f - 2.0f;

        copy( host_input, input );

        d.gelu_module->build( d.shape );

        ASSERT_NO_THROW( d.gelu_module->forward( input, output ) );
        EXPECT_EQ( output.size(), input.size() );
    }

    TEST_F( GeluCudaTests, Forward_OutputMatchesReference )
    {
        if (!isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" ))
        {
            GTEST_SKIP() << "GeluOp not registered for CUDA FP32";
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
        auto gelu = std::make_shared<GeluCudaModule>( ctx, GeluConfig() );

        std::vector<int64_t> shape = { 2, 3, 4 };
        auto cuda_device = ctx->getDevice();

        Tensor<TensorDataType::FP32, MR> input( cuda_device, shape );
        Tensor<TensorDataType::FP32, MR> output( cuda_device, shape );

        // Initialize with known values
        auto cpu_ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto cpu_device = cpu_ctx->getDevice();
        Tensor<TensorDataType::FP32, CpuMemoryResource> host_input( cpu_device, shape );

        for (size_t i = 0; i < host_input.size(); ++i)
            host_input.data()[i] = static_cast<float>( i ) / host_input.size() * 4.0f - 2.0f;

        copy( host_input, input );

        gelu->build( shape );
        gelu->forward( input, output );

        // Copy output back to host
        auto host_output = toHost<TensorDataType::FP32>( output );

        // Verify against reference implementation
        const float tolerance = 1e-4f;
        for (size_t i = 0; i < host_input.size(); ++i)
        {
            float input_val = host_input.data()[i];
            float expected = geluReference( input_val );
            float actual = host_output.data()[i];
            float diff = std::abs( expected - actual );

            EXPECT_LT( diff, tolerance )
                << "Forward mismatch at index " << i
                << ": input=" << input_val
                << ", expected=" << expected
                << ", actual=" << actual;
        }
    }

    // ========================================================================
    // Backward Pass Tests
    // ========================================================================

    TEST_F( GeluCudaTests, Backward_ExecutesWithoutError )
    {
        if (!isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" ))
        {
            GTEST_SKIP() << "GeluOp not registered for CUDA FP32";
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
        auto gelu = std::make_shared<GeluCudaModule>( ctx, GeluConfig() );

        std::vector<int64_t> shape = { 2, 4, 8 };
        auto device = ctx->getDevice();

        Tensor<TensorDataType::FP32, MR> input( device, shape );
        Tensor<TensorDataType::FP32, MR> output( device, shape );
        Tensor<TensorDataType::FP32, MR> output_grad( device, shape );
        Tensor<TensorDataType::FP32, MR> input_grad( device, shape );

        // Initialize tensors
        auto cpu_ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto cpu_device = cpu_ctx->getDevice();

        Tensor<TensorDataType::FP32, CpuMemoryResource> host_input( cpu_device, shape );
        Tensor<TensorDataType::FP32, CpuMemoryResource> host_output_grad( cpu_device, shape );

        for (size_t i = 0; i < host_input.size(); ++i)
        {
            host_input.data()[i] = static_cast<float>( i ) / host_input.size() * 4.0f - 2.0f;
            host_output_grad.data()[i] = 1.0f;  // Uniform gradient
        }

        copy( host_input, input );
        copy( host_output_grad, output_grad );

        gelu->setTraining( true );
        gelu->build( shape );
        gelu->forward( input, output );

        EXPECT_NO_THROW( gelu->backward( input, output_grad, input_grad ) );
    }

    TEST_F( GeluCudaTests, Backward_ProducesCorrectShape )
    {
        if (!isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" ))
        {
            GTEST_SKIP() << "GeluOp not registered for CUDA FP32";
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
        auto gelu = std::make_shared<GeluCudaModule>( ctx, GeluConfig() );

        std::vector<int64_t> shape = { 3, 5, 7 };
        auto device = ctx->getDevice();

        Tensor<TensorDataType::FP32, MR> input( device, shape );
        Tensor<TensorDataType::FP32, MR> output( device, shape );
        Tensor<TensorDataType::FP32, MR> output_grad( device, shape );
        Tensor<TensorDataType::FP32, MR> input_grad( device, shape );

        gelu->setTraining( true );
        gelu->build( shape );
        gelu->forward( input, output );
        gelu->backward( input, output_grad, input_grad );

        EXPECT_EQ( input_grad.shape(), input.shape() );
        EXPECT_EQ( input_grad.size(), input.size() );
    }

    TEST_F( GeluCudaTests, Backward_GradientsMatchReference )
    {
        if (!isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" ))
        {
            GTEST_SKIP() << "GeluOp not registered for CUDA FP32";
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
        auto gelu = std::make_shared<GeluCudaModule>( ctx, GeluConfig() );

        std::vector<int64_t> shape = { 2, 3, 4 };
        auto device = ctx->getDevice();

        Tensor<TensorDataType::FP32, MR> input( device, shape );
        Tensor<TensorDataType::FP32, MR> output( device, shape );
        Tensor<TensorDataType::FP32, MR> output_grad( device, shape );
        Tensor<TensorDataType::FP32, MR> input_grad( device, shape );

        // Initialize on host
        auto cpu_ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto cpu_device = cpu_ctx->getDevice();

        Tensor<TensorDataType::FP32, CpuMemoryResource> host_input( cpu_device, shape );
        Tensor<TensorDataType::FP32, CpuMemoryResource> host_output_grad( cpu_device, shape );
        Tensor<TensorDataType::FP32, CpuMemoryResource> host_input_grad( cpu_device, shape );

        for (size_t i = 0; i < host_input.size(); ++i)
        {
            host_input.data()[i] = static_cast<float>( i ) / host_input.size() * 4.0f - 2.0f;
            host_output_grad.data()[i] = 1.0f;
            host_input_grad.data()[i] = 0.0f;
        }

        copy( host_input, input );
        copy( host_output_grad, output_grad );
        copy( host_input_grad, input_grad );

        gelu->setTraining( true );
        gelu->build( shape );
        gelu->forward( input, output );
        gelu->backward( input, output_grad, input_grad );

        // Copy result back to host
        auto host_input_grad_result = toHost<TensorDataType::FP32>( input_grad );

        // Verify against reference gradient computation
        const float tolerance = 1e-3f;
        for (size_t i = 0; i < host_input.size(); ++i)
        {
            float x = host_input.data()[i];
            float grad_out = host_output_grad.data()[i];
            float expected = geluGradientReference( x ) * grad_out;
            float actual = host_input_grad_result.data()[i];
            float diff = std::abs( expected - actual );

            EXPECT_LT( diff, tolerance )
                << "Backward gradient mismatch at index " << i
                << ": input=" << x
                << ", expected=" << expected
                << ", actual=" << actual;
        }
    }

    TEST_F( GeluCudaTests, Backward_ChainRuleWithNonUniformGradients )
    {
        if (!isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" ))
        {
            GTEST_SKIP() << "GeluOp not registered for CUDA FP32";
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
        auto gelu = std::make_shared<GeluCudaModule>( ctx, GeluConfig() );

        std::vector<int64_t> shape = { 2, 3, 4 };
        auto device = ctx->getDevice();

        Tensor<TensorDataType::FP32, MR> input( device, shape );
        Tensor<TensorDataType::FP32, MR> output( device, shape );
        Tensor<TensorDataType::FP32, MR> output_grad( device, shape );
        Tensor<TensorDataType::FP32, MR> input_grad( device, shape );

        auto cpu_ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto cpu_device = cpu_ctx->getDevice();

        Tensor<TensorDataType::FP32, CpuMemoryResource> host_input( cpu_device, shape );
        Tensor<TensorDataType::FP32, CpuMemoryResource> host_output_grad( cpu_device, shape );

        // Non-uniform gradients
        for (size_t i = 0; i < host_input.size(); ++i)
        {
            host_input.data()[i] = static_cast<float>( i ) / host_input.size() * 4.0f - 2.0f;
            host_output_grad.data()[i] = static_cast<float>( i + 1 ) * 0.1f;
        }

        copy( host_input, input );
        copy( host_output_grad, output_grad );

        gelu->setTraining( true );
        gelu->build( shape );
        gelu->forward( input, output );
        gelu->backward( input, output_grad, input_grad );

        auto host_input_grad = toHost<TensorDataType::FP32>( input_grad );

        const float tolerance = 1e-3f;
        for (size_t i = 0; i < host_input.size(); ++i)
        {
            float x = host_input.data()[i];
            float grad_out = host_output_grad.data()[i];
            float expected = geluGradientReference( x ) * grad_out;
            float actual = host_input_grad.data()[i];
            float diff = std::abs( expected - actual );

            EXPECT_LT( diff, tolerance )
                << "Chain rule gradient mismatch at index " << i;
        }
    }

    TEST_F( GeluCudaTests, Backward_HandlesZeroOutputGradient )
    {
        if (!isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" ))
        {
            GTEST_SKIP() << "GeluOp not registered for CUDA FP32";
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
        auto gelu = std::make_shared<GeluCudaModule>( ctx, GeluConfig() );

        std::vector<int64_t> shape = { 2, 3, 4 };
        auto device = ctx->getDevice();

        Tensor<TensorDataType::FP32, MR> input( device, shape );
        Tensor<TensorDataType::FP32, MR> output( device, shape );
        Tensor<TensorDataType::FP32, MR> output_grad( device, shape );
        Tensor<TensorDataType::FP32, MR> input_grad( device, shape );

        auto cpu_ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto cpu_device = cpu_ctx->getDevice();

        Tensor<TensorDataType::FP32, CpuMemoryResource> host_input( cpu_device, shape );
        Tensor<TensorDataType::FP32, CpuMemoryResource> host_output_grad( cpu_device, shape );

        for (size_t i = 0; i < host_input.size(); ++i)
        {
            host_input.data()[i] = static_cast<float>( i ) / host_input.size() * 2.0f;
            host_output_grad.data()[i] = 0.0f;  // Zero gradient
        }

        copy( host_input, input );
        copy( host_output_grad, output_grad );

        gelu->setTraining( true );
        gelu->build( shape );
        gelu->forward( input, output );
        gelu->backward( input, output_grad, input_grad );

        auto host_input_grad = toHost<TensorDataType::FP32>( input_grad );

        // All gradients should be zero
        for (size_t i = 0; i < host_input_grad.size(); ++i)
        {
            EXPECT_FLOAT_EQ( host_input_grad.data()[i], 0.0f )
                << "Expected zero gradient at index " << i;
        }
    }

    TEST_F( GeluCudaTests, Backward_HandlesEdgeCaseInputs )
    {
        if (!isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" ))
        {
            GTEST_SKIP() << "GeluOp not registered for CUDA FP32";
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
        auto gelu = std::make_shared<GeluCudaModule>( ctx, GeluConfig() );

        std::vector<int64_t> shape = { 1, 8 };
        auto device = ctx->getDevice();

        Tensor<TensorDataType::FP32, MR> input( device, shape );
        Tensor<TensorDataType::FP32, MR> output( device, shape );
        Tensor<TensorDataType::FP32, MR> output_grad( device, shape );
        Tensor<TensorDataType::FP32, MR> input_grad( device, shape );

        auto cpu_ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto cpu_device = cpu_ctx->getDevice();

        Tensor<TensorDataType::FP32, CpuMemoryResource> host_input( cpu_device, shape );
        Tensor<TensorDataType::FP32, CpuMemoryResource> host_output_grad( cpu_device, shape );

        // Edge cases: large positive, large negative, zero, small values
        std::vector<float> test_values = { -10.0f, -1.0f, -0.1f, 0.0f, 0.1f, 1.0f, 10.0f, 100.0f };

        for (size_t i = 0; i < test_values.size(); ++i)
        {
            host_input.data()[i] = test_values[i];
            host_output_grad.data()[i] = 1.0f;
        }

        copy( host_input, input );
        copy( host_output_grad, output_grad );

        gelu->setTraining( true );
        gelu->build( shape );

        EXPECT_NO_THROW( gelu->forward( input, output ) );
        EXPECT_NO_THROW( gelu->backward( input, output_grad, input_grad ) );

        auto host_input_grad = toHost<TensorDataType::FP32>( input_grad );

        // Verify no NaN or Inf in gradients
        for (size_t i = 0; i < host_input_grad.size(); ++i)
        {
            EXPECT_FALSE( std::isnan( host_input_grad.data()[i] ) )
                << "NaN gradient at index " << i << " for input " << test_values[i];
            EXPECT_FALSE( std::isinf( host_input_grad.data()[i] ) )
                << "Inf gradient at index " << i << " for input " << test_values[i];
        }
    }

    // ========================================================================
    // CPU-CUDA Equivalence Tests
    // ========================================================================

    TEST_F( GeluCudaTests, ForwardBackward_CpuCudaEquivalence )
    {
        bool cpu_registered = isOperationRegistered<DeviceType::Cpu, TensorDataType::FP32>( "GeluOp" );
        bool cuda_registered = isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" );

        if (!cpu_registered || !cuda_registered)
        {
            GTEST_SKIP() << "Both CPU and CUDA GeluOp required for equivalence test";
        }

        std::vector<int64_t> shape = { 2, 3, 4 };
        GeluConfig config;

        auto cpu_ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto cuda_ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

        auto cpu_gelu = std::make_shared<GeluCpuModule>( cpu_ctx, config );
        auto cuda_gelu = std::make_shared<GeluCudaModule>( cuda_ctx, config );

        // Prepare CPU tensors
        auto cpu_device = cpu_ctx->getDevice();
        Tensor<TensorDataType::FP32, CpuMemoryResource> cpu_input( cpu_device, shape );
        Tensor<TensorDataType::FP32, CpuMemoryResource> cpu_output( cpu_device, shape );
        Tensor<TensorDataType::FP32, CpuMemoryResource> cpu_output_grad( cpu_device, shape );
        Tensor<TensorDataType::FP32, CpuMemoryResource> cpu_input_grad( cpu_device, shape );

        for (size_t i = 0; i < cpu_input.size(); ++i)
        {
            cpu_input.data()[i] = static_cast<float>( i ) / cpu_input.size() * 4.0f - 2.0f;
            cpu_output_grad.data()[i] = static_cast<float>( i + 1 ) * 0.1f;
        }

        cpu_gelu->setTraining( true );
        cpu_gelu->build( shape );
        cpu_gelu->forward( cpu_input, cpu_output );
        cpu_gelu->backward( cpu_input, cpu_output_grad, cpu_input_grad );

        // Run on CUDA
        Tensor<TensorDataType::FP32, MR> cuda_input( cuda_ctx->getDevice(), shape );
        Tensor<TensorDataType::FP32, MR> cuda_output( cuda_ctx->getDevice(), shape );
        Tensor<TensorDataType::FP32, MR> cuda_output_grad( cuda_ctx->getDevice(), shape );
        Tensor<TensorDataType::FP32, MR> cuda_input_grad( cuda_ctx->getDevice(), shape );

        copy( cpu_input, cuda_input );
        copy( cpu_output_grad, cuda_output_grad );

        cuda_gelu->setTraining( true );
        cuda_gelu->build( shape );
        cuda_gelu->forward( cuda_input, cuda_output );
        cuda_gelu->backward( cuda_input, cuda_output_grad, cuda_input_grad );

        // Compare results
        auto cuda_output_host = toHost<TensorDataType::FP32>( cuda_output );
        auto cuda_input_grad_host = toHost<TensorDataType::FP32>( cuda_input_grad );

        const float tolerance = 1e-3f;

        // Verify forward pass equivalence
        for (size_t i = 0; i < cpu_output.size(); ++i)
        {
            float cpu_val = cpu_output.data()[i];
            float gpu_val = cuda_output_host.data()[i];
            float diff = std::abs( cpu_val - gpu_val );

            EXPECT_LT( diff, tolerance )
                << "Forward pass mismatch at index " << i
                << ": CPU=" << cpu_val
                << ", CUDA=" << gpu_val;
        }

        // Verify backward pass equivalence
        for (size_t i = 0; i < cpu_input_grad.size(); ++i)
        {
            float cpu_grad = cpu_input_grad.data()[i];
            float gpu_grad = cuda_input_grad_host.data()[i];
            float diff = std::abs( cpu_grad - gpu_grad );

            EXPECT_LT( diff, tolerance )
                << "Backward pass mismatch at index " << i
                << ": CPU=" << cpu_grad
                << ", CUDA=" << gpu_grad;
        }
    }

    TEST_F( GeluCudaTests, CpuCuda_ForwardEquivalence_OrConstructorThrows )
    {
        bool cpu_registered = isOperationRegistered<DeviceType::Cpu, TensorDataType::FP32>( "GeluOp" );
        bool cuda_registered = isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" );

        if (!cpu_registered || !cuda_registered)
        {
            if (!cpu_registered)
            {
                auto cpu_ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
                EXPECT_THROW( (GeluCpuModule( cpu_ctx, GeluConfig() )), std::runtime_error );
            }
            if (!cuda_registered)
            {
                auto cuda_ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );
                EXPECT_THROW( (GeluCudaModule( cuda_ctx, GeluConfig() )), std::runtime_error );
            }

            return;
        }

        // Forward equivalence test (existing test preserved)
        std::vector<int64_t> shape = { 2, 2, 4 };
        GeluConfig config;
        auto cpu_ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto cuda_ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

        auto cpu_gelu = std::make_shared<GeluCpuModule>( cpu_ctx, config );
        auto cuda_gelu = std::make_shared<GeluCudaModule>( cuda_ctx, config );

        auto cpu_device = cpu_ctx->getDevice();
        Tensor<TensorDataType::FP32, CpuMemoryResource> cpu_input( cpu_device, shape );
        Tensor<TensorDataType::FP32, CpuMemoryResource> cpu_output( cpu_device, shape );

        for (size_t i = 0; i < cpu_input.size(); ++i)
            cpu_input.data()[i] = static_cast<float>( i ) / cpu_input.size() * 4.0f - 2.0f;

        cpu_gelu->build( cpu_input.shape() );
        cpu_gelu->forward( cpu_input, cpu_output );

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

    // ========================================================================
    // Metadata Tests
    // ========================================================================

    TEST_F( GeluCudaTests, ToString_ContainsGeluOrConstructorThrows )
    {
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

    TEST_F( GeluCudaTests, Construct_WithExecutionContext_WorksOrThrows )
    {
        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cuda>>( 0 );

        if (!isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" ))
        {
            EXPECT_THROW( (GeluCudaModule( ctx, GeluConfig() )), std::runtime_error );
            return;
        }

        auto d = GeluCudaTestData::CreateWithExecutionContext( ctx, batch_, seq_, chan_ );
        EXPECT_EQ( ctx->getDevice()->getDeviceType(), DeviceType::Cuda );
    }
}