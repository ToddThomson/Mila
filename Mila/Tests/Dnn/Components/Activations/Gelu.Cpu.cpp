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
 * - Forward and backward passes are validated for correctness.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <algorithm>

import Mila;

namespace Modules::Activations::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    using MR = CpuMemoryResource;
    using GeluCpu = Gelu<DeviceType::Cpu, dtype_t::FP32>;

    struct GeluCpuTestData
    {
        std::vector<int64_t> shape;
        std::shared_ptr<GeluCpu> gelu_;

        static GeluCpuTestData CreateWithExecutionContext(
            std::shared_ptr<ExecutionContext<DeviceType::Cpu>> ctx,
            int64_t batch, int64_t seq, int64_t chan )
        {
            GeluCpuTestData d;
            d.shape = { batch, seq, chan };
            GeluConfig config;
            
            d.gelu_ = std::make_shared<GeluCpu>( ctx, config );
            
            return d;
        }
    };

    class GeluCpuTests : public ::testing::Test
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
            data_by_ctx_.gelu_.reset();
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

        dim_t batch_{ 0 }, seq_{ 0 }, chan_{ 0 };
        GeluCpuTestData data_by_ctx_;
    };

    // ========================================================================
    // Construction Tests
    // ========================================================================

    TEST_F( GeluCpuTests, Construct_WithExecutionContext_ThrowsWhenOpNotRegistered )
    {
        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();

        if (!isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ))
        {
            EXPECT_THROW( GeluCpu( ctx, GeluConfig() ), std::runtime_error );
        }
        else
        {
            EXPECT_NO_THROW( GeluCpu( ctx, GeluConfig() ) );
        }
    }

    TEST_F( GeluCpuTests, Constructor_NullExecutionContext_ThrowsInvalidArgument )
    {
        std::shared_ptr<ExecutionContext<DeviceType::Cpu>> null_ctx = nullptr;
        EXPECT_THROW( GeluCpu( null_ctx, GeluConfig() ), std::invalid_argument );
    }

    TEST_F( GeluCpuTests, Constructor_InvalidConfig_ThrowsInvalidArgument )
    {
        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        GeluConfig bad_cfg = GeluConfig().withApproximationMethod( GeluConfig::ApproximationMethod::Exact );

        EXPECT_THROW( GeluCpu( ctx, bad_cfg ), std::invalid_argument );
    }

    // ========================================================================
    // Forward Pass Tests
    // ========================================================================

    TEST_F( GeluCpuTests, Forward_BehaviorDependsOnRegistration )
    {
        if (!isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ))
        {
            auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
            EXPECT_THROW( GeluCpu( ctx, GeluConfig() ), std::runtime_error );
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

        d.gelu_->build( d.shape );

        EXPECT_NO_THROW( d.gelu_->forward( input, output ) );
        EXPECT_EQ( output.size(), input.size() );
    }

    TEST_F( GeluCpuTests, Forward_OutputMatchesReference )
    {
        if (!isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ))
        {
            GTEST_SKIP() << "GeluOp not registered for CPU FP32";
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto gelu = std::make_shared<GeluCpu>( ctx, GeluConfig() );

        std::vector<int64_t> shape = { 2, 3, 4 };
        auto device = ctx->getDevice();

        Tensor<dtype_t::FP32, MR> input( device, shape );
        Tensor<dtype_t::FP32, MR> output( device, shape );

        for (size_t i = 0; i < input.size(); ++i)
            input.data()[i] = static_cast<float>( i ) / input.size() * 4.0f - 2.0f;

        gelu->build( shape );
        gelu->forward( input, output );

        // Verify against reference implementation
        const float tolerance = 1e-4f;
        for (size_t i = 0; i < input.size(); ++i)
        {
            float input_val = input.data()[i];
            float expected = geluReference( input_val );
            float actual = output.data()[i];
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

    TEST_F( GeluCpuTests, Backward_ExecutesWithoutError )
    {
        if (!isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ))
        {
            GTEST_SKIP() << "GeluOp not registered for CPU FP32";
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto gelu = std::make_shared<GeluCpu>( ctx, GeluConfig() );

        std::vector<int64_t> shape = { 2, 4, 8 };
        auto device = ctx->getDevice();

        Tensor<dtype_t::FP32, MR> input( device, shape );
        Tensor<dtype_t::FP32, MR> output( device, shape );
        Tensor<dtype_t::FP32, MR> output_grad( device, shape );
        Tensor<dtype_t::FP32, MR> input_grad( device, shape );

        for (size_t i = 0; i < input.size(); ++i)
        {
            input.data()[i] = static_cast<float>( i ) / input.size() * 4.0f - 2.0f;
            output_grad.data()[i] = 1.0f;
            input_grad.data()[i] = 0.0f;
        }

        gelu->setTraining( true );
        gelu->build( shape );
        gelu->forward( input, output );

        EXPECT_NO_THROW( gelu->backward( input, output_grad, input_grad ) );
    }

    TEST_F( GeluCpuTests, Backward_ProducesCorrectShape )
    {
        if (!isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ))
        {
            GTEST_SKIP() << "GeluOp not registered for CPU FP32";
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto gelu = std::make_shared<GeluCpu>( ctx, GeluConfig() );

        std::vector<int64_t> shape = { 3, 5, 7 };
        auto device = ctx->getDevice();

        Tensor<dtype_t::FP32, MR> input( device, shape );
        Tensor<dtype_t::FP32, MR> output( device, shape );
        Tensor<dtype_t::FP32, MR> output_grad( device, shape );
        Tensor<dtype_t::FP32, MR> input_grad( device, shape );

        gelu->setTraining( true );
        gelu->build( shape );
        gelu->forward( input, output );
        gelu->backward( input, output_grad, input_grad );

        EXPECT_EQ( input_grad.shape(), input.shape() );
        EXPECT_EQ( input_grad.size(), input.size() );
    }

    TEST_F( GeluCpuTests, Backward_GradientsMatchReference )
    {
        if (!isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ))
        {
            GTEST_SKIP() << "GeluOp not registered for CPU FP32";
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto gelu = std::make_shared<GeluCpu>( ctx, GeluConfig() );

        std::vector<int64_t> shape = { 2, 3, 4 };
        auto device = ctx->getDevice();

        Tensor<dtype_t::FP32, MR> input( device, shape );
        Tensor<dtype_t::FP32, MR> output( device, shape );
        Tensor<dtype_t::FP32, MR> output_grad( device, shape );
        Tensor<dtype_t::FP32, MR> input_grad( device, shape );

        for (size_t i = 0; i < input.size(); ++i)
        {
            input.data()[i] = static_cast<float>( i ) / input.size() * 4.0f - 2.0f;
            output_grad.data()[i] = 1.0f;
            input_grad.data()[i] = 0.0f;
        }

        gelu->setTraining( true );
        gelu->build( shape );
        gelu->forward( input, output );
        gelu->backward( input, output_grad, input_grad );

        // Verify against reference gradient computation
        const float tolerance = 1e-3f;
        for (size_t i = 0; i < input.size(); ++i)
        {
            float x = input.data()[i];
            float grad_out = output_grad.data()[i];
            float expected = geluGradientReference( x ) * grad_out;
            float actual = input_grad.data()[i];
            float diff = std::abs( expected - actual );

            EXPECT_LT( diff, tolerance )
                << "Backward gradient mismatch at index " << i
                << ": input=" << x
                << ", expected=" << expected
                << ", actual=" << actual;
        }
    }

    TEST_F( GeluCpuTests, Backward_ChainRuleWithNonUniformGradients )
    {
        if (!isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ))
        {
            GTEST_SKIP() << "GeluOp not registered for CPU FP32";
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto gelu = std::make_shared<GeluCpu>( ctx, GeluConfig() );

        std::vector<int64_t> shape = { 2, 3, 4 };
        auto device = ctx->getDevice();

        Tensor<dtype_t::FP32, MR> input( device, shape );
        Tensor<dtype_t::FP32, MR> output( device, shape );
        Tensor<dtype_t::FP32, MR> output_grad( device, shape );
        Tensor<dtype_t::FP32, MR> input_grad( device, shape );

        // Non-uniform gradients
        for (size_t i = 0; i < input.size(); ++i)
        {
            input.data()[i] = static_cast<float>( i ) / input.size() * 4.0f - 2.0f;
            output_grad.data()[i] = static_cast<float>( i + 1 ) * 0.1f;
            input_grad.data()[i] = 0.0f;
        }

        gelu->setTraining( true );
        gelu->build( shape );
        gelu->forward( input, output );
        gelu->backward( input, output_grad, input_grad );

        const float tolerance = 1e-3f;
        for (size_t i = 0; i < input.size(); ++i)
        {
            float x = input.data()[i];
            float grad_out = output_grad.data()[i];
            float expected = geluGradientReference( x ) * grad_out;
            float actual = input_grad.data()[i];
            float diff = std::abs( expected - actual );

            EXPECT_LT( diff, tolerance )
                << "Chain rule gradient mismatch at index " << i;
        }
    }

    TEST_F( GeluCpuTests, Backward_HandlesZeroOutputGradient )
    {
        if (!isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ))
        {
            GTEST_SKIP() << "GeluOp not registered for CPU FP32";
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto gelu = std::make_shared<GeluCpu>( ctx, GeluConfig() );

        std::vector<int64_t> shape = { 2, 3, 4 };
        auto device = ctx->getDevice();

        Tensor<dtype_t::FP32, MR> input( device, shape );
        Tensor<dtype_t::FP32, MR> output( device, shape );
        Tensor<dtype_t::FP32, MR> output_grad( device, shape );
        Tensor<dtype_t::FP32, MR> input_grad( device, shape );

        for (size_t i = 0; i < input.size(); ++i)
        {
            input.data()[i] = static_cast<float>( i ) / input.size() * 2.0f;
            output_grad.data()[i] = 0.0f;  // Zero gradient
            input_grad.data()[i] = 0.0f;
        }

        gelu->setTraining( true );
        gelu->build( shape );
        gelu->forward( input, output );
        gelu->backward( input, output_grad, input_grad );

        // All gradients should be zero
        for (size_t i = 0; i < input_grad.size(); ++i)
        {
            EXPECT_FLOAT_EQ( input_grad.data()[i], 0.0f )
                << "Expected zero gradient at index " << i;
        }
    }

    TEST_F( GeluCpuTests, Backward_HandlesEdgeCaseInputs )
    {
        if (!isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ))
        {
            GTEST_SKIP() << "GeluOp not registered for CPU FP32";
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto gelu = std::make_shared<GeluCpu>( ctx, GeluConfig() );

        std::vector<int64_t> shape = { 1, 8 };
        auto device = ctx->getDevice();

        Tensor<dtype_t::FP32, MR> input( device, shape );
        Tensor<dtype_t::FP32, MR> output( device, shape );
        Tensor<dtype_t::FP32, MR> output_grad( device, shape );
        Tensor<dtype_t::FP32, MR> input_grad( device, shape );

        // Edge cases: large positive, large negative, zero, small values
        std::vector<float> test_values = { -10.0f, -1.0f, -0.1f, 0.0f, 0.1f, 1.0f, 10.0f, 100.0f };

        for (size_t i = 0; i < test_values.size(); ++i)
        {
            input.data()[i] = test_values[i];
            output_grad.data()[i] = 1.0f;
            input_grad.data()[i] = 0.0f;
        }

        gelu->setTraining( true );
        gelu->build( shape );

        EXPECT_NO_THROW( gelu->forward( input, output ) );
        EXPECT_NO_THROW( gelu->backward( input, output_grad, input_grad ) );

        // Verify no NaN or Inf in gradients
        for (size_t i = 0; i < input_grad.size(); ++i)
        {
            EXPECT_FALSE( std::isnan( input_grad.data()[i] ) )
                << "NaN gradient at index " << i << " for input " << test_values[i];
            EXPECT_FALSE( std::isinf( input_grad.data()[i] ) )
                << "Inf gradient at index " << i << " for input " << test_values[i];
        }
    }

    TEST_F( GeluCpuTests, Backward_ThrowsWhenNotBuilt )
    {
        if (!isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ))
        {
            GTEST_SKIP() << "GeluOp not registered for CPU FP32";
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto gelu = std::make_shared<GeluCpu>( ctx, GeluConfig() );

        std::vector<int64_t> shape = { 2, 3 };
        auto device = ctx->getDevice();

        Tensor<dtype_t::FP32, MR> input( device, shape );
        Tensor<dtype_t::FP32, MR> output_grad( device, shape );
        Tensor<dtype_t::FP32, MR> input_grad( device, shape );

        gelu->setTraining( true );
        // Note: NOT calling build() here

        // Backward should throw because module is not built
        EXPECT_THROW( gelu->backward( input, output_grad, input_grad ), std::runtime_error );
    }

    TEST_F( GeluCpuTests, Backward_AccumulatesGradients )
    {
        if (!isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ))
        {
            GTEST_SKIP() << "GeluOp not registered for CPU FP32";
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto gelu = std::make_shared<GeluCpu>( ctx, GeluConfig() );

        std::vector<int64_t> shape = { 2, 3 };
        auto device = ctx->getDevice();

        Tensor<dtype_t::FP32, MR> input( device, shape );
        Tensor<dtype_t::FP32, MR> output( device, shape );
        Tensor<dtype_t::FP32, MR> output_grad( device, shape );
        Tensor<dtype_t::FP32, MR> input_grad( device, shape );

        for (size_t i = 0; i < input.size(); ++i)
        {
            input.data()[i] = 0.5f;
            output_grad.data()[i] = 1.0f;
            input_grad.data()[i] = 10.0f;  // Pre-existing gradient
        }

        gelu->setTraining( true );
        gelu->build( shape );
        gelu->forward( input, output );

        // Store initial gradient value
        std::vector<float> initial_grads( input_grad.size() );
        for (size_t i = 0; i < input_grad.size(); ++i)
        {
            initial_grads[i] = input_grad.data()[i];
        }

        gelu->backward( input, output_grad, input_grad );

        // Verify gradients were accumulated (not overwritten)
        float expected_delta = geluGradientReference( 0.5f ) * 1.0f;

        for (size_t i = 0; i < input_grad.size(); ++i)
        {
            float expected_total = initial_grads[i] + expected_delta;
            float actual = input_grad.data()[i];
            float diff = std::abs( expected_total - actual );

            EXPECT_LT( diff, 1e-3f )
                << "Gradient accumulation failed at index " << i
                << ": expected=" << expected_total
                << ", actual=" << actual;
        }
    }

    // ========================================================================
    // Metadata and Configuration Tests
    // ========================================================================

    TEST_F( GeluCpuTests, ToString_ContainsGeluOrConstructorThrows )
    {
        if (!isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ))
        {
            auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
            EXPECT_THROW( GeluCpu( ctx, GeluConfig() ), std::runtime_error );
            return;
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto d = GeluCpuTestData::CreateWithExecutionContext( ctx, batch_, seq_, chan_ );

        auto s = d.gelu_->toString();

        EXPECT_FALSE( s.empty() );
        EXPECT_NE( s.find( "Gelu" ), std::string::npos );
    }

    TEST_F( GeluCpuTests, DefaultApproximationMethod_IsTanhOrConstructorThrows )
    {
        if (!isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ))
        {
            auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
            EXPECT_THROW( GeluCpu( ctx, GeluConfig() ), std::runtime_error );
            return;
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto d = GeluCpuTestData::CreateWithExecutionContext( ctx, batch_, seq_, chan_ );
        EXPECT_EQ( d.gelu_->getApproximationMethod(), GeluConfig::ApproximationMethod::Tanh );
    }

    TEST_F( GeluCpuTests, ExecutionContext_DeviceTypeMatchesOrConstructorThrows )
    {
        if (!isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ))
        {
            auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
            EXPECT_THROW( GeluCpu( ctx, GeluConfig() ), std::runtime_error );
            return;
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto d = GeluCpuTestData::CreateWithExecutionContext( ctx, batch_, seq_, chan_ );

        auto dev = ctx->getDevice();

        ASSERT_NE( dev, nullptr );
        EXPECT_EQ( dev->getDeviceType(), DeviceType::Cpu );
    }

    TEST_F( GeluCpuTests, Construct_WithExecutionContext_WorksOrThrows )
    {
        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();

        if (!isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ))
        {
            EXPECT_THROW( GeluCpu( ctx, GeluConfig() ), std::runtime_error );
            return;
        }

        auto d = GeluCpuTestData::CreateWithExecutionContext( ctx, batch_, seq_, chan_ );
        EXPECT_EQ( ctx->getDevice()->getDeviceType(), DeviceType::Cpu );
    }

    TEST_F( GeluCpuTests, Synchronize_ParameterCount_SetTraining )
    {
        if (!isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ))
        {
            GTEST_SKIP() << "GeluOp not registered for CPU FP32";
        }

        auto ctx = std::make_shared<ExecutionContext<DeviceType::Cpu>>();
        auto d = GeluCpuTestData::CreateWithExecutionContext( ctx, batch_, seq_, chan_ );

        EXPECT_NO_THROW( d.gelu_->synchronize() );

        EXPECT_EQ( d.gelu_->parameterCount(), 0u );

        d.gelu_->setTraining( true );
        EXPECT_TRUE( d.gelu_->isTraining() );

        d.gelu_->setTraining( false );
        EXPECT_FALSE( d.gelu_->isTraining() );
    }
}