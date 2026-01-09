/**
 * @file Gelu.Cpu.cpp
 * @brief Unit tests for GELU activation module on CPU device.
 *
 * Verifies basic API, forward/backward invocation, config, and constructor
 * behavior for the CPU-specialized Gelu module.
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

namespace Dnn::Components::Activations::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    using MR = CpuMemoryResource;
    using GeluCpu = Gelu<DeviceType::Cpu, dtype_t::FP32>;

    struct GeluCpuTestData
    {
        std::vector<int64_t> shape;
        std::shared_ptr<GeluCpu> gelu;

        static GeluCpuTestData Create(
            int64_t batch, int64_t seq, int64_t chan )
        {
            GeluCpuTestData d;
            d.shape = { batch, seq, chan };

            GeluConfig config;
            d.gelu = std::make_shared<GeluCpu>( "gelu", config, Device::Cpu() );

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
            data_.gelu.reset();
        }

        float geluReference( float x )
        {
            constexpr float sqrt_2_over_pi = 0.7978845608f;
            constexpr float coeff = 0.044715f;
            float x_cubed = x * x * x;
            float tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);

            return 0.5f * x * (1.0f + std::tanh( tanh_arg ));
        }

        float geluGradientReference( float x )
        {
            constexpr float sqrt_2_over_pi = 0.7978845608f;
            constexpr float coeff = 0.044715f;

            float x_squared = x * x;
            float x_cubed = x * x_squared;

            float tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);
            float tanh_val = std::tanh( tanh_arg );
            float sech_squared = 1.0f - tanh_val * tanh_val;

            float d_tanh_arg = sqrt_2_over_pi * (1.0f + 3.0f * coeff * x_squared);

            return 0.5f * (1.0f + tanh_val) + 0.5f * x * sech_squared * d_tanh_arg;
        }

        dim_t batch_{ 0 }, seq_{ 0 }, chan_{ 0 };
        GeluCpuTestData data_;
    };

    // ========================================================================
    // Construction Tests
    // ========================================================================

    TEST_F( GeluCpuTests, Construct_WithDeviceId_ThrowsWhenOpNotRegistered )
    {
        DeviceId cpu_id = Device::Cpu();

        if ( !isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ) )
        {
            EXPECT_THROW(
                GeluCpu( "gelu", GeluConfig(), cpu_id ),
                std::runtime_error
            );
        }
        else
        {
            EXPECT_NO_THROW(
                GeluCpu( "gelu", GeluConfig(), cpu_id )
            );
        }
    }

    TEST_F( GeluCpuTests, Constructor_NoDeviceId_GetDeviceIdThrows )
    {
        GeluConfig cfg;
        GeluCpu gelu( "gelu", cfg );

        EXPECT_THROW(
            gelu.getDeviceId(),
            std::runtime_error
        );
    }

    TEST_F( GeluCpuTests, Constructor_InvalidConfig_ThrowsInvalidArgument )
    {
        GeluConfig bad_cfg = GeluConfig().withApproximationMethod( GeluConfig::ApproximationMethod::Exact );

        EXPECT_THROW(
            GeluCpu( "gelu", bad_cfg, Device::Cpu() ),
            std::invalid_argument
        );
    }

    TEST_F( GeluCpuTests, Constructor_DeviceTypeMismatch_ThrowsInvalidArgument )
    {
        DeviceId cuda_id = Device::Cuda( 0 );

        EXPECT_THROW(
            GeluCpu( "gelu", GeluConfig(), cuda_id ),
            std::invalid_argument
        );
    }

    // ========================================================================
    // Forward Pass Tests
    // ========================================================================

    TEST_F( GeluCpuTests, Forward_BehaviorDependsOnRegistration )
    {
        if ( !isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ) )
        {
            DeviceId cpu_id = Device::Cpu();
            EXPECT_THROW(
                GeluCpu( "gelu", GeluConfig(), cpu_id ),
                std::runtime_error
            );
            return;
        }

        auto d = GeluCpuTestData::Create( batch_, seq_, chan_ );
        auto device = d.gelu->getDeviceId();

        Tensor<dtype_t::FP32, MR> input( device, d.shape );

        for ( size_t i = 0; i < input.size(); ++i )
        {
            input.data()[ i ] = static_cast<float>( i ) / input.size() * 4.0f - 2.0f;
        }

        d.gelu->build( d.shape );

        auto& out = d.gelu->forward( input );

        EXPECT_EQ( out.size(), input.size() );
    }

    TEST_F( GeluCpuTests, Forward_OutputMatchesReference )
    {
        if ( !isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ) )
        {
            GTEST_SKIP() << "GeluOp not registered for CPU FP32";
        }

        DeviceId device_id = Device::Cpu();
        auto gelu = std::make_shared<GeluCpu>( "gelu", GeluConfig(), device_id );

        std::vector<int64_t> shape = { 2, 3, 4 };

        Tensor<dtype_t::FP32, MR> input( device_id, shape );

        for ( size_t i = 0; i < input.size(); ++i )
        {
            input.data()[ i ] = static_cast<float>( i ) / input.size() * 4.0f - 2.0f;
        }

        gelu->build( shape );

        auto& out = gelu->forward( input );

        const float tolerance = 1e-4f;

        for ( size_t i = 0; i < input.size(); ++i )
        {
            float input_val = input.data()[ i ];
            float expected = geluReference( input_val );
            float actual = out.data()[ i ];
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
        if ( !isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ) )
        {
            GTEST_SKIP() << "GeluOp not registered for CPU FP32";
        }

        DeviceId device_id = Device::Cpu();
        auto gelu = std::make_shared<GeluCpu>( "gelu", GeluConfig(), device_id );

        std::vector<int64_t> shape = { 2, 4, 8 };

        Tensor<dtype_t::FP32, MR> input( device_id, shape );
        Tensor<dtype_t::FP32, MR> output_grad( device_id, shape );

        for ( size_t i = 0; i < input.size(); ++i )
        {
            input.data()[ i ] = static_cast<float>( i ) / input.size() * 4.0f - 2.0f;
            output_grad.data()[ i ] = 1.0f;
        }

        gelu->build( shape );
        gelu->setTraining( true );
        gelu->forward( input );

        auto& in_grad = gelu->backward( input, output_grad );

        (void)in_grad; // silence unused-variable in release builds
    }

    TEST_F( GeluCpuTests, Backward_ProducesCorrectShape )
    {
        if ( !isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ) )
        {
            GTEST_SKIP() << "GeluOp not registered for CPU FP32";
        }

        DeviceId device_id = Device::Cpu();
        auto gelu = std::make_shared<GeluCpu>( "gelu", GeluConfig(), device_id );

        std::vector<int64_t> shape = { 3, 5, 7 };

        Tensor<dtype_t::FP32, MR> input( device_id, shape );
        Tensor<dtype_t::FP32, MR> output_grad( device_id, shape );

        gelu->build( shape );
        gelu->setTraining( true );
        gelu->forward( input );

        auto& in_grad = gelu->backward( input, output_grad );

        EXPECT_EQ( in_grad.shape(), input.shape() );
        EXPECT_EQ( in_grad.size(), input.size() );
    }

    TEST_F( GeluCpuTests, Backward_GradientsMatchReference )
    {
        if ( !isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ) )
        {
            GTEST_SKIP() << "GeluOp not registered for CPU FP32";
        }

        DeviceId device_id = Device::Cpu();
        auto gelu = std::make_shared<GeluCpu>( "gelu", GeluConfig(), device_id );

        std::vector<int64_t> shape = { 2, 3, 4 };

        Tensor<dtype_t::FP32, MR> input( device_id, shape );
        Tensor<dtype_t::FP32, MR> output_grad( device_id, shape );

        for ( size_t i = 0; i < input.size(); ++i )
        {
            input.data()[ i ] = static_cast<float>( i ) / input.size() * 4.0f - 2.0f;
            output_grad.data()[ i ] = 1.0f;
        }

        gelu->build( shape );
        gelu->setTraining( true );
        gelu->forward( input );

        auto& in_grad = gelu->backward( input, output_grad );

        const float tolerance = 1e-3f;

        for ( size_t i = 0; i < input.size(); ++i )
        {
            float x = input.data()[ i ];
            float grad_out = output_grad.data()[ i ];
            float expected = geluGradientReference( x ) * grad_out;
            float actual = in_grad.data()[ i ];
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
        if ( !isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ) )
        {
            GTEST_SKIP() << "GeluOp not registered for CPU FP32";
        }

        DeviceId device_id = Device::Cpu();
        auto gelu = std::make_shared<GeluCpu>( "gelu", GeluConfig(), device_id );

        std::vector<int64_t> shape = { 2, 3, 4 };

        Tensor<dtype_t::FP32, MR> input( device_id, shape );
        Tensor<dtype_t::FP32, MR> output_grad( device_id, shape );

        for ( size_t i = 0; i < input.size(); ++i )
        {
            input.data()[ i ] = static_cast<float>( i ) / input.size() * 4.0f - 2.0f;
            output_grad.data()[ i ] = static_cast<float>( i + 1 ) * 0.1f;
        }

        gelu->build( shape );
        gelu->setTraining( true );
        gelu->forward( input );

        auto& in_grad = gelu->backward( input, output_grad );

        const float tolerance = 1e-3f;

        for ( size_t i = 0; i < input.size(); ++i )
        {
            float x = input.data()[ i ];
            float grad_out = output_grad.data()[ i ];
            float expected = geluGradientReference( x ) * grad_out;
            float actual = in_grad.data()[ i ];
            float diff = std::abs( expected - actual );

            EXPECT_LT( diff, tolerance )
                << "Chain rule gradient mismatch at index " << i;
        }
    }

    TEST_F( GeluCpuTests, Backward_HandlesZeroOutputGradient )
    {
        if ( !isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ) )
        {
            GTEST_SKIP() << "GeluOp not registered for CPU FP32";
        }

        DeviceId device_id = Device::Cpu();
        auto gelu = std::make_shared<GeluCpu>( "gelu", GeluConfig(), device_id );

        std::vector<int64_t> shape = { 2, 3, 4 };

        Tensor<dtype_t::FP32, MR> input( device_id, shape );
        Tensor<dtype_t::FP32, MR> output_grad( device_id, shape );

        for ( size_t i = 0; i < input.size(); ++i )
        {
            input.data()[ i ] = static_cast<float>( i ) / input.size() * 2.0f;
            output_grad.data()[ i ] = 0.0f;
        }

        gelu->build( shape );
        gelu->setTraining( true );
        gelu->forward( input );

        auto& in_grad = gelu->backward( input, output_grad );

        for ( size_t i = 0; i < in_grad.size(); ++i )
        {
            EXPECT_FLOAT_EQ( in_grad.data()[ i ], 0.0f )
                << "Expected zero gradient at index " << i;
        }
    }

    TEST_F( GeluCpuTests, Backward_HandlesEdgeCaseInputs )
    {
        if ( !isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ) )
        {
            GTEST_SKIP() << "GeluOp not registered for CPU FP32";
        }

        DeviceId device_id = Device::Cpu();
        auto gelu = std::make_shared<GeluCpu>( "gelu", GeluConfig(), device_id );

        std::vector<int64_t> shape = { 1, 8 };

        Tensor<dtype_t::FP32, MR> input( device_id, shape );
        Tensor<dtype_t::FP32, MR> output_grad( device_id, shape );

        std::vector<float> test_values = { -10.0f, -1.0f, -0.1f, 0.0f, 0.1f, 1.0f, 10.0f, 100.0f };

        for ( size_t i = 0; i < test_values.size(); ++i )
        {
            input.data()[ i ] = test_values[ i ];
            output_grad.data()[ i ] = 1.0f;
        }

        gelu->build( shape );
        gelu->setTraining( true );

        EXPECT_NO_THROW( gelu->forward( input ) );

        auto& in_grad = gelu->backward( input, output_grad );

        for ( size_t i = 0; i < in_grad.size(); ++i )
        {
            EXPECT_FALSE( std::isnan( in_grad.data()[ i ] ) )
                << "NaN gradient at index " << i << " for input " << test_values[ i ];
            EXPECT_FALSE( std::isinf( in_grad.data()[ i ] ) )
                << "Inf gradient at index " << i << " for input " << test_values[ i ];
        }
    }

    TEST_F( GeluCpuTests, Backward_ThrowsWhenNotBuilt )
    {
        if ( !isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ) )
        {
            GTEST_SKIP() << "GeluOp not registered for CPU FP32";
        }

        DeviceId device_id = Device::Cpu();
        auto gelu = std::make_shared<GeluCpu>( "gelu", GeluConfig(), device_id );

        std::vector<int64_t> shape = { 2, 3 };

        Tensor<dtype_t::FP32, MR> input( device_id, shape );
        Tensor<dtype_t::FP32, MR> output_grad( device_id, shape );

        EXPECT_THROW( gelu->backward( input, output_grad ), std::runtime_error );
    }

    TEST_F( GeluCpuTests, Backward_AccumulatesGradients )
    {
        if ( !isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ) )
        {
            GTEST_SKIP() << "GeluOp not registered for CPU FP32";
        }

        DeviceId device_id = Device::Cpu();
        auto gelu = std::make_shared<GeluCpu>( "gelu", GeluConfig(), device_id );

        std::vector<int64_t> shape = { 2, 3 };

        Tensor<dtype_t::FP32, MR> input( device_id, shape );
        Tensor<dtype_t::FP32, MR> output_grad_zero( device_id, shape );
        Tensor<dtype_t::FP32, MR> output_grad_one( device_id, shape );

        for ( size_t i = 0; i < input.size(); ++i )
        {
            input.data()[ i ] = 0.5f;
            output_grad_zero.data()[ i ] = 0.0f;
            output_grad_one.data()[ i ] = 1.0f;
        }

        gelu->build( shape );
        gelu->setTraining( true );
        gelu->forward( input );

        // First obtain the owned input-grad tensor and initialize it to 10.0f
        auto& in_grad = gelu->backward( input, output_grad_zero );

        for ( size_t i = 0; i < in_grad.size(); ++i )
        {
            in_grad.data()[ i ] = 10.0f;
        }

        // Now run backward with non-zero output gradient; backend should accumulate
        gelu->backward( input, output_grad_one );

        float expected_delta = geluGradientReference( 0.5f ) * 1.0f;

        for ( size_t i = 0; i < in_grad.size(); ++i )
        {
            float expected_total = 10.0f + expected_delta;
            float actual = in_grad.data()[ i ];
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
        if ( !isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ) )
        {
            DeviceId cpu_id = Device::Cpu();
            EXPECT_THROW(
                GeluCpu( "gelu", GeluConfig(), cpu_id ),
                std::runtime_error
            );
            return;
        }

        auto d = GeluCpuTestData::Create( batch_, seq_, chan_ );
        auto s = d.gelu->toString();

        EXPECT_FALSE( s.empty() );
        EXPECT_NE( s.find( "Gelu" ), std::string::npos );
    }

    TEST_F( GeluCpuTests, DefaultApproximationMethod_IsTanhOrConstructorThrows )
    {
        if ( !isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ) )
        {
            DeviceId cpu_id = Device::Cpu();
            EXPECT_THROW(
                GeluCpu( "gelu", GeluConfig(), cpu_id ),
                std::runtime_error
            );
            return;
        }

        auto d = GeluCpuTestData::Create( batch_, seq_, chan_ );

        EXPECT_EQ( d.gelu->getApproximationMethod(), GeluConfig::ApproximationMethod::Tanh );
    }

    TEST_F( GeluCpuTests, DeviceId_DeviceTypeMatchesOrConstructorThrows )
    {
        if ( !isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ) )
        {
            DeviceId cpu_id = Device::Cpu();
            EXPECT_THROW(
                GeluCpu( "gelu", GeluConfig(), cpu_id ),
                std::runtime_error
            );
            return;
        }

        auto d = GeluCpuTestData::Create( batch_, seq_, chan_ );
        auto dev = d.gelu->getDeviceId();

        EXPECT_EQ( dev.type, DeviceType::Cpu );
    }

    TEST_F( GeluCpuTests, Construct_WithDeviceId_WorksOrThrows )
    {
        if ( !isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ) )
        {
            EXPECT_THROW(
                GeluCpu( "gelu", GeluConfig(), Device::Cpu() ),
                std::runtime_error
            );

            return;
        }

        auto d = GeluCpuTestData::Create( batch_, seq_, chan_ );

        EXPECT_EQ( d.gelu->getDeviceId().type, DeviceType::Cpu );
    }

    TEST_F( GeluCpuTests, Synchronize_ParameterCount_SetTraining )
    {
        if ( !isOperationRegistered<DeviceType::Cpu, dtype_t::FP32>( "GeluOp" ) )
        {
            GTEST_SKIP() << "GeluOp not registered for CPU FP32";
        }

        auto d = GeluCpuTestData::Create( batch_, seq_, chan_ );
        d.gelu->build( d.shape );

        EXPECT_NO_THROW( d.gelu->synchronize() );

        EXPECT_EQ( d.gelu->parameterCount(), 0u );

        d.gelu->setTraining( true );
        EXPECT_TRUE( d.gelu->isTraining() );

        d.gelu->setTraining( false );
        EXPECT_FALSE( d.gelu->isTraining() );
    }
}