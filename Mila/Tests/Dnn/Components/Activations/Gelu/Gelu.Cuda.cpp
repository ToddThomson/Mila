/**
 * @file Gelu.Cuda.cpp
 * @brief Unit tests for GELU activation module on CUDA device.
 *
 * Tests assert behavior for both registered and unregistered backend operation:
 * - If the backend operation is registered the module should construct and operate.
 * - If the backend operation is not registered the Gelu constructor must throw.
 * - Forward and backward passes are validated for correctness.
 * - CPU-CUDA equivalence is verified for forward results.
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

namespace Dnn::Components::Activations::Tests
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    using MR = CudaDeviceMemoryResource;

    using GeluCuda = Gelu<DeviceType::Cuda, TensorDataType::FP32>;
    using GeluCpu = Gelu<DeviceType::Cpu, TensorDataType::FP32>;

    struct GeluCudaTestData
    {
        std::vector<int64_t> shape;
        std::shared_ptr<GeluCuda> gelu_module;
        std::unique_ptr<IExecutionContext> exec_context;

        static GeluCudaTestData Create(
            int64_t batch, int64_t seq, int64_t chan )
        {
            GeluCudaTestData d;
            d.shape = { batch, seq, chan };
            d.exec_context = createExecutionContext( Device::Cuda( 0 ) );

            GeluConfig config;
            d.gelu_module = std::make_shared<GeluCuda>( "gelu", config, Device::Cuda( 0 ) );

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
            data_.gelu_module.reset();
            data_.exec_context.reset();
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

        size_t batch_{ 0 }, seq_{ 0 }, chan_{ 0 };
        GeluCudaTestData data_;
    };

    // ========================================================================
    // Construction Tests
    // ========================================================================

    TEST_F( GeluCudaTests, Construct_WithDeviceId_BehaviorDependsOnRegistration )
    {
        DeviceId cuda_id = Device::Cuda( 0 );

        if ( isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" ) )
        {
            EXPECT_NO_THROW(
                GeluCuda( "gelu", GeluConfig(), cuda_id )
            );
        }
        else
        {
            EXPECT_THROW(
                GeluCuda( "gelu", GeluConfig(), cuda_id ),
                std::runtime_error
            );
        }
    }

    TEST_F( GeluCudaTests, Constructor_NoDeviceId_GetDeviceIdThrows )
    {
        GeluConfig cfg;
        GeluCuda gelu( "gelu", cfg );

        EXPECT_THROW(
            gelu.getDeviceId(),
            std::runtime_error
        );
    }

    TEST_F( GeluCudaTests, Constructor_DeviceTypeMismatch_ThrowsInvalidArgument )
    {
        DeviceId cpu_id = Device::Cpu();

        EXPECT_THROW(
            GeluCuda( "gelu", GeluConfig(), cpu_id ),
            std::invalid_argument
        );
    }

    // ========================================================================
    // Forward Pass Tests
    // ========================================================================

    TEST_F( GeluCudaTests, Forward_BehaviorDependsOnRegistration )
    {
        if ( !isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" ) )
        {
            DeviceId cuda_id = Device::Cuda( 0 );
            EXPECT_THROW(
                GeluCuda( "gelu", GeluConfig(), cuda_id ),
                std::runtime_error
            );
            return;
        }

        auto d = GeluCudaTestData::Create( batch_, seq_, chan_ );
        auto cuda_device = d.gelu_module->getDeviceId();

        Tensor<TensorDataType::FP32, MR> input( cuda_device, d.shape );

        auto cpu_device = Device::Cpu();
        Tensor<TensorDataType::FP32, CpuMemoryResource> host_input( cpu_device, d.shape );

        for ( size_t i = 0; i < host_input.size(); ++i )
        {
            host_input.data()[ i ] = static_cast<float>( i ) / host_input.size() * 4.0f - 2.0f;
        }

        copy( host_input, input );

        d.gelu_module->build( d.shape );

        Tensor<TensorDataType::FP32, MR>* out_ptr = nullptr;
        ASSERT_NO_THROW(
            {
                auto& out_ref = d.gelu_module->forward( input );
                out_ptr = &out_ref;
            }
        );
        ASSERT_NE( out_ptr, nullptr );

        EXPECT_EQ( out_ptr->size(), input.size() );
    }

    TEST_F( GeluCudaTests, Forward_OutputMatchesReference )
    {
        if ( !isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" ) )
        {
            GTEST_SKIP() << "GeluOp not registered for CUDA FP32";
        }

        DeviceId cuda_id = Device::Cuda( 0 );
        auto gelu = std::make_shared<GeluCuda>( "gelu", GeluConfig(), cuda_id );

        std::vector<int64_t> shape = { 2, 3, 4 };
        auto cuda_device = gelu->getDeviceId();

        Tensor<TensorDataType::FP32, MR> input( cuda_device, shape );

        DeviceId cpu_device{ DeviceType::Cpu, 0 };
        Tensor<TensorDataType::FP32, CpuMemoryResource> host_input( cpu_device, shape );

        for ( size_t i = 0; i < host_input.size(); ++i )
        {
            host_input.data()[ i ] = static_cast<float>( i ) / host_input.size() * 4.0f - 2.0f;
        }

        copy( host_input, input );

        gelu->build( shape );

        auto& out_tensor = gelu->forward( input );

        gelu->synchronize();

        auto host_output = toHost<TensorDataType::FP32>( out_tensor );

        const float tolerance = 1e-4f;

        for ( size_t i = 0; i < host_input.size(); ++i )
        {
            float input_val = host_input.data()[ i ];
            float expected = geluReference( input_val );
            float actual = host_output.data()[ i ];
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
        if ( !isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" ) )
        {
            GTEST_SKIP() << "GeluOp not registered for CUDA FP32";
        }

        DeviceId cuda_id = Device::Cuda( 0 );
        auto gelu = std::make_shared<GeluCuda>( "gelu", GeluConfig(), cuda_id );

        std::vector<int64_t> shape = { 2, 4, 8 };
        auto cuda_device = gelu->getDeviceId();

        Tensor<TensorDataType::FP32, MR> input( cuda_device, shape );
        Tensor<TensorDataType::FP32, MR> output_grad( cuda_device, shape );

        DeviceId cpu_device{ DeviceType::Cpu, 0 };
        Tensor<TensorDataType::FP32, CpuMemoryResource> host_input( cpu_device, shape );
        Tensor<TensorDataType::FP32, CpuMemoryResource> host_output_grad( cpu_device, shape );

        for ( size_t i = 0; i < host_input.size(); ++i )
        {
            host_input.data()[ i ] = static_cast<float>( i ) / host_input.size() * 4.0f - 2.0f;
            host_output_grad.data()[ i ] = 1.0f;
        }

        copy( host_input, input );
        copy( host_output_grad, output_grad );

        // Build before enabling training to satisfy Component lifecycle contract.
        gelu->build( shape );
        gelu->setTraining( true );
        gelu->forward( input );

        Tensor<TensorDataType::FP32, MR>* in_grad_ptr = nullptr;

        EXPECT_NO_THROW(
            {
                auto& ig = gelu->backward( input, output_grad );
                in_grad_ptr = &ig;
            }
        );

        ASSERT_NE( in_grad_ptr, nullptr );
    }

    TEST_F( GeluCudaTests, Backward_ProducesCorrectShape )
    {
        if ( !isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" ) )
        {
            GTEST_SKIP() << "GeluOp not registered for CUDA FP32";
        }

        DeviceId cuda_id = Device::Cuda( 0 );
        auto gelu = std::make_shared<GeluCuda>( "gelu", GeluConfig(), cuda_id );

        std::vector<int64_t> shape = { 3, 5, 7 };
        auto cuda_device = gelu->getDeviceId();

        Tensor<TensorDataType::FP32, MR> input( cuda_device, shape );
        Tensor<TensorDataType::FP32, MR> output_grad( cuda_device, shape );

        // Build before enabling training to satisfy Component lifecycle contract.
        gelu->build( shape );
        gelu->setTraining( true );
        gelu->forward( input );

        Tensor<TensorDataType::FP32, MR>* in_grad_ptr = nullptr;

        ASSERT_NO_THROW(
            {
                auto& ig = gelu->backward( input, output_grad );
                in_grad_ptr = &ig;
            }
        );

        ASSERT_NE( in_grad_ptr, nullptr );

        auto& in_grad = *in_grad_ptr;

        EXPECT_EQ( in_grad.shape(), input.shape() );
        EXPECT_EQ( in_grad.size(), input.size() );
    }

    TEST_F( GeluCudaTests, Backward_GradientsMatchReference )
    {
        if ( !isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" ) )
        {
            GTEST_SKIP() << "GeluOp not registered for CUDA FP32";
        }

        DeviceId cuda_id = Device::Cuda( 0 );
        auto gelu = std::make_shared<GeluCuda>( "gelu", GeluConfig(), cuda_id );

        std::vector<int64_t> shape = { 2, 3, 4 };
        auto cuda_device = gelu->getDeviceId();

        Tensor<TensorDataType::FP32, MR> input( cuda_device, shape );
        Tensor<TensorDataType::FP32, MR> output_grad( cuda_device, shape );

        DeviceId cpu_device{ DeviceType::Cpu, 0 };
        Tensor<TensorDataType::FP32, CpuMemoryResource> host_input( cpu_device, shape );
        Tensor<TensorDataType::FP32, CpuMemoryResource> host_output_grad( cpu_device, shape );

        for ( size_t i = 0; i < host_input.size(); ++i )
        {
            host_input.data()[ i ] = static_cast<float>( i ) / host_input.size() * 4.0f - 2.0f;
            host_output_grad.data()[ i ] = 1.0f;
        }

        copy( host_input, input );
        copy( host_output_grad, output_grad );

        // Build before enabling training to satisfy Component lifecycle contract.
        gelu->build( shape );
        gelu->setTraining( true );
        gelu->forward( input );

        Tensor<TensorDataType::FP32, MR>* in_grad_ptr = nullptr;

        ASSERT_NO_THROW(
            {
                auto& ig = gelu->backward( input, output_grad );
                in_grad_ptr = &ig;
            }
        );

        ASSERT_NE( in_grad_ptr, nullptr );

        auto& in_grad_device = *in_grad_ptr;
        auto host_input_grad_result = toHost<TensorDataType::FP32>( in_grad_device );

        const float tolerance = 1e-3f;

        for ( size_t i = 0; i < host_input.size(); ++i )
        {
            float x = host_input.data()[ i ];
            float grad_out = host_output_grad.data()[ i ];
            float expected = geluGradientReference( x ) * grad_out;
            float actual = host_input_grad_result.data()[ i ];
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
        if ( !isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" ) )
        {
            GTEST_SKIP() << "GeluOp not registered for CUDA FP32";
        }

        DeviceId cuda_id = Device::Cuda( 0 );
        auto gelu = std::make_shared<GeluCuda>( "gelu", GeluConfig(), cuda_id );

        std::vector<int64_t> shape = { 2, 3, 4 };
        auto cuda_device = gelu->getDeviceId();

        Tensor<TensorDataType::FP32, MR> input( cuda_device, shape );
        Tensor<TensorDataType::FP32, MR> output_grad( cuda_device, shape );

        DeviceId cpu_device{ DeviceType::Cpu, 0 };
        Tensor<TensorDataType::FP32, CpuMemoryResource> host_input( cpu_device, shape );
        Tensor<TensorDataType::FP32, CpuMemoryResource> host_output_grad( cpu_device, shape );

        for ( size_t i = 0; i < host_input.size(); ++i )
        {
            host_input.data()[ i ] = static_cast<float>( i ) / host_input.size() * 4.0f - 2.0f;
            host_output_grad.data()[ i ] = static_cast<float>( i + 1 ) * 0.1f;
        }

        copy( host_input, input );
        copy( host_output_grad, output_grad );

        // Build before enabling training to satisfy Component lifecycle contract.
        gelu->build( shape );
        gelu->setTraining( true );
        gelu->forward( input );

        auto& in_grad_device = gelu->backward( input, output_grad );
        auto host_input_grad = toHost<TensorDataType::FP32>( in_grad_device );

        const float tolerance = 1e-3f;

        for ( size_t i = 0; i < host_input.size(); ++i )
        {
            float x = host_input.data()[ i ];
            float grad_out = host_output_grad.data()[ i ];
            float expected = geluGradientReference( x ) * grad_out;
            float actual = host_input_grad.data()[ i ];
            float diff = std::abs( expected - actual );

            EXPECT_LT( diff, tolerance )
                << "Chain rule gradient mismatch at index " << i;
        }
    }

    TEST_F( GeluCudaTests, Backward_HandlesZeroOutputGradient )
    {
        if ( !isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" ) )
        {
            GTEST_SKIP() << "GeluOp not registered for CUDA FP32";
        }

        DeviceId cuda_id = Device::Cuda( 0 );
        auto gelu = std::make_shared<GeluCuda>( "gelu", GeluConfig(), cuda_id );

        std::vector<int64_t> shape = { 2, 3, 4 };
        auto cuda_device = gelu->getDeviceId();

        Tensor<TensorDataType::FP32, MR> input( cuda_device, shape );
        Tensor<TensorDataType::FP32, MR> output_grad( cuda_device, shape );

        DeviceId cpu_device{ DeviceType::Cpu, 0 };
        Tensor<TensorDataType::FP32, CpuMemoryResource> host_input( cpu_device, shape );
        Tensor<TensorDataType::FP32, CpuMemoryResource> host_output_grad( cpu_device, shape );

        for ( size_t i = 0; i < host_input.size(); ++i )
        {
            host_input.data()[ i ] = static_cast<float>( i ) / host_input.size() * 2.0f;
            host_output_grad.data()[ i ] = 0.0f;
        }

        copy( host_input, input );
        copy( host_output_grad, output_grad );

        // Build before enabling training to satisfy Component lifecycle contract.
        gelu->build( shape );
        gelu->setTraining( true );

        gelu->forward( input );

        auto& in_grad_device = gelu->backward( input, output_grad );
        auto host_input_grad = toHost<TensorDataType::FP32>( in_grad_device );

        for ( size_t i = 0; i < host_input_grad.size(); ++i )
        {
            EXPECT_FLOAT_EQ( host_input_grad.data()[ i ], 0.0f )
                << "Expected zero gradient at index " << i;
        }
    }

    TEST_F( GeluCudaTests, Backward_HandlesEdgeCaseInputs )
    {
        if ( !isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" ) )
        {
            GTEST_SKIP() << "GeluOp not registered for CUDA FP32";
        }

        DeviceId cuda_id = Device::Cuda( 0 );
        auto gelu = std::make_shared<GeluCuda>( "gelu", GeluConfig(), cuda_id );

        std::vector<int64_t> shape = { 1, 8 };
        auto cuda_device = gelu->getDeviceId();

        Tensor<TensorDataType::FP32, MR> input( cuda_device, shape );
        Tensor<TensorDataType::FP32, MR> output_grad( cuda_device, shape );

        DeviceId cpu_device{ DeviceType::Cpu, 0 };
        Tensor<TensorDataType::FP32, CpuMemoryResource> host_input( cpu_device, shape );
        Tensor<TensorDataType::FP32, CpuMemoryResource> host_output_grad( cpu_device, shape );

        std::vector<float> test_values = { -10.0f, -1.0f, -0.1f, 0.0f, 0.1f, 1.0f, 10.0f, 50.0f };

        for ( size_t i = 0; i < test_values.size(); ++i )
        {
            host_input.data()[ i ] = test_values[ i ];
            host_output_grad.data()[ i ] = 1.0f + i;
        }

        copy( host_input, input );
        copy( host_output_grad, output_grad );

        // Build before enabling training to satisfy Component lifecycle contract.
        gelu->build( shape );
        gelu->setTraining( true );

        EXPECT_NO_THROW( gelu->forward( input ) );

        EXPECT_NO_THROW( gelu->zeroGradients() );

        Tensor<TensorDataType::FP32, MR>* in_grad_ptr = nullptr;
        ASSERT_NO_THROW(
            {
                auto& ig = gelu->backward( input, output_grad );
                in_grad_ptr = &ig;
            }
        );

        ASSERT_NE( in_grad_ptr, nullptr );

        auto host_input_grad = toHost<TensorDataType::FP32>( *in_grad_ptr );

        for ( size_t i = 0; i < host_input_grad.size(); ++i )
        {
            EXPECT_FALSE( std::isnan( host_input_grad.data()[ i ] ) )
                << "NaN gradient at index " << i << " for input " << test_values[ i ];
            EXPECT_FALSE( std::isinf( host_input_grad.data()[ i ] ) )
                << "Inf gradient at index " << i << " for input " << test_values[ i ];
        }
    }

    // ========================================================================
    // CPU-CUDA Equivalence Tests
    // ========================================================================

    TEST_F( GeluCudaTests, CpuCuda_ForwardEquivalence_OrConstructorThrows )
    {
        bool cpu_registered = isOperationRegistered<DeviceType::Cpu, TensorDataType::FP32>( "GeluOp" );
        bool cuda_registered = isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" );

        if ( !cpu_registered || !cuda_registered )
        {
            if ( !cpu_registered )
            {
                DeviceId cpu_id = Device::Cpu();
                EXPECT_THROW(
                    GeluCpu( "gelu", GeluConfig(), cpu_id ),
                    std::runtime_error
                );
            }

            if ( !cuda_registered )
            {
                DeviceId cuda_id = Device::Cuda( 0 );
                EXPECT_THROW(
                    GeluCuda( "gelu", GeluConfig(), cuda_id ),
                    std::runtime_error
                );
            }

            return;
        }

        std::vector<int64_t> shape = { 2, 2, 4 };
        GeluConfig config;

        DeviceId cpu_id = Device::Cpu();
        DeviceId cuda_id = Device::Cuda( 0 );

        auto cpu_gelu = std::make_shared<GeluCpu>( "gelu", config, cpu_id );
        auto cuda_gelu = std::make_shared<GeluCuda>( "gelu", config, cuda_id );

        Tensor<TensorDataType::FP32, CpuMemoryResource> cpu_input( cpu_id, shape );

        for ( size_t i = 0; i < cpu_input.size(); ++i )
        {
            cpu_input.data()[ i ] = static_cast<float>( i ) / cpu_input.size() * 4.0f - 2.0f;
        }

        cpu_gelu->build( cpu_input.shape() );

        auto& cpu_out_tensor = cpu_gelu->forward( cpu_input );
        ASSERT_FALSE( &cpu_out_tensor == nullptr );

        Tensor<TensorDataType::FP32, MR> cuda_input( cuda_id, shape );
        // Copy the original CPU input to the CUDA input so both implementations
        // receive the same raw data for forward equivalence testing.
        copy( cpu_input, cuda_input );

        cuda_gelu->build( cuda_input.shape() );

        auto& cuda_out_tensor = cuda_gelu->forward( cuda_input );
        cuda_gelu->synchronize();

        ASSERT_FALSE( &cuda_out_tensor == nullptr );

        auto cuda_output_host = toHost<TensorDataType::FP32>( cuda_out_tensor );

        const float epsilon = 1e-3f;

        for ( size_t i = 0; i < cpu_out_tensor.size(); ++i )
        {
            float cpu_val = cpu_out_tensor.data()[ i ];
            float gpu_val = cuda_output_host.data()[ i ];
            float diff = std::abs( cpu_val - gpu_val );

            EXPECT_LT( diff, epsilon )
                << "Mismatch at index " << i
                << ": CPU=" << cpu_val
                << ", CUDA=" << gpu_val;
        }
    }

    // ========================================================================
    // Metadata Tests
    // ========================================================================

    TEST_F( GeluCudaTests, ToString_ContainsGeluOrConstructorThrows )
    {
        if ( !isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" ) )
        {
            DeviceId cuda_id = Device::Cuda( 0 );
            EXPECT_THROW(
                GeluCuda( "gelu", GeluConfig(), cuda_id ),
                std::runtime_error
            );
            return;
        }

        auto d = GeluCudaTestData::Create( batch_, seq_, chan_ );
        auto s = d.gelu_module->toString();

        EXPECT_FALSE( s.empty() );
        EXPECT_NE( s.find( "Gelu" ), std::string::npos );
    }

    TEST_F( GeluCudaTests, Construct_WithDeviceId_WorksOrThrows )
    {
        if ( !isOperationRegistered<DeviceType::Cuda, TensorDataType::FP32>( "GeluOp" ) )
        {
            DeviceId cuda_id = Device::Cuda( 0 );
            EXPECT_THROW(
                GeluCuda( "gelu", GeluConfig(), cuda_id ),
                std::runtime_error
            );
            return;
        }

        auto d = GeluCudaTestData::Create( batch_, seq_, chan_ );

        EXPECT_EQ( d.gelu_module->getDeviceId().type, DeviceType::Cuda );
    }
}