/**
 * @file Gelu.Cuda.cpp
 * @brief Concrete-component tests for Gelu<DeviceType::Cuda, FP32>.
 *
 * The CUDA companion to Gelu.Cpu.cpp (see Specifications/Testing.md). CUDA GeluOp
 * has a single working precision today -- the kernel dispatch (cuda_gelu_impl)
 * supports only float/half, so FP32 is the one precision the component can
 * instantiate -- so by the methodology this file is explicit (pragmatic), not
 * precision-parameterized. The TYPED_TEST mechanism appears first on a component
 * that genuinely supports more than one CUDA precision.
 *
 * Compiled only under MILA_ENABLE_CUDA (the Tests/CMakeLists.txt CUDA block), so
 * no #ifdef. CUDA may still be absent at runtime; SetUp() skips if no device.
 *
 * Numerics stage inputs host->device (copy) and read outputs device->host
 * (toHost). FP32 is exact host-side, so the reference is taken from the original
 * host input. Base-contract behavior is covered once in Core/Component.cpp and is
 * not retested here.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Components::Activations::Gelu
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        using GeluCuda = Mila::Dnn::Gelu<DeviceType::Cuda, TensorDataType::FP32>;
        using DeviceTensor = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>;
        using HostTensor = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        // GELU tanh-approximation reference, computed independently in float.
        constexpr float kSqrt2OverPi = 0.7978845608f;
        constexpr float kGeluCoeff = 0.044715f;

        float geluReference( float x )
        {
            const float x_cubed = x * x * x;

            return 0.5f * x * (1.0f + std::tanh( kSqrt2OverPi * (x + kGeluCoeff * x_cubed) ));
        }

        float geluGradientReference( float x )
        {
            const float x_squared = x * x;
            const float arg = kSqrt2OverPi * (x + kGeluCoeff * x * x_squared);
            const float tanh_value = std::tanh( arg );
            const float sech_squared = 1.0f - tanh_value * tanh_value;
            const float d_arg = kSqrt2OverPi * (1.0f + 3.0f * kGeluCoeff * x_squared);

            return 0.5f * (1.0f + tanh_value) + 0.5f * x * sech_squared * d_arg;
        }

        static_assert( GeluCuda::getDeviceType() == DeviceType::Cuda );
        static_assert( GeluCuda::getPrecision() == TensorDataType::FP32 );
    }

    class GeluCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            cpu_context_ = createExecutionContext( Device::Cpu() );

            try
            {
                cuda_context_ = createExecutionContext( Device::Cuda( 0 ) );
            }
            catch ( const std::exception& )
            {
                cuda_context_ = nullptr;
            }

            if ( !cuda_context_ )
            {
                GTEST_SKIP() << "CUDA device not available";
            }
        }

        HostTensor makeSpreadHost( const shape_t& shape )
        {
            HostTensor host( cpu_context_->getDeviceId(), shape );
            auto* data = host.data();

            for ( size_t i = 0; i < host.size(); ++i )
            {
                data[ i ] = static_cast<float>( i ) / host.size() * 4.0f - 2.0f;
            }

            return host;
        }

        DeviceTensor toDevice( const HostTensor& host, const shape_t& shape )
        {
            DeviceTensor device( cuda_context_->getDeviceId(), shape );
            copy( host, device, cuda_context_.get() );
            cuda_context_->synchronize();

            return device;
        }

        std::unique_ptr<IExecutionContext> cpu_context_;
        std::unique_ptr<IExecutionContext> cuda_context_;
    };

    // ====================================================================
    // A. Construction
    // ====================================================================

    TEST_F( GeluCudaTests, Construct_StandaloneSucceeds )
    {
        GeluCuda gelu( "gelu", GeluConfig(), Device::Cuda( 0 ) );

        EXPECT_EQ( gelu.getApproximationMethod(), ApproximationMethod::Tanh );
        EXPECT_EQ( gelu.getDeviceId().type, DeviceType::Cuda );
    }

    // ====================================================================
    // E. Forward (numeric vs reference)
    // ====================================================================

    TEST_F( GeluCudaTests, Forward_MatchesReference )
    {
        const shape_t shape{ 2, 3, 4 };

        auto host_in = makeSpreadHost( shape );
        auto device_in = toDevice( host_in, shape );

        GeluCuda gelu( "gelu", GeluConfig(), Device::Cuda( 0 ) );
        gelu.build( BuildContext( shape, RuntimeMode::Inference ) );

        auto& device_out = gelu.forward( device_in );
        gelu.synchronize();

        auto host_out = toHost<TensorDataType::FP32>( device_out, cuda_context_.get() );

        ASSERT_EQ( host_out.size(), host_in.size() );

        constexpr float tolerance = 1e-4f;

        for ( size_t i = 0; i < host_out.size(); ++i )
        {
            const float expected = geluReference( host_in.data()[ i ] );

            EXPECT_NEAR( host_out.data()[ i ], expected, tolerance )
                << "forward mismatch at index " << i << " input=" << host_in.data()[ i ];
        }
    }

    // ====================================================================
    // F. Backward (numeric vs analytic gradient)
    // ====================================================================

    TEST_F( GeluCudaTests, Backward_MatchesGradientReference )
    {
        const shape_t shape{ 2, 3, 4 };

        auto host_in = makeSpreadHost( shape );

        HostTensor host_grad( cpu_context_->getDeviceId(), shape );
        for ( size_t i = 0; i < host_grad.size(); ++i )
        {
            host_grad.data()[ i ] = 1.0f;
        }

        auto device_in = toDevice( host_in, shape );
        auto device_grad = toDevice( host_grad, shape );

        // Training build allocates the input-gradient buffer backward needs.
        GeluCuda gelu( "gelu", GeluConfig(), Device::Cuda( 0 ) );
        gelu.build( BuildContext( shape, RuntimeMode::Training ) );

        gelu.forward( device_in );
        auto& device_in_grad = gelu.backward( device_in, device_grad );
        gelu.synchronize();

        auto host_in_grad = toHost<TensorDataType::FP32>( device_in_grad, cuda_context_.get() );

        ASSERT_EQ( host_in_grad.size(), host_in.size() );

        constexpr float tolerance = 1e-3f;

        for ( size_t i = 0; i < host_in_grad.size(); ++i )
        {
            const float expected = geluGradientReference( host_in.data()[ i ] ) * host_grad.data()[ i ];

            EXPECT_NEAR( host_in_grad.data()[ i ], expected, tolerance )
                << "backward mismatch at index " << i << " input=" << host_in.data()[ i ];
        }
    }

    // ====================================================================
    // G. Parameters & Gradients (Gelu is stateless)
    // ====================================================================

    TEST_F( GeluCudaTests, Parameters_AreEmpty )
    {
        GeluCuda gelu( "gelu", GeluConfig(), Device::Cuda( 0 ) );
        gelu.build( BuildContext( shape_t{ 2, 4 }, RuntimeMode::Inference ) );

        EXPECT_EQ( gelu.parameterCount(), 0u );
        EXPECT_TRUE( gelu.getParameters().empty() );
        EXPECT_TRUE( gelu.getGradients().empty() );
    }

    // ====================================================================
    // J. Type identity
    // ====================================================================

    TEST_F( GeluCudaTests, GetType_IsGelu )
    {
        GeluCuda gelu( "gelu", GeluConfig(), Device::Cuda( 0 ) );

        EXPECT_EQ( gelu.getType(), ComponentType::Gelu );
    }
}
