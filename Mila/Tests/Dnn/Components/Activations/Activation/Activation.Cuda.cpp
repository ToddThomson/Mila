/**
 * @file Activation.Cuda.cpp
 * @brief Concrete-component tests for Activation<DeviceType::Cuda, {FP32, BF16}, TFn>.
 *
 * CUDA companion to Activation.Cpu.cpp. Two complementary sweeps:
 *  - A precision sweep (FP32, BF16) over SiLU exercising the functor-templated
 *    kernel through both native element types (Specifications/Testing.md, mirroring
 *    Swiglu.Cuda.cpp: device input is read back to float so the reference sees the
 *    precision-rounded values the kernel consumed).
 *  - An FP32 forward pass over several functions confirming the compile-time
 *    function selection reaches distinct specialized kernels.
 *
 * Compiled only under MILA_ENABLE_CUDA; SetUp() skips if no device is present.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>

import Mila;

namespace Mila::Tests::Dnn::Components::Activations::Activation
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        float siluRef( float x ) { return x / (1.0f + std::exp( -x )); }
        float siluDf( float x )
        {
            float s = 1.0f / (1.0f + std::exp( -x ));
            return s * (1.0f + x * (1.0f - s));
        }

        float geluRef( float x )
        {
            constexpr float k = 0.7978845608f;
            constexpr float c = 0.044715f;
            return 0.5f * x * (1.0f + std::tanh( k * (x + c * x * x * x) ));
        }

        float reluRef( float x ) { return x > 0.0f ? x : 0.0f; }
        float sigmoidRef( float x ) { return 1.0f / (1.0f + std::exp( -x )); }

        struct Fp32Precision
        {
            static constexpr TensorDataType value = TensorDataType::FP32;
            static constexpr float forward_tol = 2e-3f;
            static constexpr float backward_tol = 2e-3f;
            static constexpr const char* name = "Fp32";
        };

        struct Bf16Precision
        {
            static constexpr TensorDataType value = TensorDataType::BF16;
            static constexpr float forward_tol = 5e-2f;
            static constexpr float backward_tol = 5e-2f;
            static constexpr const char* name = "Bf16";
        };

        using ActivationPrecisions = ::testing::Types<Fp32Precision, Bf16Precision>;

        class PrecisionNames
        {
        public:
            template<typename TPrecisionTag>
            static std::string GetName( int ) { return TPrecisionTag::name; }
        };
    }

    template<typename TPrecisionTag>
    class ActivationCudaTests : public ::testing::Test
    {
    protected:
        static constexpr TensorDataType P = TPrecisionTag::value;

        // SiLU is the representative function for the precision sweep.
        using ActivationType_ = Mila::Dnn::Activation<DeviceType::Cuda, P, Mila::Dnn::ActivationType::Silu>;
        using DeviceTensor = Tensor<P, CudaDeviceMemoryResource>;
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        static_assert( ActivationType_::getDeviceType() == DeviceType::Cuda );
        static_assert( ActivationType_::getPrecision() == P );

        void SetUp() override
        {
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

        std::unique_ptr<ActivationType_> built( const shape_t& shape, RuntimeMode mode )
        {
            auto act = std::make_unique<ActivationType_>(
                "act", ActivationConfig( Mila::Dnn::ActivationType::Silu ), Device::Cuda( 0 ) );
            act->build( BuildContext( shape, mode ) );

            return act;
        }

        HostFp32 spreadHost( const shape_t& shape )
        {
            HostFp32 host( Device::Cpu(), shape );

            for ( dim_t i = 0; i < host.size(); ++i )
            {
                host.data()[ i ] = static_cast<float>( i ) / host.size() * 4.0f - 2.0f;
            }

            return host;
        }

        DeviceTensor toDevice( const HostFp32& host )
        {
            DeviceTensor device( Device::Cuda( 0 ), host.shape() );
            copy( host, device, cuda_context_.get() );
            cuda_context_->synchronize();

            return device;
        }

        HostFp32 toFloat( const DeviceTensor& device )
        {
            auto host = toHost<TensorDataType::FP32>( device, cuda_context_.get() );
            cuda_context_->synchronize();

            return host;
        }

        std::unique_ptr<IExecutionContext> cuda_context_;
    };

    TYPED_TEST_SUITE( ActivationCudaTests, ActivationPrecisions, PrecisionNames );

    TYPED_TEST( ActivationCudaTests, Construct_StandaloneSucceeds )
    {
        typename TestFixture::ActivationType_ act(
            "act", ActivationConfig( Mila::Dnn::ActivationType::Silu ), Device::Cuda( 0 ) );

        EXPECT_EQ( act.getDeviceId().type, DeviceType::Cuda );
        EXPECT_EQ( act.getType(), ComponentType::Activation );
    }

    TYPED_TEST( ActivationCudaTests, Forward_MatchesReference )
    {
        const shape_t shape{ 2, 3, 8 };

        auto act = this->built( shape, RuntimeMode::Inference );

        auto host_in = this->spreadHost( shape );
        auto device_in = this->toDevice( host_in );

        // Reference sees the precision-rounded values the kernel actually consumed.
        auto rounded_in = this->toFloat( device_in );

        auto& device_out = act->forward( device_in );
        act->synchronize();

        auto host_out = this->toFloat( device_out );

        ASSERT_EQ( host_out.size(), host_in.size() );

        for ( dim_t i = 0; i < host_out.size(); ++i )
        {
            const float expected = siluRef( rounded_in.data()[ i ] );

            EXPECT_NEAR( host_out.data()[ i ], expected, TypeParam::forward_tol )
                << "forward mismatch at index " << i;
        }
    }

    TYPED_TEST( ActivationCudaTests, Backward_MatchesReference )
    {
        const shape_t shape{ 2, 3, 8 };

        auto act = this->built( shape, RuntimeMode::Training );

        auto host_in = this->spreadHost( shape );

        typename TestFixture::HostFp32 host_grad( Device::Cpu(), shape );
        for ( dim_t i = 0; i < host_grad.size(); ++i )
        {
            host_grad.data()[ i ] = 1.0f;
        }

        auto device_in = this->toDevice( host_in );
        auto device_grad = this->toDevice( host_grad );

        auto rounded_in = this->toFloat( device_in );
        auto rounded_grad = this->toFloat( device_grad );

        act->forward( device_in );
        auto& device_in_grad = act->backward( device_in, device_grad );
        act->synchronize();

        auto host_in_grad = this->toFloat( device_in_grad );

        ASSERT_EQ( host_in_grad.size(), host_in.size() );

        for ( dim_t i = 0; i < host_in_grad.size(); ++i )
        {
            const float expected = siluDf( rounded_in.data()[ i ] ) * rounded_grad.data()[ i ];

            EXPECT_NEAR( host_in_grad.data()[ i ], expected, TypeParam::backward_tol )
                << "backward mismatch at index " << i;
        }
    }

    // ====================================================================
    // FP32 forward across several functions -- confirms the compile-time function
    // selection reaches distinct specialized kernels.
    // ====================================================================

    class ActivationCudaFp32FunctorTests : public ::testing::Test
    {
    protected:
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        using DeviceFp32 = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>;

        void SetUp() override
        {
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

        template<Mila::Dnn::ActivationType TFn, typename TRef>
        void checkForward( TRef reference )
        {
            const shape_t shape{ 2, 3, 4 };

            HostFp32 host_in( Device::Cpu(), shape );
            for ( dim_t i = 0; i < host_in.size(); ++i )
            {
                host_in.data()[ i ] = static_cast<float>( i ) / host_in.size() * 4.0f - 2.0f;
            }

            DeviceFp32 device_in( Device::Cuda( 0 ), shape );
            copy( host_in, device_in, cuda_context_.get() );
            cuda_context_->synchronize();

            Mila::Dnn::Activation<DeviceType::Cuda, TensorDataType::FP32, TFn> act(
                "act", ActivationConfig( TFn ), Device::Cuda( 0 ) );
            act.build( BuildContext( shape, RuntimeMode::Inference ) );

            auto& device_out = act.forward( device_in );
            act.synchronize();

            auto host_out = toHost<TensorDataType::FP32>( device_out, cuda_context_.get() );

            for ( dim_t i = 0; i < host_out.size(); ++i )
            {
                EXPECT_NEAR( host_out.data()[ i ], reference( host_in.data()[ i ] ), 1e-4f )
                    << "forward mismatch at index " << i;
            }
        }

        std::unique_ptr<IExecutionContext> cuda_context_;
    };

    TEST_F( ActivationCudaFp32FunctorTests, Gelu_MatchesReference )
    {
        checkForward<Mila::Dnn::ActivationType::Gelu>( geluRef );
    }

    TEST_F( ActivationCudaFp32FunctorTests, Relu_MatchesReference )
    {
        checkForward<Mila::Dnn::ActivationType::Relu>( reluRef );
    }

    TEST_F( ActivationCudaFp32FunctorTests, Sigmoid_MatchesReference )
    {
        checkForward<Mila::Dnn::ActivationType::Sigmoid>( sigmoidRef );
    }
}
