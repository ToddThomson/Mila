/**
 * @file Softmax.Cuda.cpp
 * @brief Concrete-component tests for Softmax<DeviceType::Cuda, {FP32}>.
 *
 * Written as a precision sweep (TYPED_TEST) per the methodology default. The supported-precision list holds FP32 only: the CUDA
 * SoftmaxOp kernel dispatch is constrained to float/half (CudaSoftmaxOp.ixx /
 * Softmax.cuh: `float || half`), there is no BF16 path -- the Llama inference path
 * fuses its attention softmax inside the GQA kernel. Add a Bf16Precision tag to
 * SoftmaxPrecisions once a BF16 kernel exists and the suite re-runs for it.
 * Row-wise softmax over the trailing dimension:
 *   y_i = exp(x_i - max) / sum_j exp(x_j - max)
 *
 * Compiled only under MILA_ENABLE_CUDA; SetUp() skips if no device at runtime.
 * Re-greened from the pre-methodology version.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

import Mila;

namespace Mila::Tests::Dnn::Components::Normalization
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        constexpr int64_t kRows = 4;
        constexpr int64_t kChannels = 8;

        void referenceSoftmax( const float* X, int64_t rows, int64_t channels, std::vector<float>& Y )
        {
            Y.assign( static_cast<size_t>( rows * channels ), 0.0f );

            for ( int64_t r = 0; r < rows; ++r )
            {
                const float* x = X + r * channels;

                float maxv = -INFINITY;
                for ( int64_t i = 0; i < channels; ++i )
                {
                    maxv = std::max( maxv, x[ i ] );
                }

                double sum = 0.0;
                for ( int64_t i = 0; i < channels; ++i )
                {
                    sum += std::exp( static_cast<double>( x[ i ] - maxv ) );
                }

                for ( int64_t i = 0; i < channels; ++i )
                {
                    Y[ r * channels + i ] = static_cast<float>( std::exp( static_cast<double>( x[ i ] - maxv ) ) / sum );
                }
            }
        }

        struct Fp32Precision
        {
            static constexpr TensorDataType value = TensorDataType::FP32;
            static constexpr float atol = 1e-4f;
            static constexpr float rtol = 1e-4f;
            static constexpr const char* name = "Fp32";
        };

        // FP32 only: the CUDA SoftmaxOp kernel is float/half (no BF16). Add a
        // Bf16Precision tag here once a BF16 kernel exists -- the rest of the suite
        // re-runs unchanged.
        using SoftmaxPrecisions = ::testing::Types<Fp32Precision>;

        class PrecisionNames
        {
        public:
            template<typename TPrecisionTag>
            static std::string GetName( int )
            {
                return TPrecisionTag::name;
            }
        };
    }

    template<typename TPrecisionTag>
    class SoftmaxCudaTests : public ::testing::Test
    {
    protected:
        static constexpr TensorDataType P = TPrecisionTag::value;

        using SoftmaxType = Mila::Dnn::Softmax<DeviceType::Cuda, P>;
        using DeviceTensor = Tensor<P, CudaDeviceMemoryResource>;
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        static_assert( SoftmaxType::getDeviceType() == DeviceType::Cuda );
        static_assert( SoftmaxType::getPrecision() == P );

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

        std::unique_ptr<SoftmaxType> builtSoftmax( const shape_t& shape, RuntimeMode mode )
        {
            auto softmax = std::make_unique<SoftmaxType>( "softmax", SoftmaxConfig().withAxis( -1 ), Device::Cuda( 0 ) );
            softmax->build( BuildContext( shape, mode, false ) );

            return softmax;
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

    TYPED_TEST_SUITE( SoftmaxCudaTests, SoftmaxPrecisions, PrecisionNames );

    // ====================================================================
    // A. Construction
    // ====================================================================

    TYPED_TEST( SoftmaxCudaTests, Construct_StandaloneSucceeds )
    {
        typename TestFixture::SoftmaxType softmax( "softmax", SoftmaxConfig(), Device::Cuda( 0 ) );

        EXPECT_EQ( softmax.getDeviceId().type, DeviceType::Cuda );
    }

    // ====================================================================
    // B. Build Lifecycle (preconditions)
    // ====================================================================

    TYPED_TEST( SoftmaxCudaTests, Forward_ThrowsBeforeBuild )
    {
        typename TestFixture::SoftmaxType softmax( "softmax", SoftmaxConfig(), Device::Cuda( 0 ) );
        typename TestFixture::DeviceTensor input( Device::Cuda( 0 ), shape_t{ kRows, kChannels } );
        typename TestFixture::DeviceTensor output( Device::Cuda( 0 ), shape_t{ kRows, kChannels } );

        EXPECT_THROW( softmax.forward( input, output ), std::runtime_error );
    }

    TYPED_TEST( SoftmaxCudaTests, Build_ThrowsOnAxisOutOfBounds )
    {
        typename TestFixture::SoftmaxType softmax( "softmax", SoftmaxConfig().withAxis( 5 ), Device::Cuda( 0 ) );

        // Axis 5 is out of bounds for a rank-2 input.
        EXPECT_THROW(
            softmax.build( BuildContext( shape_t{ kRows, kChannels }, RuntimeMode::Inference, false ) ),
            std::invalid_argument );
    }

    // ====================================================================
    // E. Forward (numeric vs reference)
    // ====================================================================

    TYPED_TEST( SoftmaxCudaTests, Forward_MatchesReference )
    {
        const shape_t shape{ kRows, kChannels };
        auto softmax = this->builtSoftmax( shape, RuntimeMode::Inference );

        auto host_in = this->spreadHost( shape );
        auto device_in = this->toDevice( host_in );
        typename TestFixture::DeviceTensor device_out( Device::Cuda( 0 ), shape );

        softmax->forward( device_in, device_out );
        softmax->synchronize();

        auto out = this->toFloat( device_out );
        auto in = this->toFloat( device_in );

        std::vector<float> expected;
        referenceSoftmax( in.data(), kRows, kChannels, expected );

        ASSERT_EQ( out.size(), expected.size() );

        for ( dim_t i = 0; i < out.size(); ++i )
        {
            const float tolerance = TypeParam::atol + TypeParam::rtol * std::fabs( expected[ i ] );
            EXPECT_NEAR( out.data()[ i ], expected[ i ], tolerance ) << "forward mismatch at index " << i;
        }
    }

    // ====================================================================
    // D. Runtime mode — backward requires training build
    // ====================================================================

    TYPED_TEST( SoftmaxCudaTests, Backward_ThrowsWhenNotTraining )
    {
        const shape_t shape{ kRows, kChannels };
        auto softmax = this->builtSoftmax( shape, RuntimeMode::Inference );

        auto device_in = this->toDevice( this->spreadHost( shape ) );
        typename TestFixture::DeviceTensor device_grad( Device::Cuda( 0 ), shape );
        typename TestFixture::DeviceTensor device_in_grad( Device::Cuda( 0 ), shape );

        EXPECT_THROW( softmax->backward( device_in, device_grad, device_in_grad ), std::runtime_error );
    }

    // ====================================================================
    // F. Backward (analytic reference)
    // ====================================================================

    TYPED_TEST( SoftmaxCudaTests, Backward_MatchesReferenceGradient )
    {
        const shape_t shape{ kRows, kChannels };
        auto softmax = this->builtSoftmax( shape, RuntimeMode::Training );

        auto device_in = this->toDevice( this->spreadHost( shape ) );
        typename TestFixture::DeviceTensor device_out( Device::Cuda( 0 ), shape );

        // Non-uniform, so a kernel that ignored its output gradient fails.
        typename TestFixture::HostFp32 host_grad( Device::Cpu(), shape );
        typename TestFixture::HostFp32 host_zero( Device::Cpu(), shape );

        for ( dim_t i = 0; i < host_grad.size(); ++i )
        {
            host_grad.data()[ i ] = 0.1f * static_cast<float>( ( i % 7 ) + 1 );
            host_zero.data()[ i ] = 0.0f;
        }

        auto device_grad = this->toDevice( host_grad );

        // Backward consumes the forward output Y and accumulates into the input gradient.
        auto device_in_grad = this->toDevice( host_zero );

        softmax->forward( device_in, device_out );
        softmax->backward( device_out, device_grad, device_in_grad );
        softmax->synchronize();

        auto input_grad = this->toFloat( device_in_grad );
        auto in = this->toFloat( device_in );

        std::vector<float> y;
        referenceSoftmax( in.data(), kRows, kChannels, y );

        // dX_i = Y_i * ( dY_i - sum_j Y_j dY_j ), per row.
        for ( int64_t r = 0; r < kRows; ++r )
        {
            double dot = 0.0;

            for ( int64_t c = 0; c < kChannels; ++c )
            {
                dot += static_cast<double>( y[ r * kChannels + c ] ) * host_grad.data()[ r * kChannels + c ];
            }

            for ( int64_t c = 0; c < kChannels; ++c )
            {
                const double expected = y[ r * kChannels + c ] * ( host_grad.data()[ r * kChannels + c ] - dot );
                const float tolerance = TypeParam::atol + TypeParam::rtol * static_cast<float>( std::fabs( expected ) );

                EXPECT_NEAR( input_grad.data()[ r * kChannels + c ], static_cast<float>( expected ), tolerance )
                    << "dX mismatch at row " << r << " channel " << c;
            }
        }
    }

    // A middle axis takes the strided kernels rather than the row kernels.
    TYPED_TEST( SoftmaxCudaTests, Backward_AlongAMiddleAxisMatchesReferenceGradient )
    {
        constexpr int64_t kOuter = 2;
        constexpr int64_t kInner = 3;
        const shape_t shape{ kOuter, kChannels, kInner };

        typename TestFixture::SoftmaxType softmax( "softmax", SoftmaxConfig().withAxis( 1 ), Device::Cuda( 0 ) );
        softmax.build( BuildContext( shape, RuntimeMode::Training, false ) );

        typename TestFixture::HostFp32 host_grad( Device::Cpu(), shape );

        for ( dim_t i = 0; i < host_grad.size(); ++i )
        {
            host_grad.data()[ i ] = 0.1f * static_cast<float>( ( i % 7 ) + 1 );
        }

        auto device_in = this->toDevice( this->spreadHost( shape ) );
        auto device_grad = this->toDevice( host_grad );
        typename TestFixture::DeviceTensor device_out( Device::Cuda( 0 ), shape );
        typename TestFixture::DeviceTensor device_in_grad( Device::Cuda( 0 ), shape );

        softmax.forward( device_in, device_out );
        softmax.backward( device_out, device_grad, device_in_grad );
        softmax.synchronize();

        auto in = this->toFloat( device_in );
        auto input_grad = this->toFloat( device_in_grad );

        for ( int64_t o = 0; o < kOuter; ++o )
        {
            for ( int64_t k = 0; k < kInner; ++k )
            {
                auto at = [&]( int64_t c ) { return ( o * kChannels + c ) * kInner + k; };

                std::vector<float> column( kChannels );

                for ( int64_t c = 0; c < kChannels; ++c )
                {
                    column[ c ] = in.data()[ at( c ) ];
                }

                std::vector<float> y;
                referenceSoftmax( column.data(), 1, kChannels, y );

                double dot = 0.0;

                for ( int64_t c = 0; c < kChannels; ++c )
                {
                    dot += static_cast<double>( y[ c ] ) * host_grad.data()[ at( c ) ];
                }

                for ( int64_t c = 0; c < kChannels; ++c )
                {
                    const double expected = y[ c ] * ( host_grad.data()[ at( c ) ] - dot );
                    const float tolerance = TypeParam::atol + TypeParam::rtol * static_cast<float>( std::fabs( expected ) );

                    EXPECT_NEAR( input_grad.data()[ at( c ) ], static_cast<float>( expected ), tolerance )
                        << "dX mismatch at outer " << o << " channel " << c << " inner " << k;
                }
            }
        }
    }

    // ====================================================================
    // G. Parameters (stateless) & J. Type
    // ====================================================================

    TYPED_TEST( SoftmaxCudaTests, Parameters_AreEmpty )
    {
        auto softmax = this->builtSoftmax( shape_t{ kRows, kChannels }, RuntimeMode::Inference );

        EXPECT_EQ( softmax->parameterCount(), 0 );
        EXPECT_TRUE( softmax->getParameters().empty() );
    }

    TYPED_TEST( SoftmaxCudaTests, GetType_IsSoftmax )
    {
        typename TestFixture::SoftmaxType softmax( "softmax", SoftmaxConfig(), Device::Cuda( 0 ) );

        EXPECT_EQ( softmax.getType(), ComponentType::Softmax );
    }
}
