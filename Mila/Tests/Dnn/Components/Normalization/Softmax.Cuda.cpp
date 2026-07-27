/**
 * @file Softmax.Cuda.cpp
 * @brief Concrete-component tests for Softmax<DeviceType::Cuda, {FP32}>.
 *
 * CUDA companion to Softmax.Cpu.cpp, written as a precision sweep (TYPED_TEST) per
 * the methodology default. The supported-precision list holds FP32 only: the CUDA
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
#include <string>
#include <vector>

import Mila;
import Compute.ExecutionContext;

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
            for ( size_t i = 0; i < host.size(); ++i )
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

        for ( size_t i = 0; i < out.size(); ++i )
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
