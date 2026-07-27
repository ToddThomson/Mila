/**
 * @file RmsNorm.Cuda.cpp
 * @brief Concrete-component tests for RmsNorm<DeviceType::Cuda, {FP32, BF16}>.
 *
 * RmsNorm is CUDA-only (no CPU RmsNormOp), so there is no RmsNorm.Cpu.cpp -- this
 * is the whole concrete surface, a TYPED_TEST sweep over FP32 and BF16 (see
 * Linear.Cuda.cpp for the precision-sweep reference).
 *
 * RMS normalization over the trailing normalized dimension:
 *   rstd = 1 / sqrt( mean(x^2) + eps )
 *   y_i  = x_i * rstd * weight_i + bias_i
 *
 * Compiled only under MILA_ENABLE_CUDA; SetUp() skips if no device at runtime.
 *
 * Numeric strategy matches Linear.Cuda.cpp: deterministic weight/bias/input
 * uploaded with conversion to the device precision, read back to float so the
 * reference sees the precision-rounded values the kernel consumed.
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

namespace Mila::Tests::Dnn::Components::Normalization::RmsNorm
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        constexpr int64_t kChannels = 16;
        constexpr float kEpsilon = 1e-5f;

        float weightValue( int64_t i )
        {
            return 0.5f + 0.1f * static_cast<float>( ( i % 5 ) - 2 );
        }

        float biasValue( int64_t i )
        {
            return 0.05f * static_cast<float>( ( i % 7 ) - 3 );
        }

        // y_i = x_i * rstd * w_i + b_i, rstd = 1/sqrt(mean(x^2)+eps), per outer row.
        void referenceForward(
            const float* X, const float* W, const float* B,
            int64_t outer, int64_t channels, float eps, std::vector<float>& Y )
        {
            Y.assign( static_cast<size_t>( outer * channels ), 0.0f );

            for ( int64_t r = 0; r < outer; ++r )
            {
                const float* x = X + r * channels;

                double sumsq = 0.0;
                for ( int64_t i = 0; i < channels; ++i )
                {
                    sumsq += static_cast<double>( x[ i ] ) * x[ i ];
                }
                const double rstd = 1.0 / std::sqrt( sumsq / channels + eps );

                for ( int64_t i = 0; i < channels; ++i )
                {
                    Y[ r * channels + i ] = static_cast<float>( x[ i ] * rstd * W[ i ] + B[ i ] );
                }
            }
        }

        struct Fp32Precision
        {
            static constexpr TensorDataType value = TensorDataType::FP32;
            static constexpr float forward_atol = 2e-3f;
            static constexpr float forward_rtol = 1e-3f;
            static constexpr const char* name = "Fp32";
        };

        struct Bf16Precision
        {
            static constexpr TensorDataType value = TensorDataType::BF16;
            static constexpr float forward_atol = 5e-2f;
            static constexpr float forward_rtol = 5e-2f;
            static constexpr const char* name = "Bf16";
        };

        using RmsNormPrecisions = ::testing::Types<Fp32Precision, Bf16Precision>;

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
    class RmsNormCudaTests : public ::testing::Test
    {
    protected:
        static constexpr TensorDataType P = TPrecisionTag::value;

        using RmsNormType = Mila::Dnn::RmsNorm<DeviceType::Cuda, P>;
        using DeviceTensor = Tensor<P, CudaDeviceMemoryResource>;
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        static_assert( RmsNormType::getDeviceType() == DeviceType::Cuda );
        static_assert( RmsNormType::getPrecision() == P );

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

        RmsNormConfig config()
        {
            return RmsNormConfig( shape_t{ kChannels } ).withEpsilon( kEpsilon );
        }

        std::unique_ptr<RmsNormType> builtRmsNorm( const shape_t& shape, RuntimeMode mode )
        {
            auto norm = std::make_unique<RmsNormType>( "rmsnorm", config(), Device::Cuda( 0 ) );
            norm->build( BuildContext( shape, mode, false ) );

            return norm;
        }

        void setKnownParameters( RmsNormType& norm )
        {
            auto params = norm.getParameters();

            HostFp32 host_weight( Device::Cpu(), shape_t{ kChannels } );
            HostFp32 host_bias( Device::Cpu(), shape_t{ kChannels } );
            for ( int64_t i = 0; i < kChannels; ++i )
            {
                host_weight.data()[ i ] = weightValue( i );
                host_bias.data()[ i ] = biasValue( i );
            }

            copy( host_weight, *static_cast<DeviceTensor*>( params[ 0 ] ), cuda_context_.get() );
            copy( host_bias, *static_cast<DeviceTensor*>( params[ 1 ] ), cuda_context_.get() );
            cuda_context_->synchronize();
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

    TYPED_TEST_SUITE( RmsNormCudaTests, RmsNormPrecisions, PrecisionNames );

    // ====================================================================
    // A. Construction
    // ====================================================================

    TYPED_TEST( RmsNormCudaTests, Construct_StandaloneSucceeds )
    {
        typename TestFixture::RmsNormType norm( "rmsnorm", this->config(), Device::Cuda( 0 ) );

        EXPECT_EQ( norm.getDeviceId().type, DeviceType::Cuda );
    }

    // ====================================================================
    // B. Build Lifecycle (preconditions)
    // ====================================================================

    TYPED_TEST( RmsNormCudaTests, Forward_ThrowsBeforeBuild )
    {
        typename TestFixture::RmsNormType norm( "rmsnorm", this->config(), Device::Cuda( 0 ) );
        typename TestFixture::DeviceTensor input( Device::Cuda( 0 ), shape_t{ 2, 3, kChannels } );

        EXPECT_THROW( norm.forward( input ), std::runtime_error );
    }

    TYPED_TEST( RmsNormCudaTests, Build_ThrowsOnTrailingDimMismatch )
    {
        typename TestFixture::RmsNormType norm( "rmsnorm", this->config(), Device::Cuda( 0 ) );

        // Trailing dim mismatched against normalized_shape must fail validation.
        EXPECT_THROW(
            norm.build( BuildContext( shape_t{ 2, 3, kChannels + 1 }, RuntimeMode::Inference, false ) ),
            std::invalid_argument );
    }

    // ====================================================================
    // E. Forward (numeric vs reference)
    // ====================================================================

    TYPED_TEST( RmsNormCudaTests, Forward_MatchesReference )
    {
        const shape_t shape{ 2, 3, kChannels };

        auto norm = this->builtRmsNorm( shape, RuntimeMode::Inference );
        this->setKnownParameters( *norm );

        auto host_in = this->spreadHost( shape );
        auto device_in = this->toDevice( host_in );

        auto& device_out = norm->forward( device_in );
        norm->synchronize();

        auto out = this->toFloat( device_out );
        auto in = this->toFloat( device_in );

        auto params = norm->getParameters();
        auto weight = this->toFloat( *static_cast<typename TestFixture::DeviceTensor*>( params[ 0 ] ) );
        auto bias = this->toFloat( *static_cast<typename TestFixture::DeviceTensor*>( params[ 1 ] ) );

        const int64_t outer = 2 * 3;
        ASSERT_EQ( out.shape(), shape );

        std::vector<float> expected;
        referenceForward( in.data(), weight.data(), bias.data(), outer, kChannels, kEpsilon, expected );

        ASSERT_EQ( out.size(), expected.size() );

        for ( dim_t i = 0; i < out.size(); ++i )
        {
            const float tolerance = TypeParam::forward_atol + TypeParam::forward_rtol * std::fabs( expected[ i ] );

            EXPECT_NEAR( out.data()[ i ], expected[ i ], tolerance )
                << "forward mismatch at index " << i;
        }
    }

    // ====================================================================
    // D. Runtime mode — backward requires training build
    // ====================================================================

    TYPED_TEST( RmsNormCudaTests, Backward_ThrowsWhenNotTraining )
    {
        const shape_t shape{ 2, 3, kChannels };
        auto norm = this->builtRmsNorm( shape, RuntimeMode::Inference );
        this->setKnownParameters( *norm );

        auto device_in = this->toDevice( this->spreadHost( shape ) );
        auto device_grad = this->toDevice( this->spreadHost( shape ) );

        norm->forward( device_in );

        EXPECT_THROW( norm->backward( device_in, device_grad ), std::runtime_error );
    }

    // ====================================================================
    // G. Parameters & Gradients
    // ====================================================================

    TYPED_TEST( RmsNormCudaTests, Parameters_WeightAndBias )
    {
        auto norm = this->builtRmsNorm( shape_t{ 2, 3, kChannels }, RuntimeMode::Inference );

        EXPECT_EQ( norm->getParameters().size(), 2u );
        EXPECT_EQ( norm->parameterCount(), static_cast<size_t>( 2 * kChannels ) );
    }

    TYPED_TEST( RmsNormCudaTests, Gradients_PresentOnlyForTrainingBuild )
    {
        auto inference = this->builtRmsNorm( shape_t{ 2, 3, kChannels }, RuntimeMode::Inference );
        EXPECT_TRUE( inference->getGradients().empty() );

        auto training = this->builtRmsNorm( shape_t{ 2, 3, kChannels }, RuntimeMode::Training );
        EXPECT_EQ( training->getGradients().size(), 2u );
    }

    // ====================================================================
    // J. Type identity
    // ====================================================================

    TYPED_TEST( RmsNormCudaTests, GetType_IsRmsNorm )
    {
        typename TestFixture::RmsNormType norm( "rmsnorm", this->config(), Device::Cuda( 0 ) );

        EXPECT_EQ( norm.getType(), ComponentType::RmsNorm );
    }
}
