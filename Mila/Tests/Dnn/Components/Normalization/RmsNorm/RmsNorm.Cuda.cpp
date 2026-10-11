/**
 * @file RmsNorm.Cuda.cpp
 * @brief Concrete-component tests for RmsNorm<DeviceType::Cuda, {FP32, BF16}>.
 *
 * A TYPED_TEST sweep over FP32 and BF16 (see Linear.Cuda.cpp for the precision-sweep
 * reference), and FP32 tests of bias, unit offset, groups and backward against a host reference.
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
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "Common/GradientCheck.h"

import Mila;

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

        constexpr int64_t kGroupSize = 8;
        constexpr int64_t kGroupedWidth = 4 * kGroupSize;

        // Qwen4ExpTextRMSNorm( 32, group_size = 8, eps = 1e-6 ) from transformers 5.16.0, weight
        // 0.03 * j - 0.4, over the two rows x[ i ] = sin( 0.37 * i + 0.1 ).
        constexpr float kQwen4GroupedReference[ 2 * kGroupedWidth ] = {
            8.23330432e-02f, 3.92172098e-01f, 6.75521374e-01f, 8.87347519e-01f, 9.89603162e-01f, 9.57646251e-01f,
            7.85036683e-01f, 4.85865235e-01f, 9.46148112e-02f, -3.41978520e-01f, -7.61033535e-01f, -1.10080373e+00f,
            -1.30706310e+00f, -1.34156132e+00f, -1.18859851e+00f, -8.58643770e-01f, -3.91610205e-01f, 1.64938375e-01f,
            7.29231596e-01f, 1.22169447e+00f, 1.56883752e+00f, 1.71412814e+00f, 1.62693512e+00f, 1.30815840e+00f,
            7.99965680e-01f, 1.42059937e-01f, -5.65548599e-01f, -1.22585154e+00f, -1.74428618e+00f, -2.04223609e+00f,
            -2.06881571e+00f, -1.80918705e+00f, -5.01046121e-01f, -2.27570221e-01f, 1.06604151e-01f, 4.57058966e-01f,
            7.73016453e-01f, 1.00466418e+00f, 1.11085117e+00f, 1.06598425e+00f, 8.74208808e-01f, 5.31192720e-01f,
            8.79936293e-02f, -3.98279727e-01f, -8.60470533e-01f, -1.23064530e+00f, -1.45001578e+00f, -1.47807181e+00f,
            -1.31047750e+00f, -9.35767114e-01f, -4.08759743e-01f, 2.04095468e-01f, 8.20599198e-01f, 1.35382557e+00f,
            1.72435915e+00f, 1.87202418e+00f, 1.77418149e+00f, 1.41456723e+00f, 8.41468096e-01f, 1.25720426e-01f,
            -6.38640761e-01f, -1.34672403e+00f, -1.89717293e+00f, -2.20663142e+00f };

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

    // ====================================================================
    // L. FP32 against the host reference: bias, unit offset, groups, backward
    //
    // The numeric gradients are taken of the host reference, which the forward tests here tie to the kernel.
    // ====================================================================

    namespace
    {
        // y = x * rstd(group) * ( w + unit_offset ) + b, each group of group_size normalized on its own.
        void referenceGroupedForward(
            const float* x, const float* weight, const float* bias,
            int64_t rows, int64_t width, int64_t group_size, float epsilon, float unit_offset, std::vector<float>& y )
        {
            y.assign( static_cast<size_t>( rows * width ), 0.0f );

            for ( int64_t r = 0; r < rows; ++r )
            {
                for ( int64_t g = 0; g < width; g += group_size )
                {
                    const float* group = x + r * width + g;

                    double sum_of_squares = 0.0;
                    for ( int64_t i = 0; i < group_size; ++i )
                    {
                        sum_of_squares += static_cast<double>( group[ i ] ) * group[ i ];
                    }

                    const double rstd = 1.0 / std::sqrt( sum_of_squares / group_size + epsilon );

                    for ( int64_t i = 0; i < group_size; ++i )
                    {
                        const int64_t channel = g + i;
                        y[ r * width + channel ] = static_cast<float>(
                            group[ i ] * rstd * ( weight[ channel ] + unit_offset ) + ( bias ? bias[ channel ] : 0.0f ) );
                    }
                }
            }
        }
    }

    class RmsNormFp32CudaTests : public ::testing::Test
    {
    protected:
        using RmsNormType = Mila::Dnn::RmsNorm<DeviceType::Cuda, TensorDataType::FP32>;
        using DeviceTensor = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>;
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        static constexpr int64_t kWidth = 32;

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

        static RmsNormConfig config( bool has_bias, float unit_offset, dim_t group_size = 0 )
        {
            return RmsNormConfig( shape_t{ kWidth } )
                .withEpsilon( kEpsilon )
                .withBias( has_bias )
                .withUnitOffset( unit_offset )
                .withGroupSize( group_size );
        }

        std::unique_ptr<RmsNormType> built( const BuildContext& context, const RmsNormConfig& config, bool has_bias )
        {
            auto norm = std::make_unique<RmsNormType>( "rmsnorm", config, Device::Cuda( 0 ) );
            norm->build( context );

            auto params = norm->getParameters();
            copy( hostWeight(), *static_cast<DeviceTensor*>( params[ 0 ] ), cuda_context_.get() );

            if ( has_bias )
            {
                copy( hostBias(), *static_cast<DeviceTensor*>( params[ 1 ] ), cuda_context_.get() );
            }

            cuda_context_->synchronize();

            return norm;
        }

        std::unique_ptr<RmsNormType> built( const shape_t& shape, RuntimeMode mode, bool has_bias, float unit_offset, dim_t group_size = 0 )
        {
            return built( BuildContext( shape, mode, false ), config( has_bias, unit_offset, group_size ), has_bias );
        }

        static HostFp32 hostWeight()
        {
            HostFp32 host( Device::Cpu(), shape_t{ kWidth } );

            for ( int64_t i = 0; i < kWidth; ++i )
            {
                host.data()[ i ] = weightValue( i );
            }

            return host;
        }

        static HostFp32 hostBias()
        {
            HostFp32 host( Device::Cpu(), shape_t{ kWidth } );

            for ( int64_t i = 0; i < kWidth; ++i )
            {
                host.data()[ i ] = biasValue( i );
            }

            return host;
        }

        static HostFp32 sineHost( const shape_t& shape, double phase )
        {
            HostFp32 host( Device::Cpu(), shape );

            for ( dim_t i = 0; i < host.size(); ++i )
            {
                host.data()[ i ] = static_cast<float>( std::sin( 0.37 * static_cast<double>( i ) + phase ) );
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

        HostFp32 toFloat( const ITensor& device )
        {
            auto host = toHost<TensorDataType::FP32>( static_cast<const DeviceTensor&>( device ), cuda_context_.get() );
            cuda_context_->synchronize();

            return host;
        }

        void expectForwardMatchesReference(
            RmsNormType& norm, const shape_t& shape, bool has_bias, float unit_offset, int64_t group_size )
        {
            auto host_in = sineHost( shape, 0.1 );
            auto& device_out = norm.forward( toDevice( host_in ) );
            norm.synchronize();

            auto out = toFloat( device_out );
            auto weight = hostWeight();
            auto bias = hostBias();

            std::vector<float> expected;
            referenceGroupedForward( host_in.data(), weight.data(), has_bias ? bias.data() : nullptr,
                host_in.size() / kWidth, kWidth, group_size, kEpsilon, unit_offset, expected );

            ASSERT_EQ( out.shape(), shape );

            for ( dim_t i = 0; i < out.size(); ++i )
            {
                EXPECT_NEAR( out.data()[ i ], expected[ i ], 1e-5f + 1e-5f * std::fabs( expected[ i ] ) )
                    << "forward mismatch at index " << i;
            }
        }

        // Backward against numeric gradients of the host reference: input, weight and, with a bias, bias.
        void expectBackwardMatchesNumericGradient( bool has_bias, float unit_offset, int64_t group_size )
        {
            const shape_t shape{ 2, 3, kWidth };
            auto norm = built( shape, RuntimeMode::Training, has_bias, unit_offset, group_size == kWidth ? 0 : group_size );

            auto host_in = sineHost( shape, 0.1 );

            // Non-uniform, so a kernel that ignored its output gradient fails.
            HostFp32 host_grad( Device::Cpu(), shape );
            for ( dim_t i = 0; i < host_grad.size(); ++i )
            {
                host_grad.data()[ i ] = 0.1f * static_cast<float>( ( i % 7 ) + 1 );
            }

            auto device_in = toDevice( host_in );
            auto device_grad = toDevice( host_grad );

            norm->zeroGradients();
            norm->forward( device_in );
            auto& device_in_grad = norm->backward( device_in, device_grad );
            norm->synchronize();

            auto analytic_dx = toFloat( device_in_grad );
            auto grads = norm->getGradients();
            ASSERT_EQ( grads.size(), has_bias ? 2u : 1u );
            auto analytic_dw = toFloat( *grads[ 0 ] );

            auto weight = hostWeight();
            auto bias = hostBias();
            const int64_t rows = host_in.size() / kWidth;
            std::vector<float> reference_out;

            auto evaluate = [&]() -> const float* {
                referenceGroupedForward( host_in.data(), weight.data(), has_bias ? bias.data() : nullptr,
                    rows, kWidth, group_size, kEpsilon, unit_offset, reference_out );
                return reference_out.data();
            };

            const auto numeric_dx = Mila::Tests::Common::centralDifferenceGradient(
                host_in.data(), host_in.size(), host_grad.data(), host_grad.size(), evaluate, 1e-2f );
            Mila::Tests::Common::expectGradientsClose( analytic_dx.data(), numeric_dx, 1e-2f, 1e-2f, "RmsNorm dX" );

            const auto numeric_dw = Mila::Tests::Common::centralDifferenceGradient(
                weight.data(), kWidth, host_grad.data(), host_grad.size(), evaluate, 1e-2f );
            Mila::Tests::Common::expectGradientsClose( analytic_dw.data(), numeric_dw, 1e-2f, 1e-2f, "RmsNorm dW" );

            if ( has_bias )
            {
                auto analytic_db = toFloat( *grads[ 1 ] );
                const auto numeric_db = Mila::Tests::Common::centralDifferenceGradient(
                    bias.data(), kWidth, host_grad.data(), host_grad.size(), evaluate, 1e-2f );
                Mila::Tests::Common::expectGradientsClose( analytic_db.data(), numeric_db, 1e-2f, 1e-2f, "RmsNorm dB" );
            }
        }

        std::unique_ptr<IExecutionContext> cuda_context_;
    };

    TEST_F( RmsNormFp32CudaTests, Forward_MatchesReferenceWithoutBias )
    {
        const shape_t shape{ 2, 3, kWidth };
        auto norm = built( shape, RuntimeMode::Inference, false, 0.0f );

        expectForwardMatchesReference( *norm, shape, false, 0.0f, kWidth );
    }

    TEST_F( RmsNormFp32CudaTests, Forward_AppliesUnitOffset )
    {
        const shape_t shape{ 2, 3, kWidth };
        auto norm = built( shape, RuntimeMode::Inference, true, 1.0f );

        expectForwardMatchesReference( *norm, shape, true, 1.0f, kWidth );
    }

    // A component built at the prefill width is called with a single row at decode.
    TEST_F( RmsNormFp32CudaTests, Forward_NarrowerInputThanBuildMatchesReference )
    {
        auto norm = built( shape_t{ 2, 3, kWidth }, RuntimeMode::Inference, true, 0.0f );

        expectForwardMatchesReference( *norm, shape_t{ 2, 1, kWidth }, true, 0.0f, kWidth );
    }

    TEST_F( RmsNormFp32CudaTests, Backward_MatchesNumericGradient )
    {
        expectBackwardMatchesNumericGradient( true, 0.0f, kWidth );
    }

    // Qwen configures every norm with a unit offset, so the (weight + offset) factor in the input gradient is
    // exercised rather than assumed.
    TEST_F( RmsNormFp32CudaTests, Backward_MatchesNumericGradientWithUnitOffset )
    {
        expectBackwardMatchesNumericGradient( true, 1.0f, kWidth );
    }

    TEST_F( RmsNormFp32CudaTests, Grouped_BackwardMatchesNumericGradient )
    {
        expectBackwardMatchesNumericGradient( false, 1.0f, kGroupSize );
    }

    // A positive control for Grouped_MatchesQwen4Reference: the same input normalized as one group must miss
    // the reference by far more than that test's tolerance.
    TEST_F( RmsNormFp32CudaTests, Grouped_DiffersFromOneGroupOverTheSameExtent )
    {
        auto norm = std::make_unique<RmsNormType>( "rmsnorm",
            RmsNormConfig( shape_t{ kGroupedWidth } ).withEpsilon( 1e-6f ).withBias( false ).withUnitOffset( 1.0f ),
            Device::Cuda( 0 ) );
        norm->build( BuildContext( shape_t{ 2, kGroupedWidth }, RuntimeMode::Inference, false ) );

        HostFp32 host_weight( Device::Cpu(), shape_t{ kGroupedWidth } );

        for ( int64_t j = 0; j < kGroupedWidth; ++j )
        {
            host_weight.data()[ j ] = 0.03f * static_cast<float>( j ) - 0.4f;
        }

        copy( host_weight, *static_cast<DeviceTensor*>( norm->getParameters()[ 0 ] ), cuda_context_.get() );

        auto& device_out = norm->forward( toDevice( sineHost( shape_t{ 2, kGroupedWidth }, 0.1 ) ) );
        norm->synchronize();
        auto out = toFloat( device_out );

        float largest = 0.0f;

        for ( dim_t i = 0; i < out.size(); ++i )
        {
            largest = std::max( largest, std::fabs( out.data()[ i ] - kQwen4GroupedReference[ i ] ) );
        }

        EXPECT_GT( largest, 1e-3f );
    }

    TEST_F( RmsNormFp32CudaTests, Grouped_GroupSizeThatDoesNotDivideTheExtentIsRefused )
    {
        EXPECT_THROW(
            {
                RmsNormType norm( "rmsnorm", config( false, 1.0f, 5 ), Device::Cuda( 0 ) );
                norm.build( BuildContext( shape_t{ 2, kWidth }, RuntimeMode::Inference, false ) );
            },
            std::invalid_argument );
    }

    TEST_F( RmsNormFp32CudaTests, GetRequiredMemory_MatchesBuiltFootprint )
    {
        const BuildContext context = BuildContext( shape_t{ 2, 3, kWidth }, RuntimeMode::Training, false )
            .withAllocationGranularity( allocationGranularity( Device::Cuda( 0 ) ) );

        RmsNormType predictor( "rmsnorm", config( true, 0.0f ), Device::Cuda( 0 ) );
        const MemoryStats predicted = predictor.getRequiredMemory( context );

        auto norm = built( context, config( true, 0.0f ), true );
        const MemoryStats actual = norm->getMemoryStats();

        EXPECT_EQ( predicted.device_parameter_bytes, actual.device_parameter_bytes ) << "parameters";
        EXPECT_EQ( predicted.device_state_bytes, actual.device_state_bytes ) << "state";
        EXPECT_EQ( predicted.device_gradient_bytes, actual.device_gradient_bytes ) << "gradients";
    }

    // ====================================================================
    // K. BF16 at model widths
    //
    // The families' hidden widths (Llama 4096, Gemma 4 12B 3840, Qwen 3.8 5120) and head width (256) take the
    // block-per-row kernel; 36 is not a whole number of 16-byte vectors and takes the warp-per-row one. One row
    // is a decode step.
    // ====================================================================

    class RmsNormBf16WidthTests : public ::testing::Test
    {
    protected:
        using RmsNormType = Mila::Dnn::RmsNorm<DeviceType::Cuda, TensorDataType::BF16>;
        using DeviceTensor = Tensor<TensorDataType::BF16, CudaDeviceMemoryResource>;
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        static constexpr int64_t kWidths[] = { 4096, 3840, 5120, 256, 36 };

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

        std::unique_ptr<RmsNormType> built( int64_t rows, int64_t width, float unit_offset )
        {
            auto norm = std::make_unique<RmsNormType>( "rmsnorm",
                RmsNormConfig( shape_t{ width } ).withEpsilon( kEpsilon ).withBias( false ).withUnitOffset( unit_offset ),
                Device::Cuda( 0 ) );
            norm->build( BuildContext( shape_t{ rows, width }, RuntimeMode::Inference, false ) );

            HostFp32 host_weight( Device::Cpu(), shape_t{ width } );

            for ( int64_t i = 0; i < width; ++i )
                host_weight.data()[ i ] = weightValue( i );

            copy( host_weight, *static_cast<DeviceTensor*>( norm->getParameters()[ 0 ] ), cuda_context_.get() );
            cuda_context_->synchronize();

            return norm;
        }

        // Varied magnitudes and signs, so a lane or vector mapped to the wrong element changes the result.
        DeviceTensor input( int64_t rows, int64_t width, int64_t first_row = 0 )
        {
            HostFp32 host( Device::Cpu(), shape_t{ rows, width } );

            for ( int64_t r = 0; r < rows; ++r )
            {
                for ( int64_t i = 0; i < width; ++i )
                {
                    const double index = static_cast<double>( ( r + first_row ) * width + i );
                    host.data()[ r * width + i ] = static_cast<float>( 3.0 * std::sin( 0.37 * index ) * ( 1.0 + ( i % 11 ) ) );
                }
            }

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

    TEST_F( RmsNormBf16WidthTests, Forward_MatchesReferenceAtModelWidths )
    {
        for ( const float unit_offset : { 0.0f, 1.0f } )
        {
            for ( const int64_t rows : { int64_t{ 1 }, int64_t{ 5 } } )
            {
                for ( const int64_t width : kWidths )
                {
                    auto norm = built( rows, width, unit_offset );
                    auto device_in = input( rows, width );

                    auto& device_out = norm->forward( device_in );
                    norm->synchronize();

                    const auto in = toFloat( device_in );
                    const auto out = toFloat( device_out );

                    std::vector<float> weight( static_cast<size_t>( width ) );
                    std::vector<float> bias( static_cast<size_t>( width ), 0.0f );

                    // The weight as the kernel reads it: BF16, then the offset in FP32.
                    const auto device_weight = toFloat( *static_cast<DeviceTensor*>( norm->getParameters()[ 0 ] ) );

                    for ( int64_t i = 0; i < width; ++i )
                        weight[ static_cast<size_t>( i ) ] = device_weight.data()[ i ] + unit_offset;

                    std::vector<float> expected;
                    referenceForward( in.data(), weight.data(), bias.data(), rows, width, kEpsilon, expected );

                    int mismatches = 0;

                    for ( dim_t i = 0; i < out.size(); ++i )
                    {
                        // One BF16 rounding of the output, plus FP32 summation order.
                        const float tolerance = 1e-3f + 8e-3f * std::fabs( expected[ static_cast<size_t>( i ) ] );

                        if ( std::fabs( out.data()[ i ] - expected[ static_cast<size_t>( i ) ] ) > tolerance && mismatches++ < 3 )
                        {
                            ADD_FAILURE() << "width " << width << ", rows " << rows << ", offset " << unit_offset
                                << ": index " << i << " is " << out.data()[ i ] << ", expected " << expected[ static_cast<size_t>( i ) ];
                        }
                    }
                }
            }
        }
    }

    TEST_F( RmsNormBf16WidthTests, Forward_RowIsTheSameAloneAsInABatch )
    {
        constexpr int64_t kRows = 64;
        constexpr int64_t kRow = 17;

        for ( const int64_t width : kWidths )
        {
            auto batch_norm = built( kRows, width, 1.0f );
            auto batch_in = input( kRows, width );
            auto& batch_out = batch_norm->forward( batch_in );
            batch_norm->synchronize();
            const auto batch = toFloat( batch_out );

            auto single_norm = built( 1, width, 1.0f );
            auto single_in = input( 1, width, kRow );
            auto& single_out = single_norm->forward( single_in );
            single_norm->synchronize();
            const auto single = toFloat( single_out );

            for ( int64_t i = 0; i < width; ++i )
            {
                ASSERT_EQ( single.data()[ i ], batch.data()[ kRow * width + i ] )
                    << "width " << width << ": element " << i << " of row " << kRow << " differs alone and in a batch";
            }
        }
    }

    // ====================================================================
    // K. Grouped normalization (Qwen 4 hyper-connection norms)
    // ====================================================================

    template<typename TPrecisionTag>
    class RmsNormGroupedCudaTests : public RmsNormCudaTests<TPrecisionTag>
    {
    protected:
        using Base = RmsNormCudaTests<TPrecisionTag>;

        std::unique_ptr<typename Base::RmsNormType> builtGrouped( dim_t group_size, const shape_t& shape )
        {
            auto config = RmsNormConfig( shape_t{ kGroupedWidth } )
                .withEpsilon( 1e-6f )
                .withBias( false )
                .withUnitOffset( 1.0f )
                .withGroupSize( group_size );

            auto norm = std::make_unique<typename Base::RmsNormType>( "rmsnorm", config, Device::Cuda( 0 ) );
            norm->build( BuildContext( shape, RuntimeMode::Inference, false ) );

            typename Base::HostFp32 host_weight( Device::Cpu(), shape_t{ kGroupedWidth } );

            for ( int64_t j = 0; j < kGroupedWidth; ++j )
            {
                host_weight.data()[ j ] = 0.03f * static_cast<float>( j ) - 0.4f;
            }

            auto params = norm->getParameters();
            copy( host_weight, *static_cast<typename Base::DeviceTensor*>( params[ 0 ] ), this->cuda_context_.get() );
            this->cuda_context_->synchronize();

            return norm;
        }

        typename Base::HostFp32 sineHost( const shape_t& shape )
        {
            typename Base::HostFp32 host( Device::Cpu(), shape );

            for ( dim_t i = 0; i < host.size(); ++i )
            {
                host.data()[ i ] = static_cast<float>( std::sin( 0.37 * static_cast<double>( i ) + 0.1 ) );
            }

            return host;
        }
    };

    TYPED_TEST_SUITE( RmsNormGroupedCudaTests, RmsNormPrecisions, PrecisionNames );

    TYPED_TEST( RmsNormGroupedCudaTests, Grouped_MatchesQwen4Reference )
    {
        const shape_t shape{ 2, kGroupedWidth };
        auto norm = this->builtGrouped( kGroupSize, shape );
        auto device_input = this->toDevice( this->sineHost( shape ) );

        auto& device_output = norm->forward( device_input );
        norm->synchronize();
        auto output = this->toFloat( device_output );

        for ( dim_t i = 0; i < output.size(); ++i )
        {
            const float expected = kQwen4GroupedReference[ i ];
            const float tolerance = TypeParam::forward_atol + TypeParam::forward_rtol * std::fabs( expected );

            EXPECT_NEAR( output.data()[ i ], expected, tolerance ) << "at index " << i;
        }
    }

    TYPED_TEST( RmsNormGroupedCudaTests, Grouped_GroupSizeEqualToExtentIsBitIdenticalToUngrouped )
    {
        const shape_t shape{ 2, 3, kGroupedWidth };
        auto ungrouped = this->builtGrouped( 0, shape );
        auto whole = this->builtGrouped( kGroupedWidth, shape );
        auto device_input = this->toDevice( this->sineHost( shape ) );

        auto& device_expected = ungrouped->forward( device_input );
        ungrouped->synchronize();
        auto expected = this->toFloat( device_expected );

        auto& device_actual = whole->forward( device_input );
        whole->synchronize();
        auto actual = this->toFloat( device_actual );

        for ( dim_t i = 0; i < expected.size(); ++i )
        {
            EXPECT_EQ( actual.data()[ i ], expected.data()[ i ] ) << "at index " << i;
        }
    }
}
