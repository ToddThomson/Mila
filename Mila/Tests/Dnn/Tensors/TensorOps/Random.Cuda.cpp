/**
 * @file Random.Cuda.cpp
 * @brief CUDA tests for TensorOps fill_normal / fill_uniform / xavier (Dnn.TensorOps:Random).
 *
 * Device companion to Random.Cpu.cpp, and the init-at-precision oracle ROADMAP's Training
 * Revival names among its success criteria.
 *
 * These were deliberately not written before: the CUDA path generated cuRAND's FP32 output
 * straight into the tensor buffer regardless of precision, so a BF16 tensor of n elements
 * received n floats -- 4n bytes into a 2n-byte buffer. That is a heap overrun, and the
 * values that landed in range were FP32 bit patterns reinterpreted in pairs as BF16, which
 * puts roughly half of them around 1e14. The BF16 cases below fail loudly against that
 * behaviour and pass against the scratch-buffer-and-narrow path that replaced it.
 *
 * Assertions are on the distribution rather than on exact values: cuRAND's sequence is not
 * a contract Mila should pin. Seeding makes each run reproducible, so a failure is
 * repeatable rather than intermittent.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

import Mila;
import Compute.ExecutionContext;

namespace Mila::Tests::Dnn::Tensors::TensorOps
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        struct Fp32Precision
        {
            static constexpr TensorDataType value = TensorDataType::FP32;
            // FP32 round-trips the generated value exactly.
            static constexpr float bound_slack = 1e-6f;
            static constexpr const char* name = "Fp32";
        };

        struct Bf16Precision
        {
            static constexpr TensorDataType value = TensorDataType::BF16;
            // Narrowing to BF16 rounds to nearest, so a value may land just outside a
            // uniform bound by up to half an ulp. BF16 carries 8 mantissa bits, so the
            // relative step is 2^-8; this slack is that, with margin, for bounds of O(1).
            static constexpr float bound_slack = 1e-2f;
            static constexpr const char* name = "Bf16";
        };

        using RandomPrecisions = ::testing::Types<Fp32Precision, Bf16Precision>;

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
    class TensorOpsRandomCudaTests : public ::testing::Test
    {
    protected:
        static constexpr TensorDataType P = TPrecisionTag::value;

        using DeviceTensor = Tensor<P, CudaDeviceMemoryResource>;

        void SetUp() override
        {
            try
            {
                context_ = createExecutionContext( Device::Cuda( 0 ) );
            }
            catch ( const std::exception& )
            {
                context_ = nullptr;
            }

            // Deterministic across runs: a distribution assertion that fails should fail
            // every time, not one run in fifty.
            Mila::Core::RandomGenerator::getInstance().setSeed( 20260728u );
        }

        std::vector<float> readBack( const DeviceTensor& device )
        {
            auto host = toHost<TensorDataType::FP32>( device, context_.get() );
            context_->synchronize();

            return std::vector<float>( host.data(), host.data() + host.size() );
        }

        static float mean( const std::vector<float>& v )
        {
            double sum = 0.0;
            for ( float x : v ) sum += x;

            return static_cast<float>( sum / static_cast<double>( v.size() ) );
        }

        static float stddev( const std::vector<float>& v )
        {
            const double m = mean( v );
            double acc = 0.0;
            for ( float x : v ) acc += ( x - m ) * ( x - m );

            return static_cast<float>( std::sqrt( acc / static_cast<double>( v.size() ) ) );
        }

        std::unique_ptr<IExecutionContext> context_;
    };

    TYPED_TEST_SUITE( TensorOpsRandomCudaTests, RandomPrecisions, PrecisionNames );

#define SKIP_WITHOUT_CUDA()                                            \
    if ( this->context_ == nullptr )                                   \
    {                                                                  \
        GTEST_SKIP() << "CUDA device not available.";                  \
    }

    // ====================================================================
    // A. fill_normal
    // ====================================================================

    TYPED_TEST( TensorOpsRandomCudaTests, FillNormal_MatchesRequestedDistribution )
    {
        SKIP_WITHOUT_CUDA();

        // 8192 samples: the sample mean's standard error is stddev/sqrt(n), so the
        // bounds below sit far outside sampling noise while still catching a path that
        // writes the wrong type.
        constexpr dim_t kCount = 8192;
        constexpr float kStdDev = 0.02f;

        typename TestFixture::DeviceTensor tensor( Device::Cuda( 0 ), shape_t{ kCount } );
        fill_normal( tensor, 0.0f, kStdDev, this->context_.get() );
        this->context_->synchronize();

        auto values = this->readBack( tensor );
        ASSERT_EQ( values.size(), static_cast<size_t>( kCount ) );

        // The decisive assertion for the BF16 overrun: reinterpreting FP32 bits as BF16
        // pairs put roughly half the values near 1e14. Anything beyond a few dozen
        // standard deviations is not a sample from this distribution.
        for ( size_t i = 0; i < values.size(); ++i )
        {
            ASSERT_TRUE( std::isfinite( values[ i ] ) ) << "non-finite value at index " << i;
            ASSERT_LT( std::fabs( values[ i ] ), 40.0f * kStdDev )
                << "value at index " << i << " is far outside N(0, " << kStdDev << ")";
        }

        EXPECT_NEAR( TestFixture::mean( values ), 0.0f, 0.005f );
        EXPECT_NEAR( TestFixture::stddev( values ), kStdDev, 0.1f * kStdDev );
    }

    // curandGenerateNormal requires an even count (Box-Muller pairs), so an odd count
    // takes a padded scratch buffer. That padding path is independent of the precision
    // narrowing, and on FP32 it is the only reason a scratch buffer appears at all.
    TYPED_TEST( TensorOpsRandomCudaTests, FillNormal_OddElementCountIsFullyWritten )
    {
        SKIP_WITHOUT_CUDA();

        constexpr dim_t kCount = 1025;
        constexpr float kStdDev = 0.05f;

        typename TestFixture::DeviceTensor tensor( Device::Cuda( 0 ), shape_t{ kCount } );
        zero( tensor );
        this->context_->synchronize();

        fill_normal( tensor, 0.0f, kStdDev, this->context_.get() );
        this->context_->synchronize();

        auto values = this->readBack( tensor );
        ASSERT_EQ( values.size(), static_cast<size_t>( kCount ) );

        for ( size_t i = 0; i < values.size(); ++i )
        {
            ASSERT_TRUE( std::isfinite( values[ i ] ) ) << "non-finite value at index " << i;
            ASSERT_LT( std::fabs( values[ i ] ), 40.0f * kStdDev ) << "at index " << i;
        }

        // The last element is the one the padded generate-and-copy could drop.
        EXPECT_NE( values.back(), 0.0f ) << "final element left unwritten by the padded path";
    }

    TYPED_TEST( TensorOpsRandomCudaTests, FillNormal_EmptyTensorIsNoOp )
    {
        SKIP_WITHOUT_CUDA();

        typename TestFixture::DeviceTensor tensor( Device::Cuda( 0 ), shape_t{ 0 } );

        EXPECT_NO_THROW( fill_normal( tensor, 0.0f, 0.02f, this->context_.get() ) );
    }

    // ====================================================================
    // B. fill_uniform
    // ====================================================================

    TYPED_TEST( TensorOpsRandomCudaTests, FillUniform_StaysWithinBounds )
    {
        SKIP_WITHOUT_CUDA();

        constexpr dim_t kCount = 8192;
        constexpr float kMin = -0.5f;
        constexpr float kMax = 1.5f;

        typename TestFixture::DeviceTensor tensor( Device::Cuda( 0 ), shape_t{ kCount } );
        fill_uniform( tensor, kMin, kMax, this->context_.get() );
        this->context_->synchronize();

        auto values = this->readBack( tensor );
        ASSERT_EQ( values.size(), static_cast<size_t>( kCount ) );

        const float slack = TypeParam::bound_slack;

        for ( size_t i = 0; i < values.size(); ++i )
        {
            ASSERT_GE( values[ i ], kMin - slack ) << "below range at index " << i;
            ASSERT_LE( values[ i ], kMax + slack ) << "above range at index " << i;
        }

        // A uniform distribution over an asymmetric range must sit near its midpoint --
        // catches a scale/shift applied to the wrong buffer as well as no shift at all.
        EXPECT_NEAR( TestFixture::mean( values ), 0.5f * ( kMin + kMax ), 0.05f );
    }

    TYPED_TEST( TensorOpsRandomCudaTests, FillUniform_EmptyTensorIsNoOp )
    {
        SKIP_WITHOUT_CUDA();

        typename TestFixture::DeviceTensor tensor( Device::Cuda( 0 ), shape_t{ 0 } );

        EXPECT_NO_THROW( fill_uniform( tensor, -1.0f, 1.0f, this->context_.get() ) );
    }

    // ====================================================================
    // C. xavier -- the path Linear actually takes on a training build
    // ====================================================================

    TYPED_TEST( TensorOpsRandomCudaTests, Xavier_RespectsGlorotBound )
    {
        SKIP_WITHOUT_CUDA();

        constexpr dim_t kFanIn = 256;
        constexpr dim_t kFanOut = 128;

        const float limit = std::sqrt( 6.0f / static_cast<float>( kFanIn + kFanOut ) );

        typename TestFixture::DeviceTensor tensor( Device::Cuda( 0 ), shape_t{ kFanOut, kFanIn } );
        xavier( tensor, kFanIn, kFanOut, this->context_.get() );
        this->context_->synchronize();

        auto values = this->readBack( tensor );
        ASSERT_EQ( values.size(), static_cast<size_t>( kFanIn * kFanOut ) );

        const float slack = TypeParam::bound_slack;

        for ( size_t i = 0; i < values.size(); ++i )
        {
            ASSERT_TRUE( std::isfinite( values[ i ] ) ) << "non-finite value at index " << i;
            ASSERT_LE( std::fabs( values[ i ] ), limit + slack )
                << "outside the Glorot bound " << limit << " at index " << i;
        }

        EXPECT_NEAR( TestFixture::mean( values ), 0.0f, 0.05f * limit + slack );
    }

    // ====================================================================
    // D. The buffer-extent contract the overrun violated
    // ====================================================================

    // A random fill must write inside its own tensor and nowhere else. Two tensors are
    // allocated back to back and only the first is filled; the second is checked to be
    // untouched. Allocations are not guaranteed adjacent, so this is not a proof -- but
    // the old path overran by a full tensor's worth of bytes, which is exactly the size
    // that lands in a neighbouring allocation.
    TYPED_TEST( TensorOpsRandomCudaTests, FillNormal_DoesNotWriteBeyondItsTensor )
    {
        SKIP_WITHOUT_CUDA();

        constexpr dim_t kCount = 4096;

        typename TestFixture::DeviceTensor target( Device::Cuda( 0 ), shape_t{ kCount } );
        typename TestFixture::DeviceTensor neighbour( Device::Cuda( 0 ), shape_t{ kCount } );

        zero( neighbour );
        this->context_->synchronize();

        fill_normal( target, 0.0f, 0.02f, this->context_.get() );
        this->context_->synchronize();

        auto neighbour_values = this->readBack( neighbour );

        for ( size_t i = 0; i < neighbour_values.size(); ++i )
        {
            ASSERT_EQ( neighbour_values[ i ], 0.0f )
                << "neighbouring allocation modified at index " << i;
        }
    }

#undef SKIP_WITHOUT_CUDA
}
