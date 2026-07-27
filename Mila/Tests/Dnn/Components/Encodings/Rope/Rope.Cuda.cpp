/**
 * @file Rope.Cuda.cpp
 * @brief Concrete-component tests for Rope<DeviceType::Cuda, {FP32, BF16}>.
 *
 * Rope is CUDA-only (no CPU RopeOp), a stateless PAIRED leaf that rotates Q and K
 * in-place. This is the whole concrete surface, a TYPED_TEST sweep over FP32 and
 * BF16. Three dispatch modes are covered: forward (positions 0..T-1), prefill
 * (explicit position offset), and decode (single token at an explicit position).
 *
 * Convention (matches the kernel, Rope.Fp32.cu, half-split / Llama style): for
 * frequency pair i in [0, head_dim/2),
 *   theta_i = base^(-2i/head_dim),  angle = pos * theta_i
 *   x0 = q[i], x1 = q[i + head_dim/2]
 *   q'[i]            = x0*cos - x1*sin
 *   q'[i+head_dim/2] = x0*sin + x1*cos
 * Heads are laid out [B, T, n_heads, head_dim] row-major.
 *
 * Backward is the exact inverse rotation (RoPE is orthogonal), so the backward
 * test is a convention-independent round-trip: forward then backward recovers the
 * input.
 *
 * Compiled only under MILA_ENABLE_CUDA; SetUp() skips if no device at runtime.
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

namespace Mila::Tests::Dnn::Components::Encodings::Rope
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        constexpr int64_t kHeadDim = 8;
        constexpr int64_t kHeads = 2;
        constexpr int64_t kKvHeads = 1;
        constexpr int64_t kChannels = kHeads * kHeadDim;      // Q width
        constexpr int64_t kKvChannels = kKvHeads * kHeadDim;  // K width
        constexpr int64_t kMaxSeq = 16;
        constexpr float kBase = 10000.0f;

        // Rotate a flat [B, T, n_heads, head_dim] host buffer in place.
        // inverse = true applies the transpose (backward) rotation.
        // rope_pairs < 0 rotates all head_dim/2 pairs (full RoPE); a smaller value
        // rotates only the first rope_pairs pairs and passes the rest through
        // unchanged (proportional partial-rotary).
        void ropeRotate(
            float* d, int64_t B, int64_t T, int64_t n_heads, int64_t head_dim,
            float base, int position_offset, bool inverse, int rope_pairs = -1 )
        {
            const int64_t half = head_dim / 2;
            const int64_t pairs = ( rope_pairs < 0 || rope_pairs > half ) ? half : rope_pairs;

            for ( int64_t b = 0; b < B; ++b )
            {
                for ( int64_t t = 0; t < T; ++t )
                {
                    const int64_t abs_pos = t + position_offset;

                    for ( int64_t h = 0; h < n_heads; ++h )
                    {
                        const int64_t base_idx = ( ( b * T + t ) * n_heads + h ) * head_dim;

                        for ( int64_t i = 0; i < pairs; ++i )
                        {
                            const double theta = std::pow( static_cast<double>( base ),
                                -2.0 * static_cast<double>( i ) / static_cast<double>( head_dim ) );
                            const double angle = static_cast<double>( abs_pos ) * theta;
                            const float c = static_cast<float>( std::cos( angle ) );
                            const float s = static_cast<float>( std::sin( angle ) );

                            const float x0 = d[ base_idx + i ];
                            const float x1 = d[ base_idx + i + half ];

                            if ( !inverse )
                            {
                                d[ base_idx + i ] = x0 * c - x1 * s;
                                d[ base_idx + i + half ] = x0 * s + x1 * c;
                            }
                            else
                            {
                                d[ base_idx + i ] = x0 * c + x1 * s;
                                d[ base_idx + i + half ] = -x0 * s + x1 * c;
                            }
                        }
                    }
                }
            }
        }

        struct Fp32Precision
        {
            static constexpr TensorDataType value = TensorDataType::FP32;
            static constexpr float atol = 1e-3f;
            static constexpr float rtol = 1e-3f;
            static constexpr const char* name = "Fp32";
        };

        struct Bf16Precision
        {
            static constexpr TensorDataType value = TensorDataType::BF16;
            static constexpr float atol = 5e-2f;
            static constexpr float rtol = 5e-2f;
            static constexpr const char* name = "Bf16";
        };

        using RopePrecisions = ::testing::Types<Fp32Precision, Bf16Precision>;

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
    class RopeCudaTests : public ::testing::Test
    {
    protected:
        static constexpr TensorDataType P = TPrecisionTag::value;

        using RopeType = Mila::Dnn::Rope<DeviceType::Cuda, P>;
        using DeviceTensor = Tensor<P, CudaDeviceMemoryResource>;
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        static_assert( RopeType::getDeviceType() == DeviceType::Cuda );
        static_assert( RopeType::getPrecision() == P );

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

        RopeConfig config()
        {
            return RopeConfig( kChannels, kHeads, kKvHeads, kMaxSeq );
        }

        std::unique_ptr<RopeType> builtRope( const shape_t& leading_shape, RuntimeMode mode )
        {
            auto rope = std::make_unique<RopeType>( "rope", config(), Device::Cuda( 0 ) );
            rope->build( BuildContext( leading_shape, mode, false ) );

            return rope;
        }

        HostFp32 spreadHost( const shape_t& shape, float phase )
        {
            HostFp32 host( Device::Cpu(), shape );
            for ( size_t i = 0; i < host.size(); ++i )
            {
                host.data()[ i ] = std::sin( 0.3f * static_cast<float>( i ) + phase );
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

        void expectClose( const HostFp32& actual, const std::vector<float>& expected, const char* label )
        {
            ASSERT_EQ( actual.size(), expected.size() );

            for ( size_t i = 0; i < actual.size(); ++i )
            {
                const float tolerance = TPrecisionTag::atol + TPrecisionTag::rtol * std::fabs( expected[ i ] );
                EXPECT_NEAR( actual.data()[ i ], expected[ i ], tolerance )
                    << label << " mismatch at index " << i;
            }
        }

        std::unique_ptr<IExecutionContext> cuda_context_;
    };

    TYPED_TEST_SUITE( RopeCudaTests, RopePrecisions, PrecisionNames );

    // ====================================================================
    // A. Construction
    // ====================================================================

    TYPED_TEST( RopeCudaTests, Construct_StandaloneSucceeds )
    {
        typename TestFixture::RopeType rope( "rope", this->config(), Device::Cuda( 0 ) );

        EXPECT_EQ( rope.getDeviceId().type, DeviceType::Cuda );
    }

    TYPED_TEST( RopeCudaTests, Construct_DeviceTypeMismatchThrows )
    {
        EXPECT_THROW(
            typename TestFixture::RopeType( "rope", this->config(), Device::Cpu() ),
            std::invalid_argument );
    }

    // ====================================================================
    // B. Build Lifecycle (preconditions)
    // ====================================================================

    TYPED_TEST( RopeCudaTests, Forward_ThrowsBeforeBuild )
    {
        typename TestFixture::RopeType rope( "rope", this->config(), Device::Cuda( 0 ) );
        typename TestFixture::DeviceTensor Q( Device::Cuda( 0 ), shape_t{ 2, 4, kChannels } );
        typename TestFixture::DeviceTensor K( Device::Cuda( 0 ), shape_t{ 2, 4, kKvChannels } );

        EXPECT_THROW( rope.forward( Q, K ), std::runtime_error );
    }

    TYPED_TEST( RopeCudaTests, Build_ThrowsOnRank1LeadingShape )
    {
        typename TestFixture::RopeType rope( "rope", this->config(), Device::Cuda( 0 ) );

        EXPECT_THROW(
            rope.build( BuildContext( shape_t{ 4 }, RuntimeMode::Inference, false ) ),
            std::invalid_argument );
    }

    // ====================================================================
    // E. Forward (numeric vs reference)
    // ====================================================================

    TYPED_TEST( RopeCudaTests, Forward_RotatesQAndK )
    {
        const int64_t B = 2;
        const int64_t T = 4;

        auto rope = this->builtRope( shape_t{ B, T }, RuntimeMode::Inference );

        auto device_q = this->toDevice( this->spreadHost( shape_t{ B, T, kChannels }, 0.0f ) );
        auto device_k = this->toDevice( this->spreadHost( shape_t{ B, T, kKvChannels }, 1.7f ) );

        // Capture the precision-rounded inputs the kernel actually consumes.
        auto q_in = this->toFloat( device_q );
        auto k_in = this->toFloat( device_k );

        rope->forward( device_q, device_k );
        rope->synchronize();

        std::vector<float> expected_q( q_in.data(), q_in.data() + q_in.size() );
        std::vector<float> expected_k( k_in.data(), k_in.data() + k_in.size() );
        ropeRotate( expected_q.data(), B, T, kHeads, kHeadDim, kBase, 0, false );
        ropeRotate( expected_k.data(), B, T, kKvHeads, kHeadDim, kBase, 0, false );

        this->expectClose( this->toFloat( device_q ), expected_q, "forward Q" );
        this->expectClose( this->toFloat( device_k ), expected_k, "forward K" );
    }

    // Proportional partial-rotary (Gemma global layers): with rotary_dim < head_dim,
    // only the first rotary_dim dimensions are rotated; the rest pass through
    // unchanged because their cos/sin cache entries are 1/0 (zero frequency).
    TYPED_TEST( RopeCudaTests, Forward_PartialRotary_PassesThroughUpperDims )
    {
        const int64_t B = 2;
        const int64_t T = 4;
        const int64_t rotary_dim = 4;   // of kHeadDim=8 -> rotate first 2 pairs, pass through 2

        auto cfg = RopeConfig( kChannels, kHeads, kKvHeads, kMaxSeq ).withRotaryDim( rotary_dim );
        auto rope = std::make_unique<typename TestFixture::RopeType>( "rope", cfg, Device::Cuda( 0 ) );
        rope->build( BuildContext( shape_t{ B, T }, RuntimeMode::Inference, false ) );

        auto device_q = this->toDevice( this->spreadHost( shape_t{ B, T, kChannels }, 0.0f ) );
        auto device_k = this->toDevice( this->spreadHost( shape_t{ B, T, kKvChannels }, 1.7f ) );

        auto q_in = this->toFloat( device_q );
        auto k_in = this->toFloat( device_k );

        rope->forward( device_q, device_k );
        rope->synchronize();

        std::vector<float> expected_q( q_in.data(), q_in.data() + q_in.size() );
        std::vector<float> expected_k( k_in.data(), k_in.data() + k_in.size() );
        const int rope_pairs = static_cast<int>( rotary_dim / 2 );
        ropeRotate( expected_q.data(), B, T, kHeads, kHeadDim, kBase, 0, false, rope_pairs );
        ropeRotate( expected_k.data(), B, T, kKvHeads, kHeadDim, kBase, 0, false, rope_pairs );

        this->expectClose( this->toFloat( device_q ), expected_q, "partial-rotary Q" );
        this->expectClose( this->toFloat( device_k ), expected_k, "partial-rotary K" );
    }

    TYPED_TEST( RopeCudaTests, Prefill_AppliesPositionOffset )
    {
        const int64_t B = 2;
        const int64_t T = 4;
        const int offset = 2;

        auto rope = this->builtRope( shape_t{ B, T }, RuntimeMode::Inference );

        auto device_q = this->toDevice( this->spreadHost( shape_t{ B, T, kChannels }, 0.0f ) );
        auto device_k = this->toDevice( this->spreadHost( shape_t{ B, T, kKvChannels }, 1.7f ) );

        auto q_in = this->toFloat( device_q );
        auto k_in = this->toFloat( device_k );

        rope->prefill( device_q, device_k, offset );
        rope->synchronize();

        std::vector<float> expected_q( q_in.data(), q_in.data() + q_in.size() );
        std::vector<float> expected_k( k_in.data(), k_in.data() + k_in.size() );
        ropeRotate( expected_q.data(), B, T, kHeads, kHeadDim, kBase, offset, false );
        ropeRotate( expected_k.data(), B, T, kKvHeads, kHeadDim, kBase, offset, false );

        this->expectClose( this->toFloat( device_q ), expected_q, "prefill Q" );
        this->expectClose( this->toFloat( device_k ), expected_k, "prefill K" );
    }

    TYPED_TEST( RopeCudaTests, Decode_RotatesAtExplicitPosition )
    {
        const int64_t B = 2;
        const int position = 3;

        auto rope = this->builtRope( shape_t{ B, 4 }, RuntimeMode::Inference );

        auto device_q = this->toDevice( this->spreadHost( shape_t{ B, 1, kChannels }, 0.0f ) );
        auto device_k = this->toDevice( this->spreadHost( shape_t{ B, 1, kKvChannels }, 1.7f ) );

        auto q_in = this->toFloat( device_q );
        auto k_in = this->toFloat( device_k );

        rope->decode( device_q, device_k, position );
        rope->synchronize();

        std::vector<float> expected_q( q_in.data(), q_in.data() + q_in.size() );
        std::vector<float> expected_k( k_in.data(), k_in.data() + k_in.size() );
        ropeRotate( expected_q.data(), B, 1, kHeads, kHeadDim, kBase, position, false );
        ropeRotate( expected_k.data(), B, 1, kKvHeads, kHeadDim, kBase, position, false );

        this->expectClose( this->toFloat( device_q ), expected_q, "decode Q" );
        this->expectClose( this->toFloat( device_k ), expected_k, "decode K" );
    }

    // ====================================================================
    // D/F. Backward (training mode + inverse-rotation round-trip)
    // ====================================================================

    TYPED_TEST( RopeCudaTests, Backward_ThrowsWhenNotTraining )
    {
        const int64_t B = 2;
        const int64_t T = 4;
        auto rope = this->builtRope( shape_t{ B, T }, RuntimeMode::Inference );

        auto device_q = this->toDevice( this->spreadHost( shape_t{ B, T, kChannels }, 0.0f ) );
        auto device_k = this->toDevice( this->spreadHost( shape_t{ B, T, kKvChannels }, 1.7f ) );

        EXPECT_THROW( rope->backward( device_q, device_k ), std::runtime_error );
    }

    TYPED_TEST( RopeCudaTests, Backward_InverseRotationRecoversInput )
    {
        const int64_t B = 2;
        const int64_t T = 4;

        auto rope = this->builtRope( shape_t{ B, T }, RuntimeMode::Training );

        auto device_q = this->toDevice( this->spreadHost( shape_t{ B, T, kChannels }, 0.0f ) );
        auto device_k = this->toDevice( this->spreadHost( shape_t{ B, T, kKvChannels }, 1.7f ) );

        // Original (precision-rounded) values, before any rotation.
        auto q0 = this->toFloat( device_q );
        auto k0 = this->toFloat( device_k );

        // Forward rotates Q, K in place; backward inverse-rotates them back.
        rope->forward( device_q, device_k );
        rope->synchronize();

        auto [dq, dk] = rope->backward( device_q, device_k );
        rope->synchronize();

        std::vector<float> expected_q( q0.data(), q0.data() + q0.size() );
        std::vector<float> expected_k( k0.data(), k0.data() + k0.size() );

        this->expectClose( this->toFloat( dq ), expected_q, "round-trip Q" );
        this->expectClose( this->toFloat( dk ), expected_k, "round-trip K" );
    }

    // ====================================================================
    // G. Parameters (Rope is stateless)
    // ====================================================================

    TYPED_TEST( RopeCudaTests, Parameters_AreEmpty )
    {
        auto rope = this->builtRope( shape_t{ 2, 4 }, RuntimeMode::Inference );

        EXPECT_EQ( rope->parameterCount(), 0 );
        EXPECT_TRUE( rope->getParameters().empty() );
        EXPECT_TRUE( rope->getGradients().empty() );
    }

    // ====================================================================
    // J. Type identity
    // ====================================================================

    TYPED_TEST( RopeCudaTests, GetType_IsRope )
    {
        typename TestFixture::RopeType rope( "rope", this->config(), Device::Cuda( 0 ) );

        EXPECT_EQ( rope.getType(), ComponentType::Rope );
    }
}
