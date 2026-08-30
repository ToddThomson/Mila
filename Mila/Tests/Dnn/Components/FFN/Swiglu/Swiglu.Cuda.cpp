/**
 * @file Swiglu.Cuda.cpp
 * @brief Concrete-component tests for Swiglu<DeviceType::Cuda, {FP32, BF16}>.
 *
 * SwiGLU is CUDA-only (no CPU SwigluOp specialization), so there is no
 * Swiglu.Cpu.cpp companion -- this file is the whole concrete-component surface.
 * It is a TYPED_TEST precision sweep (see Specifications/Testing.md and the
 * Linear.Cuda.cpp reference) over the two CUDA precisions FP32 and BF16.
 *
 * SwiGLU is a split-and-combine stateless activation: a rank-3 input
 * [B, T, 2H] is split along the last axis into contiguous halves gate|up, and
 * the output [B, T, H] is  Y = SiLU(gate) * up  with  SiLU(x) = x / (1 + e^-x).
 * (The component's inline doc says GELU; the kernel is SiLU -- doc drift noted,
 * not a test concern.)
 *
 * Compiled only under MILA_ENABLE_CUDA. CUDA may still be absent at runtime;
 * SetUp() skips if no device.
 *
 * Numeric strategy mirrors Linear.Cuda.cpp: deterministic input uploaded with
 * conversion to the device precision, read back to float so the reference sees
 * the precision-rounded values the kernel consumed; only the kernel's internal
 * FP32-promoted arithmetic then differs, covered by the per-precision tolerance.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

import Mila;
import Dnn.ActivationType;
// Instantiating Swiglu<Cuda, P> forces CudaSwigluOp's member bodies, which call
// concrete ExecutionContext<Cuda> methods (getStream). The Mila umbrella does not
// complete that type for a consumer TU, so import it directly.

namespace Mila::Tests::Dnn::Components::Activations::Swiglu
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        // Input feature size is 2 * half_width. half_width must be a multiple of
        // the BF16 kernel vector width (8) and the FP32 width (4); 8 satisfies both.
        constexpr int64_t kHalfWidth = 8;
        constexpr int64_t kInFeatures = 2 * kHalfWidth;

        float silu( float x )
        {
            return x / ( 1.0f + std::exp( -x ) );
        }

        // Matches the shared GeluTanh functor (ElementwiseActivation.h) the GeGLU
        // kernel uses: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3))).
        float geluTanh( float x )
        {
            constexpr float kScale = 0.7978845608f;
            constexpr float kCoeff = 0.044715f;
            const float cube = kCoeff * x * x * x;
            return 0.5f * x * ( 1.0f + std::tanh( kScale * ( x + cube ) ) );
        }

        // GeGLU forward reference: Y[j] = GeluTanh(gate[j]) * up[j].
        void referenceForwardGeglu(
            const float* X, int64_t outer, int64_t half_width, std::vector<float>& Y )
        {
            Y.assign( static_cast<size_t>( outer * half_width ), 0.0f );

            for ( int64_t t = 0; t < outer; ++t )
            {
                const float* gate = X + t * 2 * half_width;
                const float* up = gate + half_width;

                for ( int64_t j = 0; j < half_width; ++j )
                {
                    Y[ t * half_width + j ] = geluTanh( gate[ j ] ) * up[ j ];
                }
            }
        }

        // Forward reference: per token (flattened B*T), Y[j] = SiLU(gate[j]) * up[j],
        // gate = X[token, 0:hw], up = X[token, hw:2hw].
        void referenceForward(
            const float* X, int64_t outer, int64_t half_width, std::vector<float>& Y )
        {
            Y.assign( static_cast<size_t>( outer * half_width ), 0.0f );

            for ( int64_t t = 0; t < outer; ++t )
            {
                const float* gate = X + t * 2 * half_width;
                const float* up = gate + half_width;

                for ( int64_t j = 0; j < half_width; ++j )
                {
                    Y[ t * half_width + j ] = silu( gate[ j ] ) * up[ j ];
                }
            }
        }

        // Backward reference (FP32 path only). dX layout mirrors X: [dgate | dup].
        //   dup   = dY * SiLU(gate)
        //   dgate = dY * up * sigmoid * (1 + gate * (1 - sigmoid))
        void referenceBackward(
            const float* X, const float* dY, int64_t outer, int64_t half_width,
            std::vector<float>& dX )
        {
            dX.assign( static_cast<size_t>( outer * 2 * half_width ), 0.0f );

            for ( int64_t t = 0; t < outer; ++t )
            {
                const float* gate = X + t * 2 * half_width;
                const float* up = gate + half_width;

                for ( int64_t j = 0; j < half_width; ++j )
                {
                    const float g = gate[ j ];
                    const float dy = dY[ t * half_width + j ];
                    const float sig = 1.0f / ( 1.0f + std::exp( -g ) );

                    dX[ t * 2 * half_width + j ] = dy * up[ j ] * sig * ( 1.0f + g * ( 1.0f - sig ) );
                    dX[ t * 2 * half_width + half_width + j ] = dy * ( g * sig );
                }
            }
        }

        struct Fp32Precision
        {
            static constexpr TensorDataType value = TensorDataType::FP32;
            static constexpr float forward_atol = 2e-3f;
            static constexpr float forward_rtol = 1e-3f;
            static constexpr float backward_atol = 2e-3f;
            static constexpr float backward_rtol = 1e-3f;
            static constexpr const char* name = "Fp32";
        };

        struct Bf16Precision
        {
            static constexpr TensorDataType value = TensorDataType::BF16;
            static constexpr float forward_atol = 5e-2f;
            static constexpr float forward_rtol = 5e-2f;
            static constexpr float backward_atol = 5e-2f;
            static constexpr float backward_rtol = 5e-2f;
            static constexpr const char* name = "Bf16";
        };

        using SwigluPrecisions = ::testing::Types<Fp32Precision, Bf16Precision>;

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
    class SwigluCudaTests : public ::testing::Test
    {
    protected:
        static constexpr TensorDataType P = TPrecisionTag::value;

        using SwigluType = Mila::Dnn::Swiglu<DeviceType::Cuda, P>;
        using DeviceTensor = Tensor<P, CudaDeviceMemoryResource>;
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        static_assert( SwigluType::getDeviceType() == DeviceType::Cuda );
        static_assert( SwigluType::getPrecision() == P );

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

        std::unique_ptr<SwigluType> builtSwiglu( const shape_t& shape, RuntimeMode mode )
        {
            auto swiglu = std::make_unique<SwigluType>( "swiglu", SwigluConfig(), Device::Cuda( 0 ) );
            swiglu->build( BuildContext( shape, mode, false ) );

            return swiglu;
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

    TYPED_TEST_SUITE( SwigluCudaTests, SwigluPrecisions, PrecisionNames );

    // ====================================================================
    // A. Construction
    // ====================================================================

    TYPED_TEST( SwigluCudaTests, Construct_StandaloneSucceeds )
    {
        typename TestFixture::SwigluType swiglu( "swiglu", SwigluConfig(), Device::Cuda( 0 ) );

        EXPECT_EQ( swiglu.getDeviceId().type, DeviceType::Cuda );
    }

    // ====================================================================
    // B. Build Lifecycle (Swiglu preconditions)
    // ====================================================================

    TYPED_TEST( SwigluCudaTests, Forward_ThrowsBeforeBuild )
    {
        typename TestFixture::SwigluType swiglu( "swiglu", SwigluConfig(), Device::Cuda( 0 ) );
        typename TestFixture::DeviceTensor input( Device::Cuda( 0 ), shape_t{ 2, 3, kInFeatures } );

        EXPECT_THROW( swiglu.forward( input ), std::runtime_error );
    }

    TYPED_TEST( SwigluCudaTests, Build_ThrowsOnOddFeatureDim )
    {
        typename TestFixture::SwigluType swiglu( "swiglu", SwigluConfig(), Device::Cuda( 0 ) );

        // Feature dimension must be even for the gate/value split.
        EXPECT_THROW(
            swiglu.build( BuildContext( shape_t{ 2, 3, 15 }, RuntimeMode::Inference, false ) ),
            std::invalid_argument );
    }

    TYPED_TEST( SwigluCudaTests, Build_ThrowsOnNonRank3 )
    {
        typename TestFixture::SwigluType swiglu( "swiglu", SwigluConfig(), Device::Cuda( 0 ) );

        EXPECT_THROW(
            swiglu.build( BuildContext( shape_t{ 2, kInFeatures }, RuntimeMode::Inference, false ) ),
            std::invalid_argument );
    }

    // ====================================================================
    // E. Forward (numeric vs reference)
    // ====================================================================

    TYPED_TEST( SwigluCudaTests, Forward_MatchesReference )
    {
        const shape_t shape{ 2, 3, kInFeatures };

        auto swiglu = this->builtSwiglu( shape, RuntimeMode::Inference );

        auto host_in = this->spreadHost( shape );
        auto device_in = this->toDevice( host_in );

        auto& device_out = swiglu->forward( device_in );
        swiglu->synchronize();

        auto out = this->toFloat( device_out );
        auto in = this->toFloat( device_in );

        const int64_t outer = 2 * 3;
        ASSERT_EQ( out.shape(), ( shape_t{ 2, 3, kHalfWidth } ) );

        std::vector<float> expected;
        referenceForward( in.data(), outer, kHalfWidth, expected );

        ASSERT_EQ( out.size(), expected.size() );

        for ( dim_t i = 0; i < out.size(); ++i )
        {
            const float tolerance = TypeParam::forward_atol + TypeParam::forward_rtol * std::fabs( expected[ i ] );

            EXPECT_NEAR( out.data()[ i ], expected[ i ], tolerance )
                << "forward mismatch at index " << i;
        }
    }

    // GeGLU gate (Swiglu<..., ActivationType::Gelu>): forward = GeluTanh(gate) * up,
    // routed to the separate CudaGegluOp / kernel. The SiLU path is untouched.
    TYPED_TEST( SwigluCudaTests, Forward_GeGLU_MatchesReference )
    {
        using GegluType = Mila::Dnn::Swiglu<DeviceType::Cuda, TestFixture::P, ActivationType::Gelu>;

        const shape_t shape{ 2, 3, kInFeatures };

        auto geglu = std::make_unique<GegluType>( "geglu", SwigluConfig(), Device::Cuda( 0 ) );
        geglu->build( BuildContext( shape, RuntimeMode::Inference, false ) );

        auto host_in = this->spreadHost( shape );
        auto device_in = this->toDevice( host_in );

        auto& device_out = geglu->forward( device_in );
        geglu->synchronize();

        auto out = this->toFloat( device_out );
        auto in = this->toFloat( device_in );

        const int64_t outer = 2 * 3;
        ASSERT_EQ( out.shape(), ( shape_t{ 2, 3, kHalfWidth } ) );

        std::vector<float> expected;
        referenceForwardGeglu( in.data(), outer, kHalfWidth, expected );

        ASSERT_EQ( out.size(), expected.size() );

        for ( dim_t i = 0; i < out.size(); ++i )
        {
            const float tolerance = TypeParam::forward_atol + TypeParam::forward_rtol * std::fabs( expected[ i ] );

            EXPECT_NEAR( out.data()[ i ], expected[ i ], tolerance )
                << "GeGLU forward mismatch at index " << i;
        }
    }

    // ====================================================================
    // F. Backward (numeric vs analytic gradient)
    // ====================================================================

    TYPED_TEST( SwigluCudaTests, Backward_MatchesReferenceGradients )
    {
        // BF16 backward is blocked: CudaSwigluOp::backward keeps gradients in FP32
        // (kernel design), casting the component's BF16 grad tensors to float* --
        // a dtype/size mismatch that corrupts the BF16 grad buffer. FP32 is the
        // consistent path. See BACKLOG.
        if ( TypeParam::value == TensorDataType::BF16 )
        {
            GTEST_SKIP() << "BF16 SwiGLU backward grad-dtype mismatch (FP32 kernel grads vs BF16 tensors) — see BACKLOG";
        }

        const shape_t input_shape{ 2, 3, kInFeatures };
        const shape_t output_shape{ 2, 3, kHalfWidth };

        auto swiglu = this->builtSwiglu( input_shape, RuntimeMode::Training );

        auto host_in = this->spreadHost( input_shape );
        auto device_in = this->toDevice( host_in );

        typename TestFixture::HostFp32 host_grad( Device::Cpu(), output_shape );
        for ( dim_t i = 0; i < host_grad.size(); ++i )
        {
            host_grad.data()[ i ] = 0.1f * static_cast<float>( ( i % 7 ) + 1 );
        }
        auto device_grad = this->toDevice( host_grad );

        swiglu->forward( device_in );
        auto& device_in_grad = swiglu->backward( device_in, device_grad );
        swiglu->synchronize();

        auto input_grad = this->toFloat( device_in_grad );
        auto in = this->toFloat( device_in );
        auto grad = this->toFloat( device_grad );

        const int64_t outer = 2 * 3;
        std::vector<float> expected;
        referenceBackward( in.data(), grad.data(), outer, kHalfWidth, expected );

        ASSERT_EQ( input_grad.size(), expected.size() );

        for ( dim_t i = 0; i < input_grad.size(); ++i )
        {
            const float tolerance = TypeParam::backward_atol + TypeParam::backward_rtol * std::fabs( expected[ i ] );

            EXPECT_NEAR( input_grad.data()[ i ], expected[ i ], tolerance )
                << "input-gradient mismatch at index " << i;
        }
    }

    // ====================================================================
    // G. Parameters & Gradients (Swiglu is stateless)
    // ====================================================================

    TYPED_TEST( SwigluCudaTests, Parameters_AreEmpty )
    {
        auto swiglu = this->builtSwiglu( shape_t{ 2, 3, kInFeatures }, RuntimeMode::Inference );

        EXPECT_EQ( swiglu->parameterCount(), 0 );
        EXPECT_TRUE( swiglu->getParameters().empty() );
        EXPECT_TRUE( swiglu->getGradients().empty() );
    }

    // ====================================================================
    // J. Type identity
    // ====================================================================

    TYPED_TEST( SwigluCudaTests, GetType_IsSwiglu )
    {
        typename TestFixture::SwigluType swiglu( "swiglu", SwigluConfig(), Device::Cuda( 0 ) );

        EXPECT_EQ( swiglu.getType(), ComponentType::Swiglu );
    }
}
