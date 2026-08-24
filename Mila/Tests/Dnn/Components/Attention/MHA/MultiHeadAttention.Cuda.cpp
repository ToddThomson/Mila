/**
 * @file MultiHeadAttention.Cuda.cpp
 * @brief Concrete-component tests for MultiHeadAttention<DeviceType::Cuda, {FP32}>.
 *
 * CUDA companion to MultiHeadAttention.Cpu.cpp, written as a precision sweep
 * (TYPED_TEST) per the methodology default: every component is potentially
 * multi-precision, so the supported-precision list is the single point of change.
 * Today the list holds FP32 only -- the CUDA MhaOp kernel dispatch is constrained
 * to float/half (CudaMhaOp.Dispatch.ixx: `float || half`), there is no BF16 path,
 * and MHA is GPT-2 lineage (the reference GPT-2 transformer is the sole consumer,
 * FP32 weights; the Llama path uses GroupedQueryAttention). When a BF16 MHA kernel
 * exists, add a Bf16Precision tag to MhaPrecisions and the suite re-runs for it.
 *
 * forward() drives the KV-cache prefill path; decode() is the single-token
 * autoregressive path. Both are validated against the same causal self-attention
 * reference (concatenated QKV [B,T,3C] -> [B,T,C]). The decode test prefills the
 * first tokens, then decodes the next token at its absolute position; the result
 * must equal that query's row in the full-sequence reference (the KV cache supplies
 * the earlier keys/values).
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

namespace Mila::Tests::Dnn::Components::Attention::MHA
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        constexpr int64_t kModelDim = 8;
        constexpr int64_t kNumHeads = 2;

        void referenceAttention(
            const float* X, int64_t B, int64_t T, int64_t C, int64_t NH, std::vector<float>& Y )
        {
            const int64_t HS = C / NH;
            const int64_t qkv = 3 * C;
            const float scale = 1.0f / std::sqrt( static_cast<float>( HS ) );
            Y.assign( static_cast<size_t>( B * T * C ), 0.0f );

            std::vector<float> scores( static_cast<size_t>( T ) );

            for ( int64_t b = 0; b < B; ++b )
            {
                for ( int64_t h = 0; h < NH; ++h )
                {
                    for ( int64_t i = 0; i < T; ++i )
                    {
                        const float* q = X + ( b * T + i ) * qkv + 0 * C + h * HS;

                        float maxv = -INFINITY;
                        for ( int64_t j = 0; j <= i; ++j )
                        {
                            const float* k = X + ( b * T + j ) * qkv + 1 * C + h * HS;
                            float s = 0.0f;
                            for ( int64_t d = 0; d < HS; ++d )
                            {
                                s += q[ d ] * k[ d ];
                            }
                            s *= scale;
                            scores[ j ] = s;
                            maxv = std::max( maxv, s );
                        }

                        double sum = 0.0;
                        for ( int64_t j = 0; j <= i; ++j )
                        {
                            scores[ j ] = std::exp( scores[ j ] - maxv );
                            sum += scores[ j ];
                        }

                        for ( int64_t d = 0; d < HS; ++d )
                        {
                            double acc = 0.0;
                            for ( int64_t j = 0; j <= i; ++j )
                            {
                                const float* v = X + ( b * T + j ) * qkv + 2 * C + h * HS;
                                acc += ( scores[ j ] / sum ) * v[ d ];
                            }
                            Y[ ( b * T + i ) * C + h * HS + d ] = static_cast<float>( acc );
                        }
                    }
                }
            }
        }

        struct Fp32Precision
        {
            static constexpr TensorDataType value = TensorDataType::FP32;
            static constexpr float atol = 2e-3f;
            static constexpr float rtol = 1e-3f;
            static constexpr const char* name = "Fp32";
        };

        // FP32 only: the CUDA MhaOp kernel is float/half (no BF16), and MHA is
        // GPT-2 lineage (FP32 weights). Add a Bf16Precision tag here once a BF16
        // MHA kernel exists -- the rest of the suite re-runs unchanged.
        using MhaPrecisions = ::testing::Types<Fp32Precision>;

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
    class MhaCudaTests : public ::testing::Test
    {
    protected:
        static constexpr TensorDataType P = TPrecisionTag::value;

        using MhaType = Mila::Dnn::MultiHeadAttention<DeviceType::Cuda, P>;
        using DeviceTensor = Tensor<P, CudaDeviceMemoryResource>;
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        static_assert( MhaType::getDeviceType() == DeviceType::Cuda );
        static_assert( MhaType::getPrecision() == P );

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

        MultiHeadAttentionConfig config()
        {
            return MultiHeadAttentionConfig( kModelDim, kNumHeads );
        }

        std::unique_ptr<MhaType> builtMha( const shape_t& shape, RuntimeMode mode )
        {
            auto mha = std::make_unique<MhaType>( "mha", config(), Device::Cuda( 0 ) );
            mha->build( BuildContext( shape, mode, false ) );

            return mha;
        }

        HostFp32 spreadHost( const shape_t& shape, float phase )
        {
            HostFp32 host( Device::Cpu(), shape );
            for ( dim_t i = 0; i < host.size(); ++i )
            {
                host.data()[ i ] = std::sin( 0.2f * static_cast<float>( i ) + phase );
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

    TYPED_TEST_SUITE( MhaCudaTests, MhaPrecisions, PrecisionNames );

    // ====================================================================
    // A. Construction
    // ====================================================================

    TYPED_TEST( MhaCudaTests, Construct_StandaloneSucceeds )
    {
        typename TestFixture::MhaType mha( "mha", this->config(), Device::Cuda( 0 ) );

        EXPECT_EQ( mha.getDeviceId().type, DeviceType::Cuda );
    }

    TYPED_TEST( MhaCudaTests, Construct_DeviceTypeMismatchThrows )
    {
        EXPECT_THROW(
            typename TestFixture::MhaType( "mha", this->config(), Device::Cpu() ),
            std::invalid_argument );
    }

    // ====================================================================
    // C. Capability resolution
    // ====================================================================

    TYPED_TEST( MhaCudaTests, SupportsKvCache_TrueAfterBuild )
    {
        auto mha = this->builtMha( shape_t{ 2, 3, 3 * kModelDim }, RuntimeMode::Inference );

        EXPECT_TRUE( mha->supportsKvCache() );
    }

    // ====================================================================
    // E. Forward (numeric vs causal-attention reference, prefill path)
    // ====================================================================

    TYPED_TEST( MhaCudaTests, Forward_MatchesCausalReference )
    {
        const int64_t B = 2;
        const int64_t T = 3;
        const shape_t shape{ B, T, 3 * kModelDim };

        auto mha = this->builtMha( shape, RuntimeMode::Inference );

        auto host_in = this->spreadHost( shape, 0.0f );
        auto device_in = this->toDevice( host_in );

        auto& device_out = mha->forward( device_in );
        mha->synchronize();

        auto out = this->toFloat( device_out );
        auto in = this->toFloat( device_in );

        std::vector<float> expected;
        referenceAttention( in.data(), B, T, kModelDim, kNumHeads, expected );

        ASSERT_EQ( out.shape(), ( shape_t{ B, T, kModelDim } ) );
        ASSERT_EQ( out.size(), expected.size() );

        for ( dim_t i = 0; i < out.size(); ++i )
        {
            const float tolerance = TypeParam::atol + TypeParam::rtol * std::fabs( expected[ i ] );
            EXPECT_NEAR( out.data()[ i ], expected[ i ], tolerance ) << "attention mismatch at index " << i;
        }
    }

    // Prefill the first two tokens, then decode the third at position 2; the
    // decode output must equal that query's row in the full-sequence reference.
    TYPED_TEST( MhaCudaTests, Decode_MatchesReferenceAfterPrefill )
    {
        const int64_t B = 2;
        const int64_t T = 3;
        const int64_t qkv = 3 * kModelDim;

        auto mha = this->builtMha( shape_t{ B, T, qkv }, RuntimeMode::Inference );

        // Full 3-token QKV sequence; prefill = tokens [0,1], decode = token 2.
        auto full = this->spreadHost( shape_t{ B, T, qkv }, 0.0f );

        typename TestFixture::HostFp32 prefill_host( Device::Cpu(), shape_t{ B, 2, qkv } );
        typename TestFixture::HostFp32 decode_host( Device::Cpu(), shape_t{ B, 1, qkv } );
        for ( int64_t b = 0; b < B; ++b )
        {
            for ( int64_t c = 0; c < qkv; ++c )
            {
                prefill_host.data()[ ( b * 2 + 0 ) * qkv + c ] = full.data()[ ( b * T + 0 ) * qkv + c ];
                prefill_host.data()[ ( b * 2 + 1 ) * qkv + c ] = full.data()[ ( b * T + 1 ) * qkv + c ];
                decode_host.data()[ b * qkv + c ] = full.data()[ ( b * T + 2 ) * qkv + c ];
            }
        }

        auto prefill_device = this->toDevice( prefill_host );
        auto decode_device = this->toDevice( decode_host );

        mha->forward( prefill_device );
        auto& device_out = mha->decode( decode_device, 2 );
        mha->synchronize();

        auto out = this->toFloat( device_out );

        // Reference over the precision-rounded full sequence; take query row i=2.
        auto full_rounded = this->toFloat( this->toDevice( full ) );
        std::vector<float> ref;
        referenceAttention( full_rounded.data(), B, T, kModelDim, kNumHeads, ref );

        ASSERT_EQ( out.shape(), ( shape_t{ B, 1, kModelDim } ) );

        for ( int64_t b = 0; b < B; ++b )
        {
            for ( int64_t c = 0; c < kModelDim; ++c )
            {
                const float expected = ref[ ( b * T + 2 ) * kModelDim + c ];
                const float actual = out.data()[ b * kModelDim + c ];
                const float tolerance = TypeParam::atol + TypeParam::rtol * std::fabs( expected );

                EXPECT_NEAR( actual, expected, tolerance ) << "decode mismatch at (b,c)=(" << b << "," << c << ")";
            }
        }
    }

    // ====================================================================
    // D. Runtime mode — backward requires training build
    // ====================================================================

    TYPED_TEST( MhaCudaTests, Backward_ThrowsWhenNotTraining )
    {
        const shape_t shape{ 2, 3, 3 * kModelDim };
        auto mha = this->builtMha( shape, RuntimeMode::Inference );

        auto device_in = this->toDevice( this->spreadHost( shape, 0.0f ) );
        typename TestFixture::DeviceTensor output_grad( Device::Cuda( 0 ), shape_t{ 2, 3, kModelDim } );

        mha->forward( device_in );

        EXPECT_THROW( mha->backward( device_in, output_grad ), std::runtime_error );
    }

    // ====================================================================
    // G. Parameters (stateless) & J. Type
    // ====================================================================

    TYPED_TEST( MhaCudaTests, Parameters_AreEmpty )
    {
        auto mha = this->builtMha( shape_t{ 2, 3, 3 * kModelDim }, RuntimeMode::Inference );

        EXPECT_EQ( mha->parameterCount(), 0 );
        EXPECT_TRUE( mha->getParameters().empty() );
    }

    TYPED_TEST( MhaCudaTests, GetType_IsMultiHeadAttention )
    {
        typename TestFixture::MhaType mha( "mha", this->config(), Device::Cuda( 0 ) );

        EXPECT_EQ( mha.getType(), ComponentType::MultiHeadAttention );
    }
}
