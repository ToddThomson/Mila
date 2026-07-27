/**
 * @file Lpe.Cuda.cpp
 * @brief Concrete-component tests for Lpe<DeviceType::Cuda, INT32, {FP32}>.
 *
 * CUDA companion to Lpe.Cpu.cpp, written as a precision sweep (TYPED_TEST) per the
 * methodology default. The supported-precision list holds FP32 only: the CUDA LpeOp
 * kernel dispatch is constrained to float/half (CudaLpeOp.Dispatch.ixx: `float ||
 * half`), there is no BF16 path, and LPE is GPT-2 lineage (Llama uses RoPE, not
 * learned positional encoding). Add a Bf16Precision tag to LpePrecisions once a BF16
 * kernel exists and the suite re-runs for it. Forward is the token gather plus
 * per-position add:
 *   output[b, t, :] = wte[ X[b, t], : ] + wpe[ t, : ]
 * Decode looks the positional row up at an explicit position (KV-cache path):
 *   output[0, 0, :] = wte[ X[0, 0], : ] + wpe[ position, : ]
 *
 * Compiled only under MILA_ENABLE_CUDA; SetUp() skips if no device at runtime.
 * Supersedes the pre-methodology Encoder.Cuda.cpp (retired in place, Section 3).
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

namespace Mila::Tests::Dnn::Components::Encodings::Lpe
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        constexpr int64_t kVocab = 8;
        constexpr int64_t kMaxSeq = 8;
        constexpr int64_t kEmbed = 4;

        float wteValue( int64_t v, int64_t c )
        {
            return 0.1f * static_cast<float>( v ) + 0.01f * static_cast<float>( c );
        }

        float wpeValue( int64_t p, int64_t c )
        {
            return -0.05f * static_cast<float>( p ) + 0.2f * static_cast<float>( c );
        }

        int32_t tokenAt( int64_t flat )
        {
            return static_cast<int32_t>( ( flat * 3 + 1 ) % kVocab );
        }

        struct Fp32Precision
        {
            static constexpr TensorDataType value = TensorDataType::FP32;
            static constexpr float atol = 1e-4f;
            static constexpr float rtol = 1e-4f;
            static constexpr const char* name = "Fp32";
        };

        // FP32 only: the CUDA LpeOp kernel is float/half (no BF16), and LPE is
        // GPT-2 lineage. Add a Bf16Precision tag here once a BF16 kernel exists --
        // the rest of the suite re-runs unchanged.
        using LpePrecisions = ::testing::Types<Fp32Precision>;

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
    class LpeCudaTests : public ::testing::Test
    {
    protected:
        static constexpr TensorDataType P = TPrecisionTag::value;

        using LpeType = Mila::Dnn::Lpe<DeviceType::Cuda, TensorDataType::INT32, P>;
        using DeviceTensor = Tensor<P, CudaDeviceMemoryResource>;
        using IndexDeviceTensor = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>;
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        using HostIndex = Tensor<TensorDataType::INT32, CpuMemoryResource>;

        static_assert( LpeType::getDeviceType() == DeviceType::Cuda );
        static_assert( LpeType::getPrecision() == P );

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

        LpeConfig config()
        {
            return LpeConfig()
                .withEmbeddingDim( kEmbed )
                .withMaxSequenceLength( kMaxSeq )
                .withVocabularyLength( kVocab );
        }

        std::unique_ptr<LpeType> builtLpe( const shape_t& token_shape, RuntimeMode mode )
        {
            auto lpe = std::make_unique<LpeType>( "lpe", config(), Device::Cuda( 0 ) );
            lpe->build( BuildContext( token_shape, mode, false ) );

            return lpe;
        }

        // params order is { wte, wpe }; upload the deterministic reference tables.
        void setKnownTables( LpeType& lpe )
        {
            auto params = lpe.getParameters();

            HostFp32 host_wte( Device::Cpu(), shape_t{ kVocab, kEmbed } );
            HostFp32 host_wpe( Device::Cpu(), shape_t{ kMaxSeq, kEmbed } );

            for ( int64_t v = 0; v < kVocab; ++v )
            {
                for ( int64_t c = 0; c < kEmbed; ++c )
                {
                    host_wte.data()[ v * kEmbed + c ] = wteValue( v, c );
                }
            }

            for ( int64_t p = 0; p < kMaxSeq; ++p )
            {
                for ( int64_t c = 0; c < kEmbed; ++c )
                {
                    host_wpe.data()[ p * kEmbed + c ] = wpeValue( p, c );
                }
            }

            copy( host_wte, *static_cast<DeviceTensor*>( params[ 0 ] ), cuda_context_.get() );
            copy( host_wpe, *static_cast<DeviceTensor*>( params[ 1 ] ), cuda_context_.get() );
            cuda_context_->synchronize();
        }

        HostIndex rampTokens( const shape_t& token_shape )
        {
            HostIndex host( Device::Cpu(), token_shape );
            for ( dim_t i = 0; i < host.size(); ++i )
            {
                host.data()[ i ] = tokenAt( static_cast<int64_t>( i ) );
            }

            return host;
        }

        IndexDeviceTensor toDevice( const HostIndex& host )
        {
            IndexDeviceTensor device( Device::Cuda( 0 ), host.shape() );
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

    TYPED_TEST_SUITE( LpeCudaTests, LpePrecisions, PrecisionNames );

    // ====================================================================
    // A. Construction
    // ====================================================================

    TYPED_TEST( LpeCudaTests, Construct_StandaloneSucceeds )
    {
        typename TestFixture::LpeType lpe( "lpe", this->config(), Device::Cuda( 0 ) );

        EXPECT_EQ( lpe.getDeviceId().type, DeviceType::Cuda );
    }

    TYPED_TEST( LpeCudaTests, Construct_DeviceTypeMismatchThrows )
    {
        EXPECT_THROW(
            typename TestFixture::LpeType( "lpe", this->config(), Device::Cpu() ),
            std::invalid_argument );
    }

    // ====================================================================
    // E. Forward (numeric vs reference)
    // ====================================================================

    TYPED_TEST( LpeCudaTests, Forward_TokenEmbeddingPlusPositional )
    {
        const shape_t token_shape{ 2, 3 };
        const int64_t B = 2;
        const int64_t T = 3;

        auto lpe = this->builtLpe( token_shape, RuntimeMode::Inference );
        this->setKnownTables( *lpe );

        auto host_tokens = this->rampTokens( token_shape );
        auto device_tokens = this->toDevice( host_tokens );

        auto& device_out = lpe->forward( device_tokens );
        lpe->synchronize();

        auto out = this->toFloat( device_out );

        auto params = lpe->getParameters();
        auto wte = this->toFloat( *static_cast<typename TestFixture::DeviceTensor*>( params[ 0 ] ) );
        auto wpe = this->toFloat( *static_cast<typename TestFixture::DeviceTensor*>( params[ 1 ] ) );

        ASSERT_EQ( out.shape(), ( shape_t{ B, T, kEmbed } ) );

        for ( int64_t b = 0; b < B; ++b )
        {
            for ( int64_t t = 0; t < T; ++t )
            {
                const int32_t idx = host_tokens.data()[ b * T + t ];
                for ( int64_t c = 0; c < kEmbed; ++c )
                {
                    const float expected = wte.data()[ idx * kEmbed + c ] + wpe.data()[ t * kEmbed + c ];
                    const float actual = out.data()[ ( b * T + t ) * kEmbed + c ];
                    const float tolerance = TypeParam::atol + TypeParam::rtol * std::fabs( expected );

                    EXPECT_NEAR( actual, expected, tolerance )
                        << "mismatch at (b,t,c)=(" << b << "," << t << "," << c << ")";
                }
            }
        }
    }

    // Decode looks up the positional row at an explicit position, not t=0.
    TYPED_TEST( LpeCudaTests, Decode_UsesExplicitPosition )
    {
        const int position = 5;

        auto lpe = this->builtLpe( shape_t{ 1, kMaxSeq }, RuntimeMode::Inference );
        this->setKnownTables( *lpe );

        typename TestFixture::HostIndex host_token( Device::Cpu(), shape_t{ 1, 1 } );
        host_token.data()[ 0 ] = 3;
        auto device_token = this->toDevice( host_token );

        auto& device_out = lpe->decode( device_token, position );
        lpe->synchronize();

        auto out = this->toFloat( device_out );

        auto params = lpe->getParameters();
        auto wte = this->toFloat( *static_cast<typename TestFixture::DeviceTensor*>( params[ 0 ] ) );
        auto wpe = this->toFloat( *static_cast<typename TestFixture::DeviceTensor*>( params[ 1 ] ) );

        ASSERT_EQ( out.shape(), ( shape_t{ 1, 1, kEmbed } ) );

        for ( int64_t c = 0; c < kEmbed; ++c )
        {
            const float expected = wte.data()[ 3 * kEmbed + c ] + wpe.data()[ position * kEmbed + c ];
            const float tolerance = TypeParam::atol + TypeParam::rtol * std::fabs( expected );

            EXPECT_NEAR( out.data()[ c ], expected, tolerance ) << "decode mismatch at c=" << c;
        }
    }

    // ====================================================================
    // D. Runtime mode — backward requires training build
    // ====================================================================

    TYPED_TEST( LpeCudaTests, Backward_ThrowsWhenNotTraining )
    {
        const shape_t token_shape{ 2, 3 };
        auto lpe = this->builtLpe( token_shape, RuntimeMode::Inference );
        this->setKnownTables( *lpe );

        auto device_tokens = this->toDevice( this->rampTokens( token_shape ) );
        lpe->forward( device_tokens );

        typename TestFixture::DeviceTensor output_grad( Device::Cuda( 0 ), shape_t{ 2, 3, kEmbed } );

        EXPECT_THROW( lpe->backward( device_tokens, output_grad ), std::runtime_error );
    }

    // ====================================================================
    // G. Parameters & Gradients
    // ====================================================================

    TYPED_TEST( LpeCudaTests, Parameters_WteAndWpe )
    {
        auto lpe = this->builtLpe( shape_t{ 2, 3 }, RuntimeMode::Inference );

        EXPECT_EQ( lpe->getParameters().size(), 2u );
        EXPECT_EQ( lpe->parameterCount(), static_cast<size_t>( kVocab * kEmbed + kMaxSeq * kEmbed ) );
    }

    TYPED_TEST( LpeCudaTests, Gradients_PresentOnlyForTrainingBuild )
    {
        auto inference = this->builtLpe( shape_t{ 2, 3 }, RuntimeMode::Inference );
        EXPECT_TRUE( inference->getGradients().empty() );

        auto training = this->builtLpe( shape_t{ 2, 3 }, RuntimeMode::Training );
        EXPECT_EQ( training->getGradients().size(), 2u );
    }

    // ====================================================================
    // J. Type identity
    // ====================================================================

    TYPED_TEST( LpeCudaTests, GetType_IsLpe )
    {
        typename TestFixture::LpeType lpe( "lpe", this->config(), Device::Cuda( 0 ) );

        EXPECT_EQ( lpe.getType(), ComponentType::Lpe );
    }
}
