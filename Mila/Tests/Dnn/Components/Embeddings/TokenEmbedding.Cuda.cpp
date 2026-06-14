/**
 * @file TokenEmbedding.Cuda.cpp
 * @brief Concrete-component tests for TokenEmbedding<DeviceType::Cuda, INT32, {FP32, BF16}>.
 *
 * TokenEmbedding is CUDA-only (no CPU TokenEmbeddingOp), so there is no
 * TokenEmbedding.Cpu.cpp -- this is the whole concrete surface, a TYPED_TEST
 * sweep over FP32 and BF16 (see Linear.Cuda.cpp / RmsNorm.Cuda.cpp for the
 * precision-sweep reference). The input is a rank-2 [B, T] INT32 token-index
 * tensor; forward is a pure vocabulary gather:
 *   output[b, t, :] = wte[ X[b, t], : ]
 *
 * The prefill -> decode runtime axis (build for [B, T], forward a [B, 1] decode
 * shape) is exercised in section E; for TokenEmbedding outer_size == 1 selects
 * the dedicated decode kernel.
 *
 * Compiled only under MILA_ENABLE_CUDA; SetUp() skips if no device at runtime.
 *
 * Numeric strategy matches RmsNorm.Cuda.cpp: deterministic wte uploaded with
 * conversion to the device precision, read back to float so the reference sees
 * the precision-rounded values the kernel gathered.
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

namespace Mila::Tests::Dnn::Components::Embeddings
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        constexpr int64_t kVocab = 8;
        constexpr int64_t kEmbed = 4;

        // wte[v, c]; spread so adjacent rows/columns are distinguishable.
        float wteValue( int64_t v, int64_t c )
        {
            return 0.25f * static_cast<float>( v ) - 0.5f + 0.1f * static_cast<float>( c );
        }

        // Deterministic token indices in [0, kVocab).
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

        struct Bf16Precision
        {
            static constexpr TensorDataType value = TensorDataType::BF16;
            static constexpr float atol = 5e-3f;
            static constexpr float rtol = 5e-3f;
            static constexpr const char* name = "Bf16";
        };

        using TokenEmbeddingPrecisions = ::testing::Types<Fp32Precision, Bf16Precision>;

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
    class TokenEmbeddingCudaTests : public ::testing::Test
    {
    protected:
        static constexpr TensorDataType P = TPrecisionTag::value;

        using EmbeddingType = Mila::Dnn::TokenEmbedding<DeviceType::Cuda, TensorDataType::INT32, P>;
        using DeviceTensor = Tensor<P, CudaDeviceMemoryResource>;
        using IndexDeviceTensor = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>;
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        using HostIndex = Tensor<TensorDataType::INT32, CpuMemoryResource>;

        static_assert( EmbeddingType::getDeviceType() == DeviceType::Cuda );
        static_assert( EmbeddingType::getPrecision() == P );

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

        TokenEmbeddingConfig config()
        {
            return TokenEmbeddingConfig().withVocabSize( kVocab ).withEmbeddingDim( kEmbed );
        }

        std::unique_ptr<EmbeddingType> builtEmbedding( const shape_t& token_shape, RuntimeMode mode )
        {
            auto embedding = std::make_unique<EmbeddingType>( "token_embedding", config(), Device::Cuda( 0 ) );
            embedding->build( BuildContext( token_shape, mode, false ) );

            return embedding;
        }

        // wte is the only parameter; fill it with the deterministic reference table.
        void setKnownWte( EmbeddingType& embedding )
        {
            auto params = embedding.getParameters();

            HostFp32 host_wte( Device::Cpu(), shape_t{ kVocab, kEmbed } );
            for ( int64_t v = 0; v < kVocab; ++v )
            {
                for ( int64_t c = 0; c < kEmbed; ++c )
                {
                    host_wte.data()[ v * kEmbed + c ] = wteValue( v, c );
                }
            }

            copy( host_wte, *static_cast<DeviceTensor*>( params[ 0 ] ), cuda_context_.get() );
            cuda_context_->synchronize();
        }

        HostIndex rampTokens( const shape_t& token_shape )
        {
            HostIndex host( Device::Cpu(), token_shape );
            for ( size_t i = 0; i < host.size(); ++i )
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

        // expected[(b*T+t)*C + c] = wte[ tokens[b*T+t], c ]
        void referenceGather(
            const HostIndex& tokens, const HostFp32& wte,
            int64_t outer, int64_t channels, std::vector<float>& expected )
        {
            expected.assign( static_cast<size_t>( outer * channels ), 0.0f );

            for ( int64_t r = 0; r < outer; ++r )
            {
                const int32_t idx = tokens.data()[ r ];
                for ( int64_t c = 0; c < channels; ++c )
                {
                    expected[ r * channels + c ] = wte.data()[ idx * channels + c ];
                }
            }
        }

        std::unique_ptr<IExecutionContext> cuda_context_;
    };

    TYPED_TEST_SUITE( TokenEmbeddingCudaTests, TokenEmbeddingPrecisions, PrecisionNames );

    // ====================================================================
    // A. Construction
    // ====================================================================

    TYPED_TEST( TokenEmbeddingCudaTests, Construct_StandaloneSucceeds )
    {
        typename TestFixture::EmbeddingType embedding( "token_embedding", this->config(), Device::Cuda( 0 ) );

        EXPECT_EQ( embedding.getDeviceId().type, DeviceType::Cuda );
    }

    TYPED_TEST( TokenEmbeddingCudaTests, Construct_DeviceTypeMismatchThrows )
    {
        EXPECT_THROW(
            typename TestFixture::EmbeddingType( "token_embedding", this->config(), Device::Cpu() ),
            std::invalid_argument );
    }

    TYPED_TEST( TokenEmbeddingCudaTests, Accessors_ReturnConfiguredDims )
    {
        typename TestFixture::EmbeddingType embedding( "token_embedding", this->config(), Device::Cuda( 0 ) );

        EXPECT_EQ( embedding.getVocabSize(), kVocab );
        EXPECT_EQ( embedding.getEmbeddingDim(), kEmbed );
    }

    // ====================================================================
    // B. Build Lifecycle (preconditions)
    // ====================================================================

    TYPED_TEST( TokenEmbeddingCudaTests, Forward_ThrowsBeforeBuild )
    {
        typename TestFixture::EmbeddingType embedding( "token_embedding", this->config(), Device::Cuda( 0 ) );
        typename TestFixture::IndexDeviceTensor input( Device::Cuda( 0 ), shape_t{ 2, 3 } );

        EXPECT_THROW( embedding.forward( input ), std::runtime_error );
    }

    TYPED_TEST( TokenEmbeddingCudaTests, Build_ThrowsOnNonRank2Input )
    {
        typename TestFixture::EmbeddingType embedding( "token_embedding", this->config(), Device::Cuda( 0 ) );

        // TokenEmbedding requires a rank-2 [B, T] token-index shape; rank-3 must fail.
        EXPECT_THROW(
            embedding.build( BuildContext( shape_t{ 2, 3, kEmbed }, RuntimeMode::Inference, false ) ),
            std::invalid_argument );
    }

    // ====================================================================
    // E. Forward (numeric vs reference)
    // ====================================================================

    TYPED_TEST( TokenEmbeddingCudaTests, Forward_GathersEmbeddingRows )
    {
        const shape_t token_shape{ 2, 3 };
        const int64_t outer = 2 * 3;

        auto embedding = this->builtEmbedding( token_shape, RuntimeMode::Inference );
        this->setKnownWte( *embedding );

        auto host_tokens = this->rampTokens( token_shape );
        auto device_tokens = this->toDevice( host_tokens );

        auto& device_out = embedding->forward( device_tokens );
        embedding->synchronize();

        auto out = this->toFloat( device_out );

        auto params = embedding->getParameters();
        auto wte = this->toFloat( *static_cast<typename TestFixture::DeviceTensor*>( params[ 0 ] ) );

        ASSERT_EQ( out.shape(), ( shape_t{ 2, 3, kEmbed } ) );

        std::vector<float> expected;
        this->referenceGather( host_tokens, wte, outer, kEmbed, expected );

        ASSERT_EQ( out.size(), expected.size() );

        for ( size_t i = 0; i < out.size(); ++i )
        {
            const float tolerance = TypeParam::atol + TypeParam::rtol * std::fabs( expected[ i ] );

            EXPECT_NEAR( out.data()[ i ], expected[ i ], tolerance )
                << "gather mismatch at index " << i;
        }
    }

    // Prefill -> decode runtime axis: build for [B, T], forward a single-token
    // [B, 1] step (outer_size == 1 selects the decode kernel).
    TYPED_TEST( TokenEmbeddingCudaTests, Forward_DecodeShapeSingleToken )
    {
        const shape_t prefill_shape{ 2, 3 };
        const shape_t decode_shape{ 2, 1 };
        const int64_t outer = 2 * 1;

        auto embedding = this->builtEmbedding( prefill_shape, RuntimeMode::Inference );
        this->setKnownWte( *embedding );

        auto host_tokens = this->rampTokens( decode_shape );
        auto device_tokens = this->toDevice( host_tokens );

        auto& device_out = embedding->forward( device_tokens );
        embedding->synchronize();

        auto out = this->toFloat( device_out );

        auto params = embedding->getParameters();
        auto wte = this->toFloat( *static_cast<typename TestFixture::DeviceTensor*>( params[ 0 ] ) );

        ASSERT_EQ( out.shape(), ( shape_t{ 2, 1, kEmbed } ) );

        std::vector<float> expected;
        this->referenceGather( host_tokens, wte, outer, kEmbed, expected );

        ASSERT_EQ( out.size(), expected.size() );

        for ( size_t i = 0; i < out.size(); ++i )
        {
            const float tolerance = TypeParam::atol + TypeParam::rtol * std::fabs( expected[ i ] );

            EXPECT_NEAR( out.data()[ i ], expected[ i ], tolerance )
                << "decode gather mismatch at index " << i;
        }
    }

    // ====================================================================
    // D. Runtime mode — backward requires training build
    // ====================================================================

    TYPED_TEST( TokenEmbeddingCudaTests, Backward_ThrowsWhenNotTraining )
    {
        const shape_t token_shape{ 2, 3 };
        auto embedding = this->builtEmbedding( token_shape, RuntimeMode::Inference );
        this->setKnownWte( *embedding );

        auto device_tokens = this->toDevice( this->rampTokens( token_shape ) );
        embedding->forward( device_tokens );

        typename TestFixture::DeviceTensor output_grad( Device::Cuda( 0 ), shape_t{ 2, 3, kEmbed } );

        EXPECT_THROW( embedding->backward( device_tokens, output_grad ), std::runtime_error );
    }

    // ====================================================================
    // G. Parameters & Gradients
    // ====================================================================

    TYPED_TEST( TokenEmbeddingCudaTests, Parameters_SingleWteTable )
    {
        auto embedding = this->builtEmbedding( shape_t{ 2, 3 }, RuntimeMode::Inference );

        EXPECT_EQ( embedding->getParameters().size(), 1u );
        EXPECT_EQ( embedding->parameterCount(), static_cast<size_t>( kVocab * kEmbed ) );
    }

    TYPED_TEST( TokenEmbeddingCudaTests, Gradients_PresentOnlyForTrainingBuild )
    {
        auto inference = this->builtEmbedding( shape_t{ 2, 3 }, RuntimeMode::Inference );
        EXPECT_TRUE( inference->getGradients().empty() );

        auto training = this->builtEmbedding( shape_t{ 2, 3 }, RuntimeMode::Training );
        EXPECT_EQ( training->getGradients().size(), 1u );
    }

    // ====================================================================
    // J. Type identity
    // ====================================================================

    TYPED_TEST( TokenEmbeddingCudaTests, GetType_IsTokenEmbedding )
    {
        typename TestFixture::EmbeddingType embedding( "token_embedding", this->config(), Device::Cuda( 0 ) );

        EXPECT_EQ( embedding.getType(), ComponentType::TokenEmbedding );
    }
}
