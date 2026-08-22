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
 *
 * Section H covers the TTableResidency axis: a table in pinned host memory, gathered
 * across PCIe by the same kernel, which is how Qwen 3.8 keeps 2.37 GiB off a 12 GiB
 * card (Specifications/Qwen3.8.md section 5, item 6).
 *
 * Section F covers the TTableQuantization = PerChannelFp8<> axis (D4 Design B):
 * quantize-on-load through loadParameter("wte", bf16_blob) and the FP8
 * gather-dequant forward, bounded by the E4M3 half-ulp error against the BF16
 * source table.
 */

#include <gtest/gtest.h>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

import Mila;
import Compute.ExecutionContext;
// The serialization metadata type is not re-exported through the Mila umbrella; import
// its module directly (as the Src consumers do). MSVC surfaced it transitively via
// import Mila, clang does not. The weight-quant policies used to need the same treatment
// and no longer do -- Mila.ixx exports them.
import Serialization.Tensor;

namespace Mila::Tests::Dnn::Components::Embeddings
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        constexpr int64_t kVocab = 16;
        // Embedding dim must be a multiple of 8: the BF16 kernel vectorizes 8 bf16
        // per int4 load and asserts C % 8 == 0 (FP32 only needs % 4). Kept distinct
        // from kVocab so a row-stride bug cannot hide behind a square shape.
        constexpr int64_t kEmbed = 8;

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
            static constexpr size_t bytes_per_element = 4;
            static constexpr const char* name = "Fp32";
        };

        struct Bf16Precision
        {
            static constexpr TensorDataType value = TensorDataType::BF16;
            static constexpr float atol = 5e-3f;
            static constexpr float rtol = 5e-3f;
            static constexpr size_t bytes_per_element = 2;
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

        for ( dim_t i = 0; i < out.size(); ++i )
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

        for ( dim_t i = 0; i < out.size(); ++i )
        {
            const float tolerance = TypeParam::atol + TypeParam::rtol * std::fabs( expected[ i ] );

            EXPECT_NEAR( out.data()[ i ], expected[ i ], tolerance )
                << "decode gather mismatch at index " << i;
        }
    }

    // Embedding scale (WeightTying.md D5): output = scale * gather. scale = 2 is exact
    // in both FP32 and BF16, so the only error is the gather's own precision rounding.
    // The default scale = 1.0 (identity) path is covered by Forward_GathersEmbeddingRows.
    TYPED_TEST( TokenEmbeddingCudaTests, Forward_WithEmbeddingScale_ScalesOutput )
    {
        const shape_t token_shape{ 2, 3 };
        const int64_t outer = 2 * 3;
        constexpr float kScale = 2.0f;

        auto config = this->config().withEmbeddingScale( kScale );
        auto embedding = std::make_unique<typename TestFixture::EmbeddingType>(
            "token_embedding", config, Device::Cuda( 0 ) );
        embedding->build( BuildContext( token_shape, RuntimeMode::Inference, false ) );
        this->setKnownWte( *embedding );

        auto host_tokens = this->rampTokens( token_shape );
        auto device_tokens = this->toDevice( host_tokens );

        auto& device_out = embedding->forward( device_tokens );
        embedding->synchronize();

        auto out = this->toFloat( device_out );

        auto params = embedding->getParameters();
        auto wte = this->toFloat( *static_cast<typename TestFixture::DeviceTensor*>( params[ 0 ] ) );

        std::vector<float> expected;
        this->referenceGather( host_tokens, wte, outer, kEmbed, expected );

        ASSERT_EQ( out.size(), expected.size() );

        for ( dim_t i = 0; i < out.size(); ++i )
        {
            const float scaled = kScale * expected[ i ];
            const float tolerance = TypeParam::atol + TypeParam::rtol * std::fabs( scaled );

            EXPECT_NEAR( out.data()[ i ], scaled, tolerance )
                << "scaled gather mismatch at index " << i;
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

    // ====================================================================
    // F. FP8 quantized table (TTableQuantization = PerChannelFp8<>, D4 Design B)
    // ====================================================================

    namespace
    {
        // Round-to-nearest-even float -> BF16 bits, matching the converter output
        // format. The blob is authoritative: the quantize kernel and the host
        // reference both start from these exact bits.
        uint16_t floatToBf16Bits( float value )
        {
            const uint32_t bits = std::bit_cast<uint32_t>( value );
            const uint32_t rounding = 0x7FFF + ( ( bits >> 16 ) & 1 );

            return static_cast<uint16_t>( ( bits + rounding ) >> 16 );
        }

        float bf16BitsToFloat( uint16_t bits )
        {
            return std::bit_cast<float>( static_cast<uint32_t>( bits ) << 16 );
        }

        constexpr float kFp8E4M3Max = 448.0f;
    }

    class TokenEmbeddingCudaFp8Tests : public ::testing::Test
    {
    protected:
        using QuantizedEmbeddingType = Mila::Dnn::TokenEmbedding<
            DeviceType::Cuda, TensorDataType::INT32, TensorDataType::BF16,
            Mila::Dnn::Quant::Weight::PerChannelFp8<>>;
        using DeviceTensor = Tensor<TensorDataType::BF16, CudaDeviceMemoryResource>;
        using IndexDeviceTensor = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>;
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        using HostIndex = Tensor<TensorDataType::INT32, CpuMemoryResource>;

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

        std::unique_ptr<QuantizedEmbeddingType> builtEmbedding( const shape_t& token_shape, RuntimeMode mode )
        {
            auto embedding = std::make_unique<QuantizedEmbeddingType>( "token_embedding_fp8", config(), Device::Cuda( 0 ) );
            embedding->build( BuildContext( token_shape, mode, false ) );

            return embedding;
        }

        // BF16 source table as raw bits -- fed to loadParameter as the blob and
        // decoded on the host as the dequantization reference.
        std::vector<uint16_t> bf16Table()
        {
            std::vector<uint16_t> table( static_cast<size_t>( kVocab * kEmbed ) );

            for ( int64_t v = 0; v < kVocab; ++v )
            {
                for ( int64_t c = 0; c < kEmbed; ++c )
                {
                    table[ v * kEmbed + c ] = floatToBf16Bits( wteValue( v, c ) );
                }
            }

            return table;
        }

        // Quantize-on-load: stream the BF16 blob through loadParameter so the op's
        // quantize() path produces the FP8 table + per-row scales, then synchronize
        // (the H2D upload inside quantize is async and the blob is stack-owned here).
        void loadTable( QuantizedEmbeddingType& embedding, const std::vector<uint16_t>& table )
        {
            Serialization::TensorMetadata meta{
                TensorDataType::BF16, shape_t{ kVocab, kEmbed }, table.size() * sizeof( uint16_t ) };
            Serialization::TensorBlobView blob( meta, table.data(), table.size() * sizeof( uint16_t ) );

            embedding.loadParameter( "wte", blob );
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

        // Per-element FP8 error budget against the BF16 source value: E4M3 RNE has
        // half-ulp relative error <= 2^-4 for normals plus scale * 2^-10 absolute in
        // the subnormal range; the BF16 output rounding (2^-8 relative) rides along.
        void expectDequantNear(
            const HostFp32& out, const HostIndex& tokens,
            const std::vector<uint16_t>& table, int64_t outer )
        {
            for ( int64_t r = 0; r < outer; ++r )
            {
                const int32_t idx = tokens.data()[ r ];

                float row_absmax = 0.0f;
                for ( int64_t c = 0; c < kEmbed; ++c )
                    row_absmax = std::max( row_absmax, std::fabs( bf16BitsToFloat( table[ idx * kEmbed + c ] ) ) );

                const float scale = ( row_absmax > 0.0f ) ? ( row_absmax / kFp8E4M3Max ) : 1.0f;

                for ( int64_t c = 0; c < kEmbed; ++c )
                {
                    const float source = bf16BitsToFloat( table[ idx * kEmbed + c ] );
                    const float tolerance = 0.07f * std::fabs( source ) + 0.004f * scale;

                    EXPECT_NEAR( out.data()[ r * kEmbed + c ], source, tolerance )
                        << "fp8 gather-dequant mismatch at row " << r << " channel " << c;
                }
            }
        }

        std::unique_ptr<IExecutionContext> cuda_context_;
    };

    /**
     * @brief A table reloaded from a pre-quantized artifact must gather identically.
     *
     * The existing pre-quantized coverage asserts that the stored BYTES round-trip, which a
     * device-to-host-to-device copy satisfies whether or not the result is usable. What was
     * never asserted is that a component rebuilt from those bytes computes the same thing --
     * and a 12B model loaded this way generated endless thinking tokens while every byte-level
     * check passed. Comparing forward output is the check that has teeth.
     *
     * Bit-identical is the right bar here, not approximate: both components hold the same FP8
     * table and the same scales, and run the same kernel.
     */
    TEST_F( TokenEmbeddingCudaFp8Tests, PreQuantizedReload_GathersIdenticallyToQuantizeOnLoad )
    {
        const shape_t token_shape{ 2, 3 };

        auto quantize_on_load = builtEmbedding( token_shape, RuntimeMode::Inference );
        loadTable( *quantize_on_load, bf16Table() );

        const auto artifact = std::filesystem::temp_directory_path()
            / "mila_tokenembedding_prequant.safetensors";

        {
            Serialization::SafeTensorsWriter writer( artifact );
            quantize_on_load->saveFlatTensors( writer, "temb",
                Serialization::TensorSavePass::Declare );
            writer.beginData();
            quantize_on_load->saveFlatTensors( writer, "temb",
                Serialization::TensorSavePass::Write );
            writer.close();
        }

        auto from_artifact = builtEmbedding( token_shape, RuntimeMode::Inference );

        {
            Serialization::PretrainedModelReader reader( artifact );

            ASSERT_TRUE( reader.hasTensor( "temb.wte" ) );
            ASSERT_TRUE( reader.hasTensor( "temb.wte_scale" ) );

            auto table = reader.readTensorBlob<CpuMemoryResource>( "temb.wte" );
            auto scales = reader.readTensorBlob<CpuMemoryResource>( "temb.wte_scale" );

            from_artifact->loadParameter( "wte", table );
            from_artifact->loadParameter( "wte_scale", scales );
            cuda_context_->synchronize();
        }

        auto host_tokens = rampTokens( token_shape );
        auto device_tokens = toDevice( host_tokens );

        auto& expected_device = quantize_on_load->forward( device_tokens );
        quantize_on_load->synchronize();
        auto expected = toFloat( expected_device );

        auto& actual_device = from_artifact->forward( device_tokens );
        from_artifact->synchronize();
        auto actual = toFloat( actual_device );

        ASSERT_EQ( expected.size(), actual.size() );

        for ( dim_t i = 0; i < expected.size(); ++i )
        {
            EXPECT_EQ( expected.data()[ i ], actual.data()[ i ] )
                << "pre-quantized reload diverges at element " << i;
        }

        std::error_code ignored;
        std::filesystem::remove( artifact, ignored );
    }

    TEST_F( TokenEmbeddingCudaFp8Tests, Forward_MatchesDequantizedReference )
    {
        const shape_t token_shape{ 2, 3 };
        const int64_t outer = 2 * 3;

        auto embedding = builtEmbedding( token_shape, RuntimeMode::Inference );

        auto table = bf16Table();
        loadTable( *embedding, table );

        auto host_tokens = rampTokens( token_shape );
        auto device_tokens = toDevice( host_tokens );

        auto& device_out = embedding->forward( device_tokens );
        embedding->synchronize();

        auto out = toFloat( device_out );

        ASSERT_EQ( out.shape(), ( shape_t{ 2, 3, kEmbed } ) );

        expectDequantNear( out, host_tokens, table, outer );
    }

    // Prefill -> decode runtime axis on the quantized table: outer_size == 1
    // selects the dedicated FP8 decode kernel.
    TEST_F( TokenEmbeddingCudaFp8Tests, Forward_DecodeShapeSingleToken )
    {
        const shape_t prefill_shape{ 2, 3 };
        const shape_t decode_shape{ 2, 1 };
        const int64_t outer = 2 * 1;

        auto embedding = builtEmbedding( prefill_shape, RuntimeMode::Inference );

        auto table = bf16Table();
        loadTable( *embedding, table );

        auto host_tokens = rampTokens( decode_shape );
        auto device_tokens = toDevice( host_tokens );

        auto& device_out = embedding->forward( device_tokens );
        embedding->synchronize();

        auto out = toFloat( device_out );

        ASSERT_EQ( out.shape(), ( shape_t{ 2, 1, kEmbed } ) );

        expectDequantNear( out, host_tokens, table, outer );
    }

    // The tied lm_head adopts both handles (Linear::installSharedWeight); pin the
    // shared surface: FP8 table [vocab, d] plus FP32 row scales [vocab].
    TEST_F( TokenEmbeddingCudaFp8Tests, SharedHandles_ExposeTableAndScales )
    {
        auto embedding = builtEmbedding( shape_t{ 2, 3 }, RuntimeMode::Inference );

        auto table = embedding->getWeightTensorShared();
        ASSERT_NE( table, nullptr );
        EXPECT_EQ( table->getDataType(), TensorDataType::FP8_E4M3 );
        EXPECT_EQ( table->shape(), ( shape_t{ kVocab, kEmbed } ) );

        auto scales = embedding->getWeightScalesTensorShared();
        ASSERT_NE( scales, nullptr );
        EXPECT_EQ( scales->getDataType(), TensorDataType::FP32 );
        EXPECT_EQ( scales->shape(), ( shape_t{ kVocab } ) );

        // Both tensors are memory-accounted as parameters (the owning transformer's
        // tie-aware stats subtract the lm_head side once).
        EXPECT_EQ( embedding->getMemoryStats().device_parameter_bytes,
            table->getStorageSize() + scales->getStorageSize() );
    }

    // The quantized table is inference-only: a training build must fail fast.
    TEST_F( TokenEmbeddingCudaFp8Tests, TrainingBuild_Throws )
    {
        auto embedding = std::make_unique<QuantizedEmbeddingType>( "token_embedding_fp8", config(), Device::Cuda( 0 ) );

        EXPECT_THROW(
            embedding->build( BuildContext( shape_t{ 2, 3 }, RuntimeMode::Training, false ) ),
            std::logic_error );
    }

    TEST_F( TokenEmbeddingCudaFp8Tests, Backward_Throws )
    {
        const shape_t token_shape{ 2, 3 };
        auto embedding = builtEmbedding( token_shape, RuntimeMode::Inference );

        auto device_tokens = toDevice( rampTokens( token_shape ) );
        DeviceTensor output_grad( Device::Cuda( 0 ), shape_t{ 2, 3, kEmbed } );

        EXPECT_THROW( embedding->backward( device_tokens, output_grad ), std::logic_error );
    }

    // ====================================================================
    // H. Host-resident table (TTableResidency = EmbeddingTableResidency::Host)
    //
    // The table is allocated in pinned host memory and the gather kernel reads its
    // rows across PCIe; nothing about the kernel changes. That the arrangement WORKS
    // at all rests on pinned memory being device-addressable under unified virtual
    // addressing, which is a property of the platform rather than of this code -- so
    // it is measured here rather than assumed.
    //
    // The assertions are equalities, not tolerances: a gather is a copy, so the two
    // residencies must agree bit-for-bit and both must agree with the stored bytes.
    // ====================================================================

    namespace
    {
        // The table as its own dtype stores it -- the bytes a converter writes and
        // loadParameter() consumes.
        template<TensorDataType TPrecision>
        std::vector<std::byte> encodedWteTable()
        {
            const size_t count = static_cast<size_t>( kVocab * kEmbed );

            if constexpr ( TPrecision == TensorDataType::BF16 )
            {
                std::vector<std::byte> bytes( count * sizeof( uint16_t ) );
                auto* encoded = reinterpret_cast<uint16_t*>( bytes.data() );

                for ( int64_t v = 0; v < kVocab; ++v )
                {
                    for ( int64_t c = 0; c < kEmbed; ++c )
                    {
                        encoded[ v * kEmbed + c ] = floatToBf16Bits( wteValue( v, c ) );
                    }
                }

                return bytes;
            }
            else
            {
                std::vector<std::byte> bytes( count * sizeof( float ) );
                auto* encoded = reinterpret_cast<float*>( bytes.data() );

                for ( int64_t v = 0; v < kVocab; ++v )
                {
                    for ( int64_t c = 0; c < kEmbed; ++c )
                    {
                        encoded[ v * kEmbed + c ] = wteValue( v, c );
                    }
                }

                return bytes;
            }
        }

        // What the stored bytes decode to. Exact in both precisions, which is what lets
        // the gather reference below carry no tolerance.
        template<TensorDataType TPrecision>
        float decodedWteValue( const std::vector<std::byte>& table, int64_t token, int64_t channel )
        {
            if constexpr ( TPrecision == TensorDataType::BF16 )
            {
                return bf16BitsToFloat(
                    reinterpret_cast<const uint16_t*>( table.data() )[ token * kEmbed + channel ] );
            }
            else
            {
                return reinterpret_cast<const float*>( table.data() )[ token * kEmbed + channel ];
            }
        }

        template<TensorDataType TPrecision, typename TEmbedding>
        void loadWteFromBlob( TEmbedding& embedding, const std::vector<std::byte>& table )
        {
            Serialization::TensorMetadata meta{ TPrecision, shape_t{ kVocab, kEmbed }, table.size() };
            Serialization::TensorBlobView blob( meta, table.data(), table.size() );

            embedding.loadParameter( "wte", blob );
            embedding.synchronize();
        }
    }

    TYPED_TEST( TokenEmbeddingCudaTests, HostResidentTable_GathersIdenticallyToDeviceResident )
    {
        using HostResidentEmbedding = Mila::Dnn::TokenEmbedding<
            DeviceType::Cuda, TensorDataType::INT32, TypeParam::value,
            Mila::Dnn::Quant::Weight::NoWeightQuant, EmbeddingTableResidency::Host>;

        const shape_t token_shape{ 2, 3 };
        const auto table = encodedWteTable<TypeParam::value>();

        auto device_resident = this->builtEmbedding( token_shape, RuntimeMode::Inference );
        loadWteFromBlob<TypeParam::value>( *device_resident, table );

        HostResidentEmbedding host_resident( "token_embedding_host", this->config(), Device::Cuda( 0 ) );
        host_resident.build( BuildContext( token_shape, RuntimeMode::Inference, false ) );
        loadWteFromBlob<TypeParam::value>( host_resident, table );

        auto host_tokens = this->rampTokens( token_shape );
        auto device_tokens = this->toDevice( host_tokens );

        auto& device_resident_out = device_resident->forward( device_tokens );
        device_resident->synchronize();
        auto from_device_table = this->toFloat( device_resident_out );

        auto& host_resident_out = host_resident.forward( device_tokens );
        host_resident.synchronize();
        auto from_host_table = this->toFloat( host_resident_out );

        ASSERT_EQ( from_host_table.shape(), ( shape_t{ 2, 3, kEmbed } ) );
        ASSERT_EQ( from_host_table.size(), from_device_table.size() );

        for ( dim_t i = 0; i < from_host_table.size(); ++i )
        {
            const int32_t token = host_tokens.data()[ i / kEmbed ];
            const float reference = decodedWteValue<TypeParam::value>( table, token, i % kEmbed );

            EXPECT_EQ( from_host_table.data()[ i ], reference )
                << "host-resident gather does not match the stored table at index " << i;

            EXPECT_EQ( from_host_table.data()[ i ], from_device_table.data()[ i ] )
                << "residency changed the gathered value at index " << i;
        }
    }

    // Decode is a separate kernel from prefill, so it needs its own case: a table the
    // prefill kernel reads correctly proves nothing about the single-token path.
    TYPED_TEST( TokenEmbeddingCudaTests, HostResidentTable_DecodeShapeGathersFromHostTable )
    {
        using HostResidentEmbedding = Mila::Dnn::TokenEmbedding<
            DeviceType::Cuda, TensorDataType::INT32, TypeParam::value,
            Mila::Dnn::Quant::Weight::NoWeightQuant, EmbeddingTableResidency::Host>;

        const shape_t prefill_shape{ 2, 3 };
        const shape_t decode_shape{ 2, 1 };
        const auto table = encodedWteTable<TypeParam::value>();

        HostResidentEmbedding host_resident( "token_embedding_host", this->config(), Device::Cuda( 0 ) );
        host_resident.build( BuildContext( prefill_shape, RuntimeMode::Inference, false ) );
        loadWteFromBlob<TypeParam::value>( host_resident, table );

        auto host_tokens = this->rampTokens( decode_shape );
        auto device_tokens = this->toDevice( host_tokens );

        auto& decode_out = host_resident.forward( device_tokens );
        host_resident.synchronize();

        auto out = this->toFloat( decode_out );

        ASSERT_EQ( out.shape(), ( shape_t{ 2, 1, kEmbed } ) );

        for ( dim_t i = 0; i < out.size(); ++i )
        {
            const int32_t token = host_tokens.data()[ i / kEmbed ];
            const float reference = decodedWteValue<TypeParam::value>( table, token, i % kEmbed );

            EXPECT_EQ( out.data()[ i ], reference )
                << "host-resident decode gather mismatch at index " << i;
        }
    }

    // The arrangement's whole purpose: the table's bytes must not be counted against
    // the card, in what was built AND in what the footprint API predicts.
    TYPED_TEST( TokenEmbeddingCudaTests, HostResidentTable_TableBytesAreNotOnTheDevice )
    {
        using HostResidentEmbedding = Mila::Dnn::TokenEmbedding<
            DeviceType::Cuda, TensorDataType::INT32, TypeParam::value,
            Mila::Dnn::Quant::Weight::NoWeightQuant, EmbeddingTableResidency::Host>;

        const shape_t token_shape{ 2, 3 };
        const size_t table_bytes = static_cast<size_t>( kVocab * kEmbed ) * TypeParam::bytes_per_element;

        HostResidentEmbedding host_resident( "token_embedding_host", this->config(), Device::Cuda( 0 ) );
        host_resident.build( BuildContext( token_shape, RuntimeMode::Inference, false ) );

        const auto built = host_resident.getMemoryStats();

        EXPECT_EQ( built.host_parameter_bytes, table_bytes );
        EXPECT_EQ( built.device_parameter_bytes, 0u );

        const auto predicted = host_resident.getRequiredMemory(
            BuildContext( token_shape, RuntimeMode::Inference, false ) );

        EXPECT_EQ( predicted.host_parameter_bytes, table_bytes );
        EXPECT_EQ( predicted.device_parameter_bytes, 0u );

        // Control: the same component at the default residency does put it on the card,
        // so the zero above is the residency and not a broken accounting path.
        auto device_resident = this->builtEmbedding( token_shape, RuntimeMode::Inference );

        EXPECT_EQ( device_resident->getMemoryStats().device_parameter_bytes, table_bytes );
        EXPECT_EQ( device_resident->getMemoryStats().host_parameter_bytes, 0u );
    }

    // What the arrangement costs, at the real Qwen 3.8 geometry rather than the toy one.
    // DISABLED because it allocates ~2.5 GB twice and is a measurement, not an assertion:
    //   MilaTests --gtest_also_run_disabled_tests
    //       --gtest_filter=TokenEmbeddingCudaHostResidentCost.DISABLED_GatherCost
    TEST( TokenEmbeddingCudaHostResidentCost, DISABLED_GatherCost )
    {
        // Qwen 3.8-27B: 248320 x 5120 BF16 = 2.37 GiB, the table section 5 moves off the card.
        constexpr int64_t kQwenVocab = 248320;
        constexpr int64_t kQwenModelDim = 5120;
        constexpr dim_t kPrefillRows = 512;
        constexpr int kDecodeIterations = 500;
        constexpr int kPrefillIterations = 50;

        using DeviceResident = Mila::Dnn::TokenEmbedding<
            DeviceType::Cuda, TensorDataType::INT32, TensorDataType::BF16>;
        using HostResident = Mila::Dnn::TokenEmbedding<
            DeviceType::Cuda, TensorDataType::INT32, TensorDataType::BF16,
            Mila::Dnn::Quant::Weight::NoWeightQuant, EmbeddingTableResidency::Host>;

        if ( getDeviceCount( DeviceType::Cuda ) == 0 )
        {
            GTEST_SKIP() << "CUDA device not available";
        }

        auto config = TokenEmbeddingConfig().withVocabSize( kQwenVocab ).withEmbeddingDim( kQwenModelDim );

        Tensor<TensorDataType::INT32, CpuMemoryResource> host_tokens( Device::Cpu(), shape_t{ 1, kPrefillRows } );

        for ( dim_t i = 0; i < host_tokens.size(); ++i )
        {
            // Spread across the table so the reads are not one cached row.
            host_tokens.data()[ i ] = static_cast<int32_t>( ( i * 7919 ) % kQwenVocab );
        }

        Tensor<TensorDataType::INT32, CudaDeviceMemoryResource> prefill_tokens( Device::Cuda( 0 ), shape_t{ 1, kPrefillRows } );
        copy( host_tokens, prefill_tokens );

        Tensor<TensorDataType::INT32, CudaDeviceMemoryResource> decode_tokens( Device::Cuda( 0 ), shape_t{ 1, 1 } );

        auto timeForward = [] ( auto& embedding, auto& tokens, int iterations ) -> double
            {
                embedding.forward( tokens );
                embedding.synchronize();

                const auto start = std::chrono::steady_clock::now();

                for ( int i = 0; i < iterations; ++i )
                {
                    embedding.forward( tokens );
                }

                embedding.synchronize();
                const auto elapsed = std::chrono::steady_clock::now() - start;

                return std::chrono::duration<double, std::micro>( elapsed ).count() / iterations;
            };

        auto report = [] ( const char* label, double microseconds, double bytes )
            {
                std::cout << label << ": " << microseconds << " us/call, "
                    << ( bytes / microseconds / 1000.0 ) << " GB/s\n";
            };

        const double decode_bytes = static_cast<double>( kQwenModelDim ) * 2.0;
        const double prefill_bytes = static_cast<double>( kPrefillRows ) * kQwenModelDim * 2.0;

        {
            DeviceResident device_resident( "device_resident", config, Device::Cuda( 0 ) );
            device_resident.build( BuildContext( shape_t{ 1, kPrefillRows }, RuntimeMode::Inference, false ) );

            report( "device-resident decode ", timeForward( device_resident, decode_tokens, kDecodeIterations ), decode_bytes );
            report( "device-resident prefill", timeForward( device_resident, prefill_tokens, kPrefillIterations ), prefill_bytes );
        }

        {
            HostResident host_resident( "host_resident", config, Device::Cuda( 0 ) );
            host_resident.build( BuildContext( shape_t{ 1, kPrefillRows }, RuntimeMode::Inference, false ) );

            report( "host-resident decode  ", timeForward( host_resident, decode_tokens, kDecodeIterations ), decode_bytes );
            report( "host-resident prefill ", timeForward( host_resident, prefill_tokens, kPrefillIterations ), prefill_bytes );
        }
    }

    // Inference-only, and refused at build rather than at the first backward: gradient
    // accumulation uses device-scope atomics, which mapped host memory does not support.
    TYPED_TEST( TokenEmbeddingCudaTests, HostResidentTable_TrainingBuildThrows )
    {
        using HostResidentEmbedding = Mila::Dnn::TokenEmbedding<
            DeviceType::Cuda, TensorDataType::INT32, TypeParam::value,
            Mila::Dnn::Quant::Weight::NoWeightQuant, EmbeddingTableResidency::Host>;

        HostResidentEmbedding host_resident( "token_embedding_host", this->config(), Device::Cuda( 0 ) );

        EXPECT_THROW(
            host_resident.build( BuildContext( shape_t{ 2, 3 }, RuntimeMode::Training, false ) ),
            std::logic_error );
    }
}
