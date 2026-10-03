/**
 * @file TokenEmbedding.Int6.Cuda.cpp
 * @brief The INT6 tied table: the gather returns the exact decoded table rounded once to BF16, the stored tensors
 * reload as written, a tied head computes what a head that quantized the same table itself computes, and a table
 * stored in another format is refused.
 *
 * Compiled only under MILA_ENABLE_CUDA; each test skips if no device is present.
 */

#include <gtest/gtest.h>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>
#include <stdexcept>
#include <system_error>
#include <vector>

import Mila;
import Serialization.Tensor;
import Dnn.Quantization.Weight.Int6Packing;

namespace Mila::Tests::Dnn::Components
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Quant::Weight;

    namespace
    {
        // The Gemma 4 26B-A4B's width; the vocabulary is small and not a multiple of the GEMM's 128-row tile.
        constexpr dim_t kVocab = 1000;
        constexpr dim_t kEmbed = 2816;
        constexpr dim_t kGroup = 32;

        using Int6Embedding = Mila::Dnn::TokenEmbedding<DeviceType::Cuda, TensorDataType::INT32, TensorDataType::BF16,
            PerGroupInt6<32>>;
        using Int6Linear = Mila::Dnn::Linear<DeviceType::Cuda, TensorDataType::BF16, PerGroupInt6<32>>;
        using DeviceBf16 = Tensor<TensorDataType::BF16, CudaDeviceMemoryResource>;
        using IndexDeviceTensor = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>;
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        using HostIndex = Tensor<TensorDataType::INT32, CpuMemoryResource>;

        std::uint16_t toBf16Bits( float value )
        {
            std::uint32_t bits = std::bit_cast<std::uint32_t>( value );
            bits += 0x7FFFu + ( ( bits >> 16 ) & 1u );

            return static_cast<std::uint16_t>( bits >> 16 );
        }

        float fromBf16Bits( std::uint16_t bits )
        {
            return std::bit_cast<float>( static_cast<std::uint32_t>( bits ) << 16 );
        }

        class TokenEmbeddingCudaInt6Tests : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                try
                {
                    context_ = createExecutionContext( Device::Cuda( 0 ) );
                }
                catch ( const std::exception& )
                {
                    GTEST_SKIP() << "CUDA device not available";
                }

                std::mt19937 generator( 20261002u );
                std::normal_distribution<float> distribution( 0.0f, 0.03f );
                table_bits_.resize( static_cast<std::size_t>( kVocab * kEmbed ) );

                for ( auto& bits : table_bits_ )
                    bits = toBf16Bits( distribution( generator ) );

                std::vector<float> source( table_bits_.size() );

                for ( std::size_t index = 0; index < source.size(); ++index )
                    source[ index ] = fromBf16Bits( table_bits_[ index ] );

                codes_.resize( static_cast<std::size_t>( kVocab * packedRowBytesForSixBitCodes( kEmbed ) ) );
                scale_bits_.resize( static_cast<std::size_t>( kVocab * kEmbed / kGroup ) );
                quantizeInt6( source.data(), kVocab, kEmbed, kGroup, codes_.data(), scale_bits_.data() );

                exact_table_.resize( source.size() );
                dequantizeInt6( codes_.data(), scale_bits_.data(), kVocab, kEmbed, kGroup, exact_table_.data() );
            }

            TokenEmbeddingConfig config() const
            {
                return TokenEmbeddingConfig().withVocabSize( kVocab ).withEmbeddingDim( kEmbed );
            }

            std::unique_ptr<Int6Embedding> builtEmbedding( const shape_t& token_shape ) const
            {
                auto embedding = std::make_unique<Int6Embedding>( "temb", config(), Device::Cuda( 0 ) );
                embedding->build( BuildContext( token_shape, RuntimeMode::Inference, false ) );

                return embedding;
            }

            void quantizeOnLoad( Int6Embedding& embedding )
            {
                const std::size_t bytes = table_bits_.size() * sizeof( std::uint16_t );
                Serialization::TensorMetadata meta{ TensorDataType::BF16, shape_t{ kVocab, kEmbed }, bytes };
                Serialization::TensorBlobView blob( meta, table_bits_.data(), bytes );

                embedding.loadParameter( "wte", blob );
                context_->synchronize();
            }

            std::vector<std::int32_t> tokens( dim_t count ) const
            {
                std::vector<std::int32_t> ids( static_cast<std::size_t>( count ) );

                for ( dim_t index = 0; index < count; ++index )
                    ids[ index ] = static_cast<std::int32_t>( ( index * 389 + 7 ) % kVocab );

                ids.back() = static_cast<std::int32_t>( kVocab - 1 );

                return ids;
            }

            std::vector<float> gather( Int6Embedding& embedding, const std::vector<std::int32_t>& ids, dim_t batch, dim_t length )
            {
                HostIndex host( Device::Cpu(), shape_t{ batch, length } );
                std::copy( ids.begin(), ids.end(), host.data() );

                IndexDeviceTensor device( Device::Cuda( 0 ), host.shape() );
                copy( host, device, context_.get() );
                context_->synchronize();

                auto& output = embedding.forward( device );
                embedding.synchronize();

                auto host_output = toHost<TensorDataType::FP32>( output, context_.get() );
                context_->synchronize();

                return std::vector<float>( host_output.data(), host_output.data() + host_output.size() );
            }

            void expectExactRows( const std::vector<float>& output, const std::vector<std::int32_t>& ids ) const
            {
                for ( std::size_t token = 0; token < ids.size(); ++token )
                {
                    for ( dim_t column = 0; column < kEmbed; ++column )
                    {
                        const float expected = fromBf16Bits( toBf16Bits(
                            exact_table_[ static_cast<std::size_t>( ids[ token ] ) * kEmbed + column ] ) );

                        ASSERT_EQ( output[ token * kEmbed + column ], expected )
                            << "token " << token << " (id " << ids[ token ] << ") column " << column;
                    }
                }
            }

            std::vector<float> headForward( Int6Linear& head, const std::vector<float>& input, dim_t rows )
            {
                HostFp32 host_input( Device::Cpu(), shape_t{ rows, kEmbed } );
                std::copy( input.begin(), input.end(), host_input.data() );

                DeviceBf16 device_input( Device::Cuda( 0 ), shape_t{ rows, kEmbed } );
                copy( host_input, device_input, context_.get() );
                context_->synchronize();

                auto& output = head.forward( device_input );
                head.synchronize();

                auto host_output = toHost<TensorDataType::FP32>( output, context_.get() );
                context_->synchronize();

                return std::vector<float>( host_output.data(), host_output.data() + host_output.size() );
            }

            std::unique_ptr<IExecutionContext> context_;
            std::vector<std::uint16_t> table_bits_;
            std::vector<std::uint8_t> codes_;
            std::vector<std::uint16_t> scale_bits_;
            std::vector<float> exact_table_;
        };
    }

    TEST_F( TokenEmbeddingCudaInt6Tests, PrefillGatherIsTheExactTableRoundedOnce )
    {
        auto embedding = builtEmbedding( shape_t{ 2, 7 } );
        quantizeOnLoad( *embedding );

        const auto ids = tokens( 14 );
        expectExactRows( gather( *embedding, ids, 2, 7 ), ids );
    }

    TEST_F( TokenEmbeddingCudaInt6Tests, DecodeGatherIsTheExactTableRoundedOnce )
    {
        auto embedding = builtEmbedding( shape_t{ 2, 7 } );
        quantizeOnLoad( *embedding );

        const auto ids = tokens( 2 );
        expectExactRows( gather( *embedding, ids, 2, 1 ), ids );
    }

    TEST_F( TokenEmbeddingCudaInt6Tests, SharedHandlesHaveTheHeadsShapes )
    {
        auto embedding = builtEmbedding( shape_t{ 1, 4 } );

        auto table = embedding->getWeightTensorShared();
        ASSERT_NE( table, nullptr );
        EXPECT_EQ( table->getDataType(), TensorDataType::UINT8 );
        EXPECT_EQ( table->shape(), ( shape_t{ kVocab, kEmbed * 3 / 4 } ) );

        auto scales = embedding->getWeightScalesTensorShared();
        ASSERT_NE( scales, nullptr );
        EXPECT_EQ( scales->getDataType(), TensorDataType::FP16 );
        EXPECT_EQ( scales->shape(), ( shape_t{ kVocab, kEmbed / kGroup } ) );
    }

    // The tensors an export writes are the codec's, and they reload to the same gather.
    TEST_F( TokenEmbeddingCudaInt6Tests, ExportedTensorsAreTheCodecsAndReloadAsWritten )
    {
        auto quantized = builtEmbedding( shape_t{ 2, 7 } );
        quantizeOnLoad( *quantized );

        const auto weights = std::filesystem::temp_directory_path() / "mila_tokenembedding_int6.safetensors";

        {
            Serialization::SafeTensorsWriter writer( weights );
            quantized->saveFlatTensors( writer, "temb", Serialization::TensorSavePass::Declare );
            writer.beginData();
            quantized->saveFlatTensors( writer, "temb", Serialization::TensorSavePass::Write );
            writer.close();
        }

        auto reloaded = builtEmbedding( shape_t{ 2, 7 } );

        {
            Serialization::WeightsReader reader( weights );

            auto table = reader.readTensorBlob<CpuMemoryResource>( "temb.wte" );
            auto scales = reader.readTensorBlob<CpuMemoryResource>( "temb.wte_scale" );

            ASSERT_EQ( table.getMetadata().dtype, TensorDataType::UINT8 );
            ASSERT_EQ( table.getMetadata().shape, ( shape_t{ kVocab, kEmbed * 3 / 4 } ) );
            ASSERT_EQ( scales.getMetadata().dtype, TensorDataType::FP16 );

            EXPECT_EQ( std::vector<std::uint8_t>( static_cast<const std::uint8_t*>( table.data() ),
                static_cast<const std::uint8_t*>( table.data() ) + codes_.size() ), codes_ );
            EXPECT_EQ( std::vector<std::uint16_t>( static_cast<const std::uint16_t*>( scales.data() ),
                static_cast<const std::uint16_t*>( scales.data() ) + scale_bits_.size() ), scale_bits_ );

            reloaded->loadParameter( "wte", table );
            reloaded->loadParameter( "wte_scale", scales );
            context_->synchronize();
        }

        const auto ids = tokens( 14 );
        EXPECT_EQ( gather( *reloaded, ids, 2, 7 ), gather( *quantized, ids, 2, 7 ) );

        std::error_code ignored;
        std::filesystem::remove( weights, ignored );
    }

    // Weights exported when the table was FP8 per row carry it as FP8_E4M3 at the logical shape. Read as BF16 they
    // would quantize garbage, so the load refuses them.
    TEST_F( TokenEmbeddingCudaInt6Tests, ATableStoredInAnotherFormatIsRefused )
    {
        auto embedding = builtEmbedding( shape_t{ 1, 4 } );

        std::vector<std::uint8_t> fp8_table( static_cast<std::size_t>( kVocab * kEmbed ), 0x38 );
        Serialization::TensorMetadata meta{ TensorDataType::FP8_E4M3, shape_t{ kVocab, kEmbed }, fp8_table.size() };
        Serialization::TensorBlobView blob( meta, fp8_table.data(), fp8_table.size() );

        EXPECT_THROW( embedding->loadParameter( "wte", blob ), std::invalid_argument );
    }

    // A head that adopts the embedding's tensors computes exactly what a head that quantized the same table through
    // its own load computes, at one row (the decode matvec) and at many (the batched GEMM).
    TEST_F( TokenEmbeddingCudaInt6Tests, TiedHeadMatchesADirectlyQuantizedHead )
    {
        auto embedding = builtEmbedding( shape_t{ 1, 4 } );
        quantizeOnLoad( *embedding );

        LinearConfig head_config( kEmbed, kVocab );
        head_config.withBias( false );

        for ( const dim_t rows : { dim_t{ 1 }, dim_t{ 64 } } )
        {
            Int6Linear direct( "lm_head_direct", head_config, Device::Cuda( 0 ) );
            direct.build( BuildContext( shape_t{ rows, kEmbed }, RuntimeMode::Inference, false ) );

            const std::size_t bytes = table_bits_.size() * sizeof( std::uint16_t );
            Serialization::TensorMetadata meta{ TensorDataType::BF16, shape_t{ kVocab, kEmbed }, bytes };
            Serialization::TensorBlobView blob( meta, table_bits_.data(), bytes );
            direct.loadParameter( "weight", blob );
            direct.synchronize();

            Int6Linear tied( "lm_head_tied", head_config, Device::Cuda( 0 ) );
            tied.installSharedWeight( embedding->getWeightTensorShared(), embedding->getWeightScalesTensorShared() );
            tied.build( BuildContext( shape_t{ rows, kEmbed }, RuntimeMode::Inference, false ) );

            auto tied_parameters = tied.getParameters();
            ASSERT_EQ( tied_parameters.size(), 1u );
            EXPECT_EQ( tied_parameters[ 0 ], static_cast<ITensor*>( embedding->getWeightTensorShared().get() ) );

            std::mt19937 generator( 7u + static_cast<unsigned>( rows ) );
            std::normal_distribution<float> distribution( 0.0f, 1.0f );
            std::vector<float> input( static_cast<std::size_t>( rows * kEmbed ) );

            for ( auto& value : input )
                value = fromBf16Bits( toBf16Bits( distribution( generator ) ) );

            EXPECT_EQ( headForward( tied, input, rows ), headForward( direct, input, rows ) ) << rows << " rows";
        }
    }
}
