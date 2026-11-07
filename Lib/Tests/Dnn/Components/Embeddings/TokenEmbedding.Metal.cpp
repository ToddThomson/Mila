/**
 * @file TokenEmbedding.Metal.cpp
 * @brief TokenEmbedding on Metal: each output row is exactly its token's row of the table as stored, unquantized or six bits.
 */

#include <gtest/gtest.h>

import Mila;
import std;
import Dnn.Quantization.Weight.Int6Packing;
import Serialization.Tensor;

#include "../MetalReference.h"

namespace Mila::Tests::Dnn::Components
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Tests::Dnn::MetalReference;

    namespace
    {
        constexpr dim_t kVocabulary = 37;

        template<typename TPrecisionConstant>
        class TokenEmbeddingMetal : public ::testing::Test
        {
        };

        TYPED_TEST_SUITE( TokenEmbeddingMetal, Precisions, PrecisionNames );

        template<TensorDataType TPrecision>
        void expectTableRows( dim_t embedding_dim, const shape_t& shape )
        {
            using Embedding = TokenEmbedding<DeviceType::Metal, TensorDataType::INT32, TPrecision>;

            const std::vector<std::uint16_t> table = randomBf16( static_cast<std::size_t>( kVocabulary * embedding_dim ), 21, 0.5f );

            Embedding embedding( "embedding", TokenEmbeddingConfig().withVocabSize( kVocabulary ).withEmbeddingDim( embedding_dim ), Device::Metal( 0 ) );
            embedding.build( BuildContext( shape, RuntimeMode::Inference, false ) );
            embedding.loadParameter( "wte", blobOver( table, shape_t{ kVocabulary, embedding_dim } ) );

            const dim_t tokens = shape[0] * shape[1];
            std::vector<std::int32_t> ids( static_cast<std::size_t>( tokens ) );

            std::mt19937 engine( 22 );
            std::ranges::generate( ids, [&] { return static_cast<std::int32_t>( engine() % kVocabulary ); } );

            Tensor<TensorDataType::INT32, MetalMemoryResource> input( Device::Metal( 0 ), shape );
            fill( input, std::span<const std::int32_t>( ids ) );

            const auto& output = embedding.forward( input );
            embedding.synchronize();

            const std::vector<float> actual = readFloats<TPrecision>( output );

            ASSERT_EQ( actual.size(), static_cast<std::size_t>( tokens * embedding_dim ) );

            for ( dim_t t = 0; t < tokens; ++t )
            {
                for ( dim_t c = 0; c < embedding_dim; ++c )
                {
                    const std::uint16_t bits = table[static_cast<std::size_t>( ids[static_cast<std::size_t>( t )] * embedding_dim + c )];

                    EXPECT_EQ( static_cast<double>( actual[static_cast<std::size_t>( t * embedding_dim + c )] ),
                        asStored<TPrecision>( fromBf16Bits( bits ) ) ) << "token " << t << ", channel " << c;
                }
            }
        }
    }

    TYPED_TEST( TokenEmbeddingMetal, EachOutputRowIsItsTokensRowOfTheTable )
    {
        expectTableRows<TypeParam::value>( 16, shape_t{ 2, 5 } );
    }

    TYPED_TEST( TokenEmbeddingMetal, ADecodesOneTokenAndAWideBatch )
    {
        expectTableRows<TypeParam::value>( 12, shape_t{ 1, 1 } );
        expectTableRows<TypeParam::value>( 12, shape_t{ 3, 7 } );
    }

    TYPED_TEST( TokenEmbeddingMetal, AppliesItsEmbeddingScale )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        TokenEmbedding<DeviceType::Metal, TensorDataType::INT32, kPrecision> embedding(
            "embedding", TokenEmbeddingConfig().withVocabSize( 2 ).withEmbeddingDim( 4 ).withEmbeddingScale( 4.0f ), Device::Metal( 0 ) );
        embedding.build( BuildContext( shape_t{ 1, 1 }, RuntimeMode::Inference, false ) );

        const std::vector<std::uint16_t> table{
            toBf16Bits( 0.0f ), toBf16Bits( 0.0f ), toBf16Bits( 0.0f ), toBf16Bits( 0.0f ),
            toBf16Bits( 0.5f ), toBf16Bits( -2.0f ), toBf16Bits( 1.0f ), toBf16Bits( 0.25f ) };
        embedding.loadParameter( "wte", blobOver( table, shape_t{ 2, 4 } ) );

        Tensor<TensorDataType::INT32, MetalMemoryResource> input( Device::Metal( 0 ), shape_t{ 1, 1 } );
        const std::array<std::int32_t, 1> id{ 1 };
        fill( input, std::span<const std::int32_t>( id ) );

        const auto& output = embedding.forward( input );
        embedding.synchronize();

        EXPECT_EQ( readFloats<kPrecision>( output ), ( std::vector<float>{ 2.0f, -8.0f, 4.0f, 1.0f } ) );
    }

    namespace
    {
        TensorBlobView codesBlob( std::span<const std::uint8_t> bytes, const shape_t& shape )
        {
            TensorMetadata metadata;
            metadata.dtype = TensorDataType::UINT8;
            metadata.shape = shape;
            metadata.total_bytes = bytes.size_bytes();

            return TensorBlobView( std::move( metadata ), bytes.data(), bytes.size_bytes() );
        }

        /// A six-bit table of `vocabulary` rows gathered at `ids`: each output against the library's own decode of that row,
        /// times `scale`, a power of two so that it rounds nothing.
        template<TensorDataType TPrecision>
        void expectSixBitRows( dim_t vocabulary, dim_t embedding_dim, const std::vector<std::int32_t>& ids, float scale )
        {
            using namespace Mila::Dnn::Quant::Weight;
            using Embedding = TokenEmbedding<DeviceType::Metal, TensorDataType::INT32, TPrecision, PerGroupInt6<32>>;

            const std::vector<float> table = randomValues( static_cast<std::size_t>( vocabulary * embedding_dim ), 23, 0.05f );
            std::vector<std::uint8_t> codes( static_cast<std::size_t>( vocabulary * packedRowBytesForSixBitCodes( embedding_dim ) ) );
            std::vector<std::uint16_t> scales( static_cast<std::size_t>( vocabulary * embedding_dim / 32 ) );
            std::vector<float> decoded( table.size() );

            quantizeInt6( table.data(), vocabulary, embedding_dim, 32, codes.data(), scales.data() );
            dequantizeInt6( codes.data(), scales.data(), vocabulary, embedding_dim, 32, decoded.data() );

            const shape_t shape{ 1, static_cast<dim_t>( ids.size() ) };

            Embedding embedding( "embedding", TokenEmbeddingConfig().withVocabSize( vocabulary ).withEmbeddingDim( embedding_dim )
                .withEmbeddingScale( scale ), Device::Metal( 0 ) );
            embedding.build( BuildContext( shape, RuntimeMode::Inference, false ) );
            embedding.loadParameter( "wte", codesBlob( codes, shape_t{ vocabulary, packedRowBytesForSixBitCodes( embedding_dim ) } ) );
            embedding.loadParameter( "wte_scale", blobOver( scales, shape_t{ vocabulary, embedding_dim / 32 }, TensorDataType::FP16 ) );

            Tensor<TensorDataType::INT32, MetalMemoryResource> input( Device::Metal( 0 ), shape );
            fill( input, std::span<const std::int32_t>( ids ) );

            const auto& output = embedding.forward( input );
            embedding.synchronize();

            const std::vector<float> actual = readFloats<TPrecision>( output );

            for ( std::size_t t = 0; t < ids.size(); ++t )
            {
                for ( dim_t c = 0; c < embedding_dim; ++c )
                {
                    const float value = decoded[static_cast<std::size_t>( ids[t] * embedding_dim + c )];

                    ASSERT_EQ( static_cast<double>( actual[static_cast<std::size_t>( t * embedding_dim + c )] ),
                        asStored<TPrecision>( value ) * scale ) << "token " << t << " (id " << ids[t] << "), channel " << c;
                }
            }
        }
    }

    TYPED_TEST( TokenEmbeddingMetal, ASixBitTableGathersEachRowAsTheLibraryDecodesIt )
    {
        // Ids from either end of the table, and one repeated.
        expectSixBitRows<TypeParam::value>( 4099, 128, { 0, 4098, 17, 17, 2048, 1 }, 1.0f );
        expectSixBitRows<TypeParam::value>( 64, 3840, { 63, 0, 5 }, 1.0f );
    }

    TYPED_TEST( TokenEmbeddingMetal, ASixBitTableAppliesTheEmbeddingScale )
    {
        expectSixBitRows<TypeParam::value>( 64, 96, { 3, 60 }, 4.0f );
    }

    TYPED_TEST( TokenEmbeddingMetal, ASixBitTableRefusesAFullPrecisionTableAndNamesIt )
    {
        using namespace Mila::Dnn::Quant::Weight;
        using Embedding = TokenEmbedding<DeviceType::Metal, TensorDataType::INT32, TypeParam::value, PerGroupInt6<32>>;

        Embedding embedding( "embedding", TokenEmbeddingConfig().withVocabSize( 8 ).withEmbeddingDim( 32 ), Device::Metal( 0 ) );
        embedding.build( BuildContext( shape_t{ 1, 1 }, RuntimeMode::Inference, false ) );

        const std::vector<std::uint16_t> bits = randomBf16( 8 * 32, 5, 0.5f );
        const TensorDataType precision = TypeParam::value;

        try
        {
            embedding.loadParameter( "wte", blobOver( bits, shape_t{ 8, 32 }, precision ) );
            FAIL() << "A full-precision table was taken";
        }
        catch ( const std::invalid_argument& error )
        {
            EXPECT_NE( std::string( error.what() ).find( "embedding" ), std::string::npos ) << error.what();
        }
    }

    TEST( TokenEmbeddingMetalRows, OnlyFp16AndBf16Exist )
    {
        using namespace Mila::Dnn::Quant::Weight;

        EXPECT_FALSE( ( OperationSupported<OperationType::TokenEmbeddingOp, DeviceType::Metal, TensorDataType::FP32, NoWeightQuant> ) );
        EXPECT_FALSE( ( OperationSupported<OperationType::RmsNormOp, DeviceType::Metal, TensorDataType::FP32> ) );
        EXPECT_FALSE( ( OperationSupported<OperationType::RopeOp, DeviceType::Metal, TensorDataType::FP32> ) );
        EXPECT_FALSE( ( OperationSupported<OperationType::SwigluOp, DeviceType::Metal, TensorDataType::FP32> ) );
        EXPECT_FALSE( ( OperationSupported<OperationType::ResidualOp, DeviceType::Metal, TensorDataType::FP32> ) );
    }
}
