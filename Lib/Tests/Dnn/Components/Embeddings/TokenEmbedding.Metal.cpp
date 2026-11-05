/**
 * @file TokenEmbedding.Metal.cpp
 * @brief TokenEmbedding on Metal: each output row is exactly its token's row of the table as stored.
 */

#include <gtest/gtest.h>

import Mila;
import std;
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
