/**
 * @file Llama.Block.Metal.cpp
 * @brief A Llama block on Metal on random weights: prefilling N tokens equals prefilling N - 1 and decoding one.
 *
 * Both ways reach the last token through every one of Llama's block operations and through the KV cache, by different paths: prefill by
 * TensorOps, decode by own kernels. A wrong cache or position shows here before Phase 4 has a whole model to read.
 */

#include <gtest/gtest.h>

import Mila;
import std;
import Compute.GqaWorkspace;
import Dnn.Components.LlamaTransformer;
import Serialization.Tensor;

#include "../../MetalReference.h"

namespace Mila::Tests::Dnn::Components
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Tests::Dnn::MetalReference;

    namespace
    {
        constexpr dim_t kModelDim = 64;
        constexpr dim_t kHeads = 4;
        constexpr dim_t kKvHeads = 2;
        constexpr dim_t kHeadSize = kModelDim / kHeads;
        constexpr dim_t kContext = 64;
        constexpr dim_t kChunk = 16;

        template<TensorDataType TPrecision>
        inline constexpr double kBlockBar = TPrecision == TensorDataType::FP16 ? 2e-3 : 2e-2;

        template<typename TPrecisionConstant>
        class LlamaBlockMetal : public ::testing::Test
        {
        };

        TYPED_TEST_SUITE( LlamaBlockMetal, Precisions, PrecisionNames );

        template<TensorDataType TPrecision>
        struct Fixture
        {
            using Block = LlamaBlock<DeviceType::Metal, TPrecision>;
            using TensorType = Tensor<TPrecision, MetalMemoryResource>;

            Block block;
            GqaWorkspace<DeviceType::Metal, TPrecision> workspace;
            std::vector<float> values;

            explicit Fixture( dim_t tokens )
                : block( "block",
                      LlamaConfig( kModelDim, 1 ).withNumHeads( kHeads ).withNumKVHeads( kKvHeads )
                          .withHiddenDimension( 128 ).withMaxSequenceLength( kContext ).withVocabularyLength( 32 ),
                      Device::Metal( 0 ) ),
                  values( randomValues( static_cast<std::size_t>( tokens * kModelDim ), 101, 1.0f ) )
            {
                block.build( BuildContext( shape_t{ 1, kContext, kModelDim }, RuntimeMode::Inference, true ).withPrefillSize( kChunk ) );

                workspace = makeGqaWorkspace<DeviceType::Metal, TPrecision>(
                    Device::Metal( 0 ), 1, kHeads, kHeadSize, kContext, kChunk, kContext, "workspace." );

                block.setState( workspace.state() );
            }

            [[nodiscard]] TensorType tokensAt( dim_t first, dim_t count ) const
            {
                return metalTensor<TPrecision>(
                    std::span<const float>( values ).subspan( static_cast<std::size_t>( first * kModelDim ), static_cast<std::size_t>( count * kModelDim ) ),
                    shape_t{ 1, count, kModelDim } );
            }

            /// Prefill [first, first + count), and read the block's output for those tokens.
            std::vector<float> prefill( dim_t first, dim_t count )
            {
                TensorType input = tokensAt( first, count );

                const TensorType& output = block.prefill( input, first );
                block.synchronize();

                return readFloats<TPrecision>( output );
            }

            std::vector<float> decode( dim_t position )
            {
                TensorType input = tokensAt( position, 1 );

                block.getExecutionContext()->setDecodePosition( position );

                const TensorType& output = block.decode( input, position );
                block.synchronize();

                return readFloats<TPrecision>( output );
            }
        };

        template<TensorDataType TPrecision>
        double difference( std::span<const float> first, std::span<const float> second )
        {
            std::vector<double> reference( first.begin(), first.end() );

            return relativeError( second, reference );
        }
    }

    TYPED_TEST( LlamaBlockMetal, ADecodedTokenEqualsThatTokenPrefilled )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Fixture<kPrecision> fixture( kChunk );

        const std::vector<float> whole = fixture.prefill( 0, kChunk );
        const std::span<const float> last_of_whole = std::span<const float>( whole ).last( kModelDim );

        fixture.prefill( 0, kChunk - 1 );
        const std::vector<float> decoded = fixture.decode( kChunk - 1 );

        EXPECT_LE( difference<kPrecision>( last_of_whole, decoded ), kBlockBar<kPrecision> );
    }

    TYPED_TEST( LlamaBlockMetal, ARunOfDecodesEqualsAPrefillOfThemAll )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        constexpr dim_t kWarm = 5;

        Fixture<kPrecision> fixture( kChunk );

        const std::vector<float> whole = fixture.prefill( 0, kChunk );

        fixture.prefill( 0, kWarm );

        for ( dim_t position = kWarm; position < kChunk; ++position )
        {
            const std::vector<float> decoded = fixture.decode( position );
            const std::span<const float> prefilled = std::span<const float>( whole ).subspan(
                static_cast<std::size_t>( position * kModelDim ), static_cast<std::size_t>( kModelDim ) );

            EXPECT_LE( difference<kPrecision>( prefilled, decoded ), kBlockBar<kPrecision> ) << "position " << position;
        }
    }

    TYPED_TEST( LlamaBlockMetal, TwoChunksEqualOneChunkFollowedByTheRest )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Fixture<kPrecision> fixture( 24 );

        const std::vector<float> first = fixture.prefill( 0, kChunk );
        const std::vector<float> second = fixture.prefill( kChunk, 8 );

        fixture.prefill( 0, kChunk );

        for ( dim_t position = kChunk; position < 24; ++position )
        {
            const std::vector<float> decoded = fixture.decode( position );
            const std::span<const float> prefilled = std::span<const float>( second ).subspan(
                static_cast<std::size_t>( ( position - kChunk ) * kModelDim ), static_cast<std::size_t>( kModelDim ) );

            EXPECT_LE( difference<kPrecision>( prefilled, decoded ), kBlockBar<kPrecision> ) << "position " << position;
        }
    }
}
