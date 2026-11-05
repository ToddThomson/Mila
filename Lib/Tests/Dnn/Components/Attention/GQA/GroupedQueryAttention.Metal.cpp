/**
 * @file GroupedQueryAttention.Metal.cpp
 * @brief Grouped-query attention on Metal, prefill and decode, against a double-precision reference over the same cache.
 *
 * The bars are the gate's for attention, which are wider than a Linear's: a score's rounding goes through an exponential.
 */

#include <gtest/gtest.h>

import Mila;
import std;
import Compute.GqaWorkspace;
import Dnn.Components.Gqa;
import Serialization.Tensor;

#include "../../MetalReference.h"

namespace Mila::Tests::Dnn::Components
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Tests::Dnn::MetalReference;

    namespace
    {
        template<TensorDataType TPrecision>
        inline constexpr double kAttentionBar = TPrecision == TensorDataType::FP16 ? 2e-3 : 2e-2;

        template<typename TPrecisionConstant>
        class GqaMetal : public ::testing::Test
        {
        };

        TYPED_TEST_SUITE( GqaMetal, Precisions, PrecisionNames );

        struct Geometry
        {
            dim_t batches{ 1 };
            dim_t heads{ 4 };
            dim_t kv_heads{ 2 };
            dim_t head_size{ 16 };
            dim_t context{ 64 };
            dim_t chunk{ 16 };
            dim_t window{ 0 };
        };

        /// A stack of one layer, with what the reference needs: everything the cache was given.
        template<TensorDataType TPrecision>
        struct Stack
        {
            using TensorType = Tensor<TPrecision, MetalMemoryResource>;

            Geometry shape;
            GroupedQueryAttention<DeviceType::Metal, TPrecision> attention;
            GqaWorkspace<DeviceType::Metal, TPrecision> workspace;
            std::vector<double> keys;
            std::vector<double> values;
            std::uint32_t seed{ 71 };

            explicit Stack( const Geometry& geometry )
                : shape( geometry ),
                  attention( "attention",
                      GqaConfig( geometry.heads * geometry.head_size, geometry.heads, geometry.kv_heads ).withWindow( geometry.window ),
                      Device::Metal( 0 ) ),
                  keys( static_cast<std::size_t>( geometry.batches * geometry.kv_heads * geometry.context * geometry.head_size ) ),
                  values( keys.size() )
            {
                const dim_t packed = ( geometry.heads + 2 * geometry.kv_heads ) * geometry.head_size;

                attention.build( BuildContext( shape_t{ geometry.batches, geometry.context, packed }, RuntimeMode::Inference, false )
                    .withPrefillSize( geometry.chunk ) );

                workspace = makeGqaWorkspace<DeviceType::Metal, TPrecision>(
                    Device::Metal( 0 ), geometry.batches, geometry.heads, geometry.head_size, geometry.context, geometry.chunk, geometry.context, "workspace." );

                attention.setState( workspace.state() );
            }

            [[nodiscard]] double scale() const
            {
                return static_cast<double>( attention.getConfig().getAttentionScale() );
            }

            /// Keep what this chunk writes to the cache, at the rows the op must write it to.
            void remember( std::span<const double> chunk_keys, std::span<const double> chunk_values, dim_t tokens, dim_t first_position )
            {
                for ( dim_t b = 0; b < shape.batches; ++b )
                {
                    for ( dim_t t = 0; t < tokens; ++t )
                    {
                        for ( dim_t h = 0; h < shape.kv_heads; ++h )
                        {
                            for ( dim_t s = 0; s < shape.head_size; ++s )
                            {
                                const auto from = static_cast<std::size_t>( ( ( b * tokens + t ) * shape.kv_heads + h ) * shape.head_size + s );
                                const auto to = static_cast<std::size_t>(
                                    ( ( b * shape.kv_heads + h ) * shape.context + first_position + t ) * shape.head_size + s );

                                keys[to] = chunk_keys[from];
                                values[to] = chunk_values[from];
                            }
                        }
                    }
                }
            }

            /// Run a chunk or one decoded token, and return its error against the reference.
            double errorOfStep( dim_t tokens, dim_t first_position, bool decode )
            {
                const auto q_count = static_cast<std::size_t>( shape.batches * tokens * shape.heads * shape.head_size );
                const auto kv_count = static_cast<std::size_t>( shape.batches * tokens * shape.kv_heads * shape.head_size );

                const std::vector<float> q_values = randomValues( q_count, seed++, 0.5f );
                const std::vector<float> k_values = randomValues( kv_count, seed++, 0.5f );
                const std::vector<float> v_values = randomValues( kv_count, seed++, 0.5f );

                TensorType q = metalTensor<TPrecision>( q_values, shape_t{ shape.batches, tokens, shape.heads * shape.head_size } );
                TensorType k = metalTensor<TPrecision>( k_values, shape_t{ shape.batches, tokens, shape.kv_heads * shape.head_size } );
                TensorType v = metalTensor<TPrecision>( v_values, shape_t{ shape.batches, tokens, shape.kv_heads * shape.head_size } );

                const std::vector<double> stored_q = asStored<TPrecision>( std::span<const float>( q_values ) );
                const std::vector<double> stored_k = asStored<TPrecision>( std::span<const float>( k_values ) );
                const std::vector<double> stored_v = asStored<TPrecision>( std::span<const float>( v_values ) );

                remember( stored_k, stored_v, tokens, first_position );

                TensorType* output = nullptr;

                if ( decode )
                {
                    attention.getExecutionContext()->setDecodePosition( first_position );
                    output = &attention.decode( q, k, v, first_position );
                } else {
                    output = &attention.prefill( q, k, v, first_position );
                }

                attention.synchronize();

                return relativeError( readFloats<TPrecision>( *output ), reference( stored_q, tokens, first_position ) );
            }

            [[nodiscard]] std::vector<double> reference( const std::vector<double>& q, dim_t tokens, dim_t first_position ) const
            {
                const dim_t group_size = shape.heads / shape.kv_heads;

                std::vector<double> result( q.size() );
                std::vector<double> scores( static_cast<std::size_t>( first_position + tokens ) );

                for ( dim_t b = 0; b < shape.batches; ++b )
                {
                    for ( dim_t t = 0; t < tokens; ++t )
                    {
                        const dim_t position = first_position + t;
                        const dim_t first_key = shape.window > 0 ? std::max( dim_t{ 0 }, position - shape.window + 1 ) : 0;

                        for ( dim_t h = 0; h < shape.heads; ++h )
                        {
                            const auto query = static_cast<std::size_t>( ( ( b * tokens + t ) * shape.heads + h ) * shape.head_size );
                            const auto cache = static_cast<std::size_t>( ( b * shape.kv_heads + h / group_size ) * shape.context * shape.head_size );

                            double largest = -std::numeric_limits<double>::infinity();

                            for ( dim_t j = first_key; j <= position; ++j )
                            {
                                double score = 0.0;

                                for ( dim_t s = 0; s < shape.head_size; ++s )
                                {
                                    score += q[query + static_cast<std::size_t>( s )]
                                        * keys[cache + static_cast<std::size_t>( j * shape.head_size + s )];
                                }

                                scores[static_cast<std::size_t>( j )] = score * scale();
                                largest = std::max( largest, scores[static_cast<std::size_t>( j )] );
                            }

                            double total = 0.0;

                            for ( dim_t j = first_key; j <= position; ++j )
                            {
                                total += std::exp( scores[static_cast<std::size_t>( j )] - largest );
                            }

                            for ( dim_t s = 0; s < shape.head_size; ++s )
                            {
                                double sum = 0.0;

                                for ( dim_t j = first_key; j <= position; ++j )
                                {
                                    sum += std::exp( scores[static_cast<std::size_t>( j )] - largest ) / total
                                        * values[cache + static_cast<std::size_t>( j * shape.head_size + s )];
                                }

                                result[query + static_cast<std::size_t>( s )] = sum;
                            }
                        }
                    }
                }

                return result;
            }
        };

        template<TensorDataType TPrecision>
        void expectChunksThenDecodes( const Geometry& geometry, dim_t chunks, dim_t decodes )
        {
            Stack<TPrecision> stack( geometry );

            dim_t position = 0;

            for ( dim_t c = 0; c < chunks; ++c )
            {
                EXPECT_LE( stack.errorOfStep( geometry.chunk, position, false ), kAttentionBar<TPrecision> ) << "chunk " << c;
                position += geometry.chunk;
            }

            for ( dim_t d = 0; d < decodes; ++d )
            {
                EXPECT_LE( stack.errorOfStep( 1, position, true ), kAttentionBar<TPrecision> ) << "decode at " << position;
                ++position;
            }
        }
    }

    TYPED_TEST( GqaMetal, AChunkThenTokensMatchTheReference )
    {
        expectChunksThenDecodes<TypeParam::value>( Geometry{}, 2, 4 );
    }

    TYPED_TEST( GqaMetal, ADecodeWithNothingBeforeItAttendsOnlyToItself )
    {
        expectChunksThenDecodes<TypeParam::value>( Geometry{}, 0, 3 );
    }

    TYPED_TEST( GqaMetal, TilesTakeAWideChunkAndTheOwnKernelTheRest )
    {
        // Two query heads per key head, so that rows are twice the chunk. 72 rows and 136 rows are off the tile.
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .context = 256, .chunk = 36 }, 2, 1 );
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .context = 256, .chunk = 68 }, 2, 1 );
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .context = 256, .chunk = 67 }, 2, 1 );
    }

    TYPED_TEST( GqaMetal, AHeadSizeOffTheTileAndOffAVector )
    {
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .head_size = 24, .context = 64 }, 2, 2 );
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .head_size = 13, .context = 64 }, 2, 2 );
    }

    TYPED_TEST( GqaMetal, AHeadWiderThanOneTileReadsEachTilesOwnChannels )
    {
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .head_size = 32, .context = 64, .chunk = 16 }, 2, 1 );
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .head_size = 48, .context = 128, .chunk = 68 }, 1, 1 );
    }

    TYPED_TEST( GqaMetal, SeveralTokensInOneDecodeAreRefusedEvenWhenBuiltForThem )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;
        using TensorType = Tensor<kPrecision, MetalMemoryResource>;

        GroupedQueryAttention<DeviceType::Metal, kPrecision> attention( "attention", GqaConfig( 64, 4, 2 ), Device::Metal( 0 ) );

        attention.build( BuildContext( shape_t{ 1, 64, 128 }, RuntimeMode::Inference, false ).withPrefillSize( 16 ).withDecodeTokens( 2 ) );

        auto workspace = makeGqaWorkspace<DeviceType::Metal, kPrecision>( Device::Metal( 0 ), 1, 4, 16, 64, 16, 64, "workspace." );

        attention.setState( workspace.state() );

        TensorType q( Device::Metal( 0 ), shape_t{ 1, 2, 64 } );
        TensorType k( Device::Metal( 0 ), shape_t{ 1, 2, 32 } );
        TensorType v( Device::Metal( 0 ), shape_t{ 1, 2, 32 } );

        attention.getExecutionContext()->setDecodePosition( 0 );

        EXPECT_THROW( attention.decode( q, k, v, 0 ), std::logic_error );
    }

    TYPED_TEST( GqaMetal, MoreThanOneBatch )
    {
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .batches = 3 }, 2, 2 );
    }

    TYPED_TEST( GqaMetal, EveryKvHeadGrouping )
    {
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .heads = 4, .kv_heads = 4 }, 2, 2 );
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .heads = 4, .kv_heads = 1 }, 2, 2 );
    }

    TYPED_TEST( GqaMetal, AWindowLimitsWhatARowAttendsTo )
    {
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .window = 5 }, 2, 4 );
    }

    TYPED_TEST( GqaMetal, ALaterChunkOverwritesAnEarlierOneFromTheSameOffset )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Stack<kPrecision> stack( Geometry{} );

        EXPECT_LE( stack.errorOfStep( 16, 0, false ), kAttentionBar<kPrecision> );
        EXPECT_LE( stack.errorOfStep( 16, 0, false ), kAttentionBar<kPrecision> );
        EXPECT_LE( stack.errorOfStep( 1, 16, true ), kAttentionBar<kPrecision> );
    }

    TYPED_TEST( GqaMetal, AsksForNoScratch )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Stack<kPrecision> stack( Geometry{} );

        stack.attention.getExecutionContext()->reserveScratch( 0 );

        // After 16 tokens, 3 more are 6 rows over 19 positions: a tile would fit their outputs, and rows of 2 to 7 are left over.
        EXPECT_LE( stack.errorOfStep( 16, 0, false ), kAttentionBar<kPrecision> );
        EXPECT_LE( stack.errorOfStep( 3, 16, false ), kAttentionBar<kPrecision> );
        EXPECT_EQ( stack.attention.getMemoryStats().device_scratch_bytes, std::size_t{ 0 } );
    }

    TYPED_TEST( GqaMetal, RefusesWhatItDoesNotCarry )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;
        using TensorType = Tensor<kPrecision, MetalMemoryResource>;

        Stack<kPrecision> stack( Geometry{} );

        TensorType q( Device::Metal( 0 ), shape_t{ 1, 2, 64 } );
        TensorType k( Device::Metal( 0 ), shape_t{ 1, 2, 32 } );
        TensorType v( Device::Metal( 0 ), shape_t{ 1, 2, 32 } );

        stack.attention.getExecutionContext()->setDecodePosition( 0 );

        // Two tokens in one decode are Phase 6's, and a chunk past the cache is never written.
        EXPECT_THROW( stack.attention.decode( q, k, v, 0 ), std::invalid_argument );
        EXPECT_THROW( stack.attention.prefill( q, k, v, 63 ), std::invalid_argument );
    }

    TEST( GqaMetalMemory, ItsPredictionIsWhatABuildAllocates )
    {
        Stack<TensorDataType::FP16> stack( Geometry{} );

        const BuildContext context = BuildContext( shape_t{ 1, 64, 128 }, RuntimeMode::Inference, false )
            .withPrefillSize( 16 )
            .withAllocationGranularity( allocationGranularity( Device::Metal( 0 ) ) );

        EXPECT_EQ( stack.attention.getMemoryStats().device_state_bytes, stack.attention.getRequiredMemory( context ).device_state_bytes );
    }

    TEST( GqaMetalRows, OnlyFp16AndBf16Exist )
    {
        using namespace Mila::Dnn::Quant::KvCache;

        EXPECT_FALSE( ( OperationSupported<OperationType::GroupedQueryAttentionOp, DeviceType::Metal, TensorDataType::FP32, NoKvCompression> ) );
    }
}
