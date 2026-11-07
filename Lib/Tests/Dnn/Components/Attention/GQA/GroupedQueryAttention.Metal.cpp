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
#include "../../QuantizedReference.h"

namespace Mila::Tests::Dnn::Components
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Quant::KvCache;
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

        /// A row of the cache as an FP8 cache stores it: E4M3 codes of x * (1 / scale), saturating, with scale max |x| / 448.
        void asFp8Row( std::span<double> row )
        {
            float largest = 0.0f;

            for ( const double x : row )
            {
                largest = std::fmax( largest, std::fabs( static_cast<float>( x ) ) );
            }

            const float scale = largest / 448.0f;
            const float inverse = largest > 0.0f ? 1.0f / scale : 0.0f;

            for ( double& x : row )
            {
                x = QuantizedReference::fp8E4M3Value( QuantizedReference::fp8E4M3Code( static_cast<float>( x ) * inverse ) ) * scale;
            }
        }

        /// A stack of one layer, with what the reference needs: everything the cache was given, as the cache holds it.
        template<TensorDataType TPrecision, typename TKvPolicy = NoKvCompression>
        struct Stack
        {
            using TensorType = Tensor<TPrecision, MetalMemoryResource>;

            static constexpr bool kFp8Cache = std::is_same_v<TKvPolicy, PerTokenKvFp8<>>;

            Geometry shape;
            GroupedQueryAttention<DeviceType::Metal, TPrecision, TKvPolicy> attention;
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

                            if constexpr ( kFp8Cache )
                            {
                                const auto row = static_cast<std::size_t>( ( ( b * shape.kv_heads + h ) * shape.context + first_position + t ) * shape.head_size );

                                asFp8Row( std::span( keys ).subspan( row, static_cast<std::size_t>( shape.head_size ) ) );
                                asFp8Row( std::span( values ).subspan( row, static_cast<std::size_t>( shape.head_size ) ) );
                            }
                        }
                    }
                }
            }

            /// Run a chunk or one decoded token, and return its error against the reference. `zero_rows` gives K and V nothing but zeros.
            double errorOfStep( dim_t tokens, dim_t first_position, bool decode, bool zero_rows = false )
            {
                const auto q_count = static_cast<std::size_t>( shape.batches * tokens * shape.heads * shape.head_size );
                const auto kv_count = static_cast<std::size_t>( shape.batches * tokens * shape.kv_heads * shape.head_size );

                const std::vector<float> q_values = randomValues( q_count, seed++, 0.5f );
                const std::vector<float> k_values = zero_rows ? std::vector<float>( kv_count, 0.0f ) : randomValues( kv_count, seed++, 0.5f );
                const std::vector<float> v_values = zero_rows ? std::vector<float>( kv_count, 0.0f ) : randomValues( kv_count, seed++, 0.5f );

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

        template<TensorDataType TPrecision, typename TKvPolicy = NoKvCompression>
        void expectChunksThenDecodes( const Geometry& geometry, dim_t chunks, dim_t decodes )
        {
            Stack<TPrecision, TKvPolicy> stack( geometry );

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

    TYPED_TEST( GqaMetal, AsksForScratchOnlyForADecodesPartials )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        const Geometry geometry{ .batches = 2 };
        Stack<kPrecision> stack( geometry );

        // 32 blocks, each with a head's output and its largest score and sum, in FP32, for every head of every batch.
        const std::size_t partials = static_cast<std::size_t>( geometry.batches * geometry.heads ) * 32
            * static_cast<std::size_t>( geometry.head_size + 2 ) * sizeof( float );

        stack.attention.getExecutionContext()->reserveScratch( partials );

        // After 16 tokens, 3 more are 6 rows over 19 positions: a tile would fit their outputs, and rows of 2 to 7 are left over. Prefill
        // pads nothing, so asks for nothing past the reservation, and a decode asks for no more than it.
        EXPECT_LE( stack.errorOfStep( 16, 0, false ), kAttentionBar<kPrecision> );
        EXPECT_LE( stack.errorOfStep( 3, 16, false ), kAttentionBar<kPrecision> );
        EXPECT_LE( stack.errorOfStep( 1, 19, true ), kAttentionBar<kPrecision> );

        const BuildContext context = BuildContext( shape_t{ geometry.batches, geometry.context, ( geometry.heads + 2 * geometry.kv_heads ) * geometry.head_size },
            RuntimeMode::Inference, false )
            .withPrefillSize( geometry.chunk )
            .withAllocationGranularity( allocationGranularity( Device::Metal( 0 ) ) );

        EXPECT_EQ( stack.attention.getMemoryStats().device_scratch_bytes, occupiedDeviceBytes( partials, allocationGranularity( Device::Metal( 0 ) ) ) );
        EXPECT_EQ( stack.attention.getMemoryStats().device_scratch_bytes, stack.attention.getRequiredMemory( context ).device_scratch_bytes );
    }

    TYPED_TEST( GqaMetal, EveryDepthAroundTheBlockCount )
    {
        // 32 blocks: from one key to one past the count, so that blocks start empty, fill one key at a time, and take a second key.
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .context = 128 }, 0, 34 );
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .head_size = 128, .context = 128 }, 0, 34 );
    }

    TYPED_TEST( GqaMetal, TheHeadSizesTheModelsUseRunTheirOwnKernels )
    {
        for ( const dim_t head_size : { dim_t{ 64 }, dim_t{ 128 }, dim_t{ 256 }, dim_t{ 512 } } )
        {
            SCOPED_TRACE( std::format( "head of {}", head_size ) );
            expectChunksThenDecodes<TypeParam::value>( Geometry{ .head_size = head_size, .context = 256, .chunk = 64 }, 2, 3 );
        }
    }

    TYPED_TEST( GqaMetal, EveryGroupingOnTheModelsKernels )
    {
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .heads = 8, .kv_heads = 2, .head_size = 256, .context = 128, .chunk = 32 }, 2, 2 );
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .heads = 4, .kv_heads = 4, .head_size = 128, .context = 128, .chunk = 32 }, 2, 2 );
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .heads = 8, .kv_heads = 1, .head_size = 128, .context = 128, .chunk = 32 }, 2, 2 );
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .batches = 2, .heads = 6, .kv_heads = 2, .head_size = 128, .context = 128, .chunk = 32 }, 2, 2 );
    }

    TYPED_TEST( GqaMetal, AWindowThatCutsABlock )
    {
        // 37 keys of window over 32 blocks: the first key moves through a block as the position advances.
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .head_size = 128, .context = 160, .chunk = 32, .window = 37 }, 3, 6 );
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .context = 160, .chunk = 32, .window = 37 }, 3, 6 );
    }

    TYPED_TEST( GqaMetal, ADecodeAtTheCachesLastPosition )
    {
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .head_size = 128, .context = 64, .chunk = 16 }, 3, 16 );
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .context = 64, .chunk = 16 }, 3, 16 );
    }

    TYPED_TEST( GqaMetal, AHeadPastWhatADecodeHoldsIsRefused )
    {
        GroupedQueryAttention<DeviceType::Metal, TypeParam::value> attention( "attention", GqaConfig( 4 * 544, 4, 2 ), Device::Metal( 0 ) );

        EXPECT_THROW( attention.build( BuildContext( shape_t{ 1, 64, 8 * 544 }, RuntimeMode::Inference, false ).withPrefillSize( 16 ) ),
            std::invalid_argument );
    }

    TYPED_TEST( GqaMetal, MoreQueryHeadsToAKvHeadThanAThreadgroupHoldsAreRefused )
    {
        // 33 SIMD-groups of 32 is 1056 threads, past the 1024 any Apple GPU's threadgroup takes.
        GroupedQueryAttention<DeviceType::Metal, TypeParam::value> attention( "attention", GqaConfig( 33 * 64, 33, 1 ), Device::Metal( 0 ) );

        EXPECT_THROW( attention.build( BuildContext( shape_t{ 1, 64, 35 * 64 }, RuntimeMode::Inference, false ).withPrefillSize( 16 ) ),
            std::invalid_argument );
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
        EXPECT_FALSE( ( OperationSupported<OperationType::GroupedQueryAttentionOp, DeviceType::Metal, TensorDataType::FP32, NoKvCompression> ) );
        EXPECT_TRUE( ( OperationSupported<OperationType::GroupedQueryAttentionOp, DeviceType::Metal, TensorDataType::FP16, PerTokenKvFp8<>> ) );
        EXPECT_TRUE( ( OperationSupported<OperationType::GroupedQueryAttentionOp, DeviceType::Metal, TensorDataType::BF16, PerTokenKvFp8<>> ) );
        EXPECT_FALSE( ( OperationSupported<OperationType::GroupedQueryAttentionOp, DeviceType::Metal, TensorDataType::FP32, PerTokenKvFp8<>> ) );
    }

    // -----------------------------------------------------------------------------------------------------------------
    // An FP8 cache: the reference holds each row as the cache stores it, so the bars measure the attention, not the format.
    // -----------------------------------------------------------------------------------------------------------------

    template<typename TPrecisionConstant>
    class GqaMetalFp8 : public ::testing::Test
    {
    };

    TYPED_TEST_SUITE( GqaMetalFp8, Precisions, PrecisionNames );

    TYPED_TEST( GqaMetalFp8, AChunkThenTokensMatchTheReference )
    {
        expectChunksThenDecodes<TypeParam::value, PerTokenKvFp8<>>( Geometry{}, 2, 4 );
    }

    TYPED_TEST( GqaMetalFp8, ADecodeWithNothingBeforeItAttendsOnlyToItself )
    {
        expectChunksThenDecodes<TypeParam::value, PerTokenKvFp8<>>( Geometry{}, 0, 3 );
    }

    TYPED_TEST( GqaMetalFp8, TilesTakeAWideChunkAndTheOwnKernelTheRest )
    {
        expectChunksThenDecodes<TypeParam::value, PerTokenKvFp8<>>( Geometry{ .context = 256, .chunk = 36 }, 2, 1 );
        expectChunksThenDecodes<TypeParam::value, PerTokenKvFp8<>>( Geometry{ .context = 256, .chunk = 67 }, 2, 1 );
    }

    TYPED_TEST( GqaMetalFp8, AHeadSizeOffTheTileAndOffAVector )
    {
        expectChunksThenDecodes<TypeParam::value, PerTokenKvFp8<>>( Geometry{ .head_size = 24, .context = 64 }, 2, 2 );
        expectChunksThenDecodes<TypeParam::value, PerTokenKvFp8<>>( Geometry{ .head_size = 13, .context = 64 }, 2, 2 );
    }

    TYPED_TEST( GqaMetalFp8, TheHeadSizesTheModelsUseRunTheirOwnKernels )
    {
        for ( const dim_t head_size : { dim_t{ 64 }, dim_t{ 128 }, dim_t{ 256 }, dim_t{ 512 } } )
        {
            SCOPED_TRACE( std::format( "head of {}", head_size ) );
            expectChunksThenDecodes<TypeParam::value, PerTokenKvFp8<>>( Geometry{ .head_size = head_size, .context = 256, .chunk = 64 }, 2, 3 );
        }
    }

    TYPED_TEST( GqaMetalFp8, EveryGroupingAndMoreThanOneBatch )
    {
        expectChunksThenDecodes<TypeParam::value, PerTokenKvFp8<>>( Geometry{ .heads = 8, .kv_heads = 2, .head_size = 128, .context = 128, .chunk = 32 }, 2, 2 );
        expectChunksThenDecodes<TypeParam::value, PerTokenKvFp8<>>( Geometry{ .heads = 4, .kv_heads = 4, .head_size = 128, .context = 128, .chunk = 32 }, 2, 2 );
        expectChunksThenDecodes<TypeParam::value, PerTokenKvFp8<>>( Geometry{ .heads = 8, .kv_heads = 1, .head_size = 128, .context = 128, .chunk = 32 }, 2, 2 );
        expectChunksThenDecodes<TypeParam::value, PerTokenKvFp8<>>( Geometry{ .batches = 3 }, 2, 2 );
    }

    TYPED_TEST( GqaMetalFp8, EveryDepthAroundTheBlockCountAndTheLastPosition )
    {
        expectChunksThenDecodes<TypeParam::value, PerTokenKvFp8<>>( Geometry{ .head_size = 128, .context = 128 }, 0, 34 );
        expectChunksThenDecodes<TypeParam::value, PerTokenKvFp8<>>( Geometry{ .head_size = 128, .context = 64, .chunk = 16 }, 3, 16 );
    }

    TYPED_TEST( GqaMetalFp8, AWindowThatCutsABlock )
    {
        expectChunksThenDecodes<TypeParam::value, PerTokenKvFp8<>>( Geometry{ .head_size = 128, .context = 160, .chunk = 32, .window = 37 }, 3, 6 );
    }

    TYPED_TEST( GqaMetalFp8, RowsOfZerosStoreZerosAndAttendAsZeros )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Stack<kPrecision, PerTokenKvFp8<>> stack( Geometry{ .head_size = 128, .context = 64 } );

        EXPECT_LE( stack.errorOfStep( 16, 0, false, true ), kAttentionBar<kPrecision> );
        EXPECT_LE( stack.errorOfStep( 16, 16, false ), kAttentionBar<kPrecision> );
        EXPECT_LE( stack.errorOfStep( 1, 32, true, true ), kAttentionBar<kPrecision> );
        EXPECT_LE( stack.errorOfStep( 1, 33, true ), kAttentionBar<kPrecision> );
    }

    TYPED_TEST( GqaMetalFp8, ALaterChunkOverwritesAnEarlierOneFromTheSameOffset )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Stack<kPrecision, PerTokenKvFp8<>> stack( Geometry{} );

        EXPECT_LE( stack.errorOfStep( 16, 0, false ), kAttentionBar<kPrecision> );
        EXPECT_LE( stack.errorOfStep( 16, 0, false ), kAttentionBar<kPrecision> );
        EXPECT_LE( stack.errorOfStep( 1, 16, true ), kAttentionBar<kPrecision> );
    }

    TYPED_TEST( GqaMetalFp8, AsksForTheScratchItUses )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        // A context long enough that one KV head's expanded rows are more than a decode's partials.
        const Geometry geometry{ .batches = 2, .head_size = 128, .context = 1024, .chunk = 64 };
        Stack<kPrecision, PerTokenKvFp8<>> stack( geometry );

        const BuildContext context = BuildContext( shape_t{ geometry.batches, geometry.context, ( geometry.heads + 2 * geometry.kv_heads ) * geometry.head_size },
            RuntimeMode::Inference, false )
            .withPrefillSize( geometry.chunk )
            .withAllocationGranularity( allocationGranularity( Device::Metal( 0 ) ) );

        const std::size_t expanded = static_cast<std::size_t>( geometry.context * geometry.head_size ) * 2;

        EXPECT_EQ( stack.attention.getMemoryStats().device_scratch_bytes, occupiedDeviceBytes( expanded, allocationGranularity( Device::Metal( 0 ) ) ) );
        EXPECT_EQ( stack.attention.getMemoryStats().device_scratch_bytes, stack.attention.getRequiredMemory( context ).device_scratch_bytes );

        // A request past the reservation throws, so these hold the footprint to what is asked: the whole cache attended, and a decode.
        stack.attention.getExecutionContext()->reserveScratch( expanded );

        for ( dim_t chunk = 0; chunk < 16; ++chunk )
        {
            EXPECT_LE( stack.errorOfStep( 64, chunk * 64, false ), kAttentionBar<kPrecision> ) << "chunk " << chunk;
        }

        EXPECT_LE( stack.errorOfStep( 1, 1023, true ), kAttentionBar<kPrecision> );
    }

    TEST( GqaMetalFp8Memory, ItsPredictionIsWhatABuildAllocatesAndIsLessThanTheUncompressedCache )
    {
        Stack<TensorDataType::FP16, PerTokenKvFp8<>> fp8( Geometry{ .head_size = 128 } );
        Stack<TensorDataType::FP16> uncompressed( Geometry{ .head_size = 128 } );

        const BuildContext context = BuildContext( shape_t{ 1, 64, 8 * 128 }, RuntimeMode::Inference, false )
            .withPrefillSize( 16 )
            .withAllocationGranularity( allocationGranularity( Device::Metal( 0 ) ) );

        const std::size_t state = fp8.attention.getMemoryStats().device_state_bytes;

        EXPECT_EQ( state, fp8.attention.getRequiredMemory( context ).device_state_bytes );
        EXPECT_LT( state, uncompressed.attention.getMemoryStats().device_state_bytes );
    }

    // -----------------------------------------------------------------------------------------------------------------
    // A sliding window's ring (SlidingWindowKvCache): the reference holds every position, so a row read from the wrong turn of the ring,
    // or a key past the window, misses the bar.
    // -----------------------------------------------------------------------------------------------------------------

    template<typename TPrecisionConstant>
    class GqaMetalRing : public ::testing::Test
    {
    };

    TYPED_TEST_SUITE( GqaMetalRing, Precisions, PrecisionNames );

    TYPED_TEST( GqaMetalRing, ChunksShorterThanTheWindowEqualToItAndPastIt )
    {
        // Rings of 52, 31 and 47 rows in a context of 256: each wraps several times over the chunks and decodes.
        expectChunksThenDecodes<TypeParam::value, SlidingWindowKvCache>( Geometry{ .context = 256, .chunk = 16, .window = 37 }, 6, 40 );
        expectChunksThenDecodes<TypeParam::value, SlidingWindowKvCache>( Geometry{ .context = 256, .chunk = 16, .window = 16 }, 6, 40 );
        expectChunksThenDecodes<TypeParam::value, SlidingWindowKvCache>( Geometry{ .context = 256, .chunk = 32, .window = 16 }, 4, 40 );
    }

    TYPED_TEST( GqaMetalRing, ADecodeBeforeAcrossAndAfterTheWrap )
    {
        // A ring of 20 rows, decoded from the first position on: each step to 60 is before, at or after a wrap.
        expectChunksThenDecodes<TypeParam::value, SlidingWindowKvCache>( Geometry{ .head_size = 128, .context = 128, .chunk = 8, .window = 13 }, 0, 60 );
    }

    TYPED_TEST( GqaMetalRing, TheHeadSizesGemmaUses )
    {
        for ( const dim_t head_size : { dim_t{ 256 }, dim_t{ 512 } } )
        {
            SCOPED_TRACE( std::format( "head of {}", head_size ) );
            expectChunksThenDecodes<TypeParam::value, SlidingWindowKvCache>(
                Geometry{ .heads = 4, .kv_heads = 2, .head_size = head_size, .context = 192, .chunk = 32, .window = 40 }, 4, 12 );
        }
    }

    TYPED_TEST( GqaMetalRing, AContextNoLongerThanTheRingIsTheUncompressedCache )
    {
        // window + chunk - 1 = 79 is past a context of 64, so the ring is the whole context.
        expectChunksThenDecodes<TypeParam::value, SlidingWindowKvCache>( Geometry{ .context = 64, .chunk = 16, .window = 64 }, 3, 16 );
    }

    TYPED_TEST( GqaMetalRing, RefusesAChunkPastTheOneItWasBuiltForAndNoWindow )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;
        using TensorType = Tensor<kPrecision, MetalMemoryResource>;

        Stack<kPrecision, SlidingWindowKvCache> stack( Geometry{ .context = 256, .chunk = 16, .window = 16 } );

        TensorType q( Device::Metal( 0 ), shape_t{ 1, 17, 64 } );
        TensorType k( Device::Metal( 0 ), shape_t{ 1, 17, 32 } );
        TensorType v( Device::Metal( 0 ), shape_t{ 1, 17, 32 } );

        EXPECT_THROW( stack.attention.prefill( q, k, v, 0 ), std::invalid_argument );

        GroupedQueryAttention<DeviceType::Metal, kPrecision, SlidingWindowKvCache> unwindowed( "attention", GqaConfig( 64, 4, 2 ), Device::Metal( 0 ) );

        EXPECT_THROW( unwindowed.build( BuildContext( shape_t{ 1, 64, 128 }, RuntimeMode::Inference, false ).withPrefillSize( 16 ) ), std::invalid_argument );
    }

    TEST( GqaMetalRingMemory, ItsPredictionIsWhatABuildAllocatesAndIsTheRingsRows )
    {
        Stack<TensorDataType::FP16, SlidingWindowKvCache> ring( Geometry{ .head_size = 128, .context = 1024, .chunk = 64, .window = 128 } );
        Stack<TensorDataType::FP16> uncompressed( Geometry{ .head_size = 128, .context = 1024, .chunk = 64, .window = 128 } );

        const BuildContext context = BuildContext( shape_t{ 1, 1024, 8 * 128 }, RuntimeMode::Inference, false )
            .withPrefillSize( 64 )
            .withAllocationGranularity( allocationGranularity( Device::Metal( 0 ) ) );

        const std::size_t state = ring.attention.getMemoryStats().device_state_bytes;

        // The ring is window + chunk - 1 = 191 rows where the whole cache is 1024: K and V, two KV heads, 128 FP16 channels a row. The
        // component's own outputs are the same either way. Both caches are under the size that rounds to the granularity.
        EXPECT_EQ( state, ring.attention.getRequiredMemory( context ).device_state_bytes );
        EXPECT_EQ( uncompressed.attention.getMemoryStats().device_state_bytes - state, std::size_t{ 2 * 2 * ( 1024 - 191 ) * 128 * 2 } );
    }

    TEST( GqaMetalRingRows, TheRingExistsAtFp16AndBf16Only )
    {
        EXPECT_TRUE( ( OperationSupported<OperationType::GroupedQueryAttentionOp, DeviceType::Metal, TensorDataType::FP16, SlidingWindowKvCache> ) );
        EXPECT_TRUE( ( OperationSupported<OperationType::GroupedQueryAttentionOp, DeviceType::Metal, TensorDataType::BF16, SlidingWindowKvCache> ) );
        EXPECT_FALSE( ( OperationSupported<OperationType::GroupedQueryAttentionOp, DeviceType::Metal, TensorDataType::FP32, SlidingWindowKvCache> ) );
    }
}
