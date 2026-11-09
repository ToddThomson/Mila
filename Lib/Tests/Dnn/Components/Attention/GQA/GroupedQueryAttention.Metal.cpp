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
import Compute.MetalSharedKvAttentionOp;
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

            // The most tokens one decode call takes, a speculative round's verify.
            dim_t decode_tokens{ 1 };
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

            /// `fused` prefills through the fused kernel, with a workspace that holds no score buffers, as a transformer sizes it then.
            explicit Stack( const Geometry& geometry, bool fused = false )
                : shape( geometry ),
                  attention( "attention",
                      GqaConfig( geometry.heads * geometry.head_size, geometry.heads, geometry.kv_heads ).withWindow( geometry.window ),
                      Device::Metal( 0 ) ),
                  keys( static_cast<std::size_t>( geometry.batches * geometry.kv_heads * geometry.context * geometry.head_size ) ),
                  values( keys.size() )
            {
                const dim_t packed = ( geometry.heads + 2 * geometry.kv_heads ) * geometry.head_size;

                attention.build( BuildContext( shape_t{ geometry.batches, geometry.context, packed }, RuntimeMode::Inference, false )
                    .withPrefillSize( geometry.chunk ).withDecodeTokens( geometry.decode_tokens ) );

                workspace = makeGqaWorkspace<DeviceType::Metal, TPrecision>( Device::Metal( 0 ), geometry.batches, geometry.heads,
                    geometry.head_size, geometry.context, geometry.chunk, fused ? dim_t{ 0 } : geometry.context, "workspace." );

                attention.setState( workspace.state() );
                attention.setUseFlashPrefill( fused );
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
        void expectChunksThenDecodes( const Geometry& geometry, dim_t chunks, dim_t decodes, bool fused = false )
        {
            Stack<TPrecision, TKvPolicy> stack( geometry, fused );

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

        // One KV head's K and V side by side, as the fused prefill expands them; the unfused pipeline uses the first half.
        const std::size_t expanded = 2 * static_cast<std::size_t>( geometry.context * geometry.head_size ) * 2;

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

    // -----------------------------------------------------------------------------------------------------------------
    // The fused prefill (MetalBackend.md section 4.5), held to the same reference and bars as the unfused pipeline, at the geometries
    // of the models and through every cache. Its decodes run the same kernels as an unfused stack's, after a fused prefill wrote the cache.
    // -----------------------------------------------------------------------------------------------------------------

    template<typename TPrecisionConstant>
    class GqaMetalFused : public ::testing::Test
    {
    };

    TYPED_TEST_SUITE( GqaMetalFused, Precisions, PrecisionNames );

    TYPED_TEST( GqaMetalFused, LlamasGeometries )
    {
        // Four query heads to a KV head (3.1 8B), and three (3.2 3B), at a head of 128.
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .heads = 8, .kv_heads = 2, .head_size = 128, .context = 512, .chunk = 64 }, 4, 2, true );
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .heads = 6, .kv_heads = 2, .head_size = 128, .context = 512, .chunk = 64 }, 4, 2, true );
    }

    TYPED_TEST( GqaMetalFused, QwensFullAttention )
    {
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .heads = 6, .kv_heads = 1, .head_size = 256, .context = 256, .chunk = 64 }, 3, 2, true );
    }

    TYPED_TEST( GqaMetalFused, GemmasGlobalLayers )
    {
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .heads = 16, .kv_heads = 1, .head_size = 512, .context = 256, .chunk = 64 }, 3, 2, true );
    }

    TYPED_TEST( GqaMetalFused, GemmasSlidingLayersOnTheRing )
    {
        // Rings of 103 and 71 rows, each wrapped several times over the chunks.
        expectChunksThenDecodes<TypeParam::value, SlidingWindowKvCache>(
            Geometry{ .heads = 4, .kv_heads = 2, .head_size = 256, .context = 384, .chunk = 64, .window = 40 }, 5, 4, true );
        expectChunksThenDecodes<TypeParam::value, SlidingWindowKvCache>(
            Geometry{ .heads = 4, .kv_heads = 2, .head_size = 128, .context = 384, .chunk = 32, .window = 40 }, 9, 4, true );
    }

    TYPED_TEST( GqaMetalFused, AWrappedRingWhoseWindowHoldsWholeTiles )
    {
        // A window of 100 in a ring of 131 rows: once wrapped, some tiles sit inside every row's window and before every row's
        // position by row number, yet hold positions after them; only the ring's own mapping tells them apart.
        expectChunksThenDecodes<TypeParam::value, SlidingWindowKvCache>(
            Geometry{ .heads = 4, .kv_heads = 2, .head_size = 128, .context = 512, .chunk = 32, .window = 100 }, 12, 4, true );
        expectChunksThenDecodes<TypeParam::value, SlidingWindowKvCache>(
            Geometry{ .heads = 4, .kv_heads = 2, .head_size = 256, .context = 512, .chunk = 64, .window = 200 }, 7, 4, true );
    }

    TYPED_TEST( GqaMetalFused, AWindowOverTheWholeCache )
    {
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .heads = 8, .kv_heads = 2, .head_size = 128, .context = 256, .chunk = 32, .window = 37 }, 5, 3, true );
    }

    TYPED_TEST( GqaMetalFused, ChunksOffTheTilesAndShorterThanOne )
    {
        // 67 positions leave a part block of rows; chunks of 5, 3 and 1 attend fewer keys than a tile reads.
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .heads = 8, .kv_heads = 2, .head_size = 128, .context = 256, .chunk = 67 }, 2, 1, true );

        constexpr TensorDataType kPrecision = TypeParam::value;

        Stack<kPrecision> stack( Geometry{ .heads = 8, .kv_heads = 2, .head_size = 128, .context = 128, .chunk = 16 }, true );

        EXPECT_LE( stack.errorOfStep( 5, 0, false ), kAttentionBar<kPrecision> );
        EXPECT_LE( stack.errorOfStep( 3, 5, false ), kAttentionBar<kPrecision> );
        EXPECT_LE( stack.errorOfStep( 1, 8, false ), kAttentionBar<kPrecision> );
        EXPECT_LE( stack.errorOfStep( 16, 9, false ), kAttentionBar<kPrecision> );
        EXPECT_LE( stack.errorOfStep( 1, 25, true ), kAttentionBar<kPrecision> );
    }

    TYPED_TEST( GqaMetalFused, MoreThanOneBatch )
    {
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .batches = 2, .heads = 8, .kv_heads = 2, .head_size = 128, .context = 128, .chunk = 32 }, 2, 2, true );
    }

    TYPED_TEST( GqaMetalFused, APartBlockOfRowsWritesOnlyItsOwnBatch )
    {
        // Five positions leave eleven rows of a 16-row block empty; written, they would land on the next batch's first rows.
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .batches = 3, .heads = 8, .kv_heads = 2, .head_size = 128, .context = 64, .chunk = 5 }, 8, 1, true );
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .batches = 3, .heads = 16, .kv_heads = 1, .head_size = 512, .context = 64, .chunk = 5 }, 4, 1, true );
    }

    TYPED_TEST( GqaMetalFused, AHeldPromptOfFifteenHundredPositions )
    {
        // Three chunks of 512 after a first: the last attends 2048 keys, 64 tiles.
        expectChunksThenDecodes<TypeParam::value>( Geometry{ .heads = 8, .kv_heads = 2, .head_size = 128, .context = 2112, .chunk = 512 }, 4, 1, true );
    }

    TYPED_TEST( GqaMetalFused, AnFp8Cache )
    {
        expectChunksThenDecodes<TypeParam::value, PerTokenKvFp8<>>(
            Geometry{ .heads = 8, .kv_heads = 2, .head_size = 128, .context = 256, .chunk = 64 }, 3, 2, true );
        expectChunksThenDecodes<TypeParam::value, PerTokenKvFp8<>>(
            Geometry{ .batches = 2, .heads = 16, .kv_heads = 1, .head_size = 512, .context = 128, .chunk = 32 }, 3, 2, true );
    }

    TYPED_TEST( GqaMetalFused, ALaterChunkOverwritesAnEarlierOneFromTheSameOffset )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        Stack<kPrecision> stack( Geometry{ .heads = 8, .kv_heads = 2, .head_size = 128, .context = 128, .chunk = 32 }, true );

        EXPECT_LE( stack.errorOfStep( 32, 0, false ), kAttentionBar<kPrecision> );
        EXPECT_LE( stack.errorOfStep( 32, 32, false ), kAttentionBar<kPrecision> );
        EXPECT_LE( stack.errorOfStep( 32, 0, false ), kAttentionBar<kPrecision> );
        EXPECT_LE( stack.errorOfStep( 1, 32, true ), kAttentionBar<kPrecision> );
    }

    TYPED_TEST( GqaMetalFused, RefusesWhatItDoesNotServe )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;
        using TensorType = Tensor<kPrecision, MetalMemoryResource>;

        EXPECT_TRUE( ( GroupedQueryAttention<DeviceType::Metal, kPrecision>::supportsFlashPrefill( 128 ) ) );
        EXPECT_TRUE( ( GroupedQueryAttention<DeviceType::Metal, kPrecision>::supportsFlashPrefill( 512 ) ) );
        EXPECT_FALSE( ( GroupedQueryAttention<DeviceType::Metal, kPrecision>::supportsFlashPrefill( 64 ) ) );

        // A head the kernel has no instance for, and an unfused prefill with no score buffers.
        Stack<kPrecision> small_head( Geometry{}, true );
        Stack<kPrecision> unsized( Geometry{ .heads = 8, .kv_heads = 2, .head_size = 128, .context = 64, .chunk = 16 }, true );

        unsized.attention.setUseFlashPrefill( false );

        TensorType q( Device::Metal( 0 ), shape_t{ 1, 4, 64 } );
        TensorType k( Device::Metal( 0 ), shape_t{ 1, 4, 32 } );
        TensorType v( Device::Metal( 0 ), shape_t{ 1, 4, 32 } );

        TensorType wide_q( Device::Metal( 0 ), shape_t{ 1, 4, 8 * 128 } );
        TensorType wide_k( Device::Metal( 0 ), shape_t{ 1, 4, 2 * 128 } );
        TensorType wide_v( Device::Metal( 0 ), shape_t{ 1, 4, 2 * 128 } );

        EXPECT_THROW( small_head.attention.prefill( q, k, v, 0 ), std::logic_error );
        EXPECT_THROW( unsized.attention.prefill( wide_q, wide_k, wide_v, 0 ), std::logic_error );
    }

    TYPED_TEST( GqaMetalFused, AContextShorterThanATile )
    {
        // A tile reads 64 rows; contexts of 16 and 32, each head size, the cache built a tile long and only the context's rows written.
        for ( const dim_t head_size : { dim_t{ 128 }, dim_t{ 256 }, dim_t{ 512 } } )
        {
            SCOPED_TRACE( std::format( "head of {}", head_size ) );
            expectChunksThenDecodes<TypeParam::value>( Geometry{ .heads = 4, .kv_heads = 2, .head_size = head_size, .context = 16, .chunk = 8 }, 2, 0, true );
            expectChunksThenDecodes<TypeParam::value>( Geometry{ .heads = 4, .kv_heads = 2, .head_size = head_size, .context = 32, .chunk = 16 }, 1, 16, true );
        }

        expectChunksThenDecodes<TypeParam::value, SlidingWindowKvCache>(
            Geometry{ .heads = 4, .kv_heads = 2, .head_size = 256, .context = 32, .chunk = 8, .window = 12 }, 3, 8, true );
    }

    TEST( GqaMetalRingRows, TheRingExistsAtFp16AndBf16Only )
    {
        EXPECT_TRUE( ( OperationSupported<OperationType::GroupedQueryAttentionOp, DeviceType::Metal, TensorDataType::FP16, SlidingWindowKvCache> ) );
        EXPECT_TRUE( ( OperationSupported<OperationType::GroupedQueryAttentionOp, DeviceType::Metal, TensorDataType::BF16, SlidingWindowKvCache> ) );
        EXPECT_FALSE( ( OperationSupported<OperationType::GroupedQueryAttentionOp, DeviceType::Metal, TensorDataType::FP32, SlidingWindowKvCache> ) );
    }

    // -----------------------------------------------------------------------------------------------------------------
    // A decode of several tokens in a row (MetalBackend.md section 4.6, Gemma4Mtp.md 4.7): each token at its own position, against the
    // reference, and equal to the bit to the same tokens decoded one at a time, since each token's sums run as its own decode's do.
    // -----------------------------------------------------------------------------------------------------------------

    namespace
    {
        /// Rows [first, first + count) of each batch of a [B, T, width] tensor's values.
        std::vector<float> tokenRows( std::span<const float> values, dim_t batches, dim_t tokens, dim_t width, dim_t first, dim_t count )
        {
            std::vector<float> rows;

            for ( dim_t b = 0; b < batches; ++b )
            {
                const auto start = values.begin() + static_cast<std::ptrdiff_t>( ( b * tokens + first ) * width );

                rows.insert( rows.end(), start, start + static_cast<std::ptrdiff_t>( count * width ) );
            }

            return rows;
        }

        /// One decode call of `count` of `tokens` tokens from `first`, its output as floats.
        template<TensorDataType TPrecision, typename TKvPolicy>
        std::vector<float> decodeCall( Stack<TPrecision, TKvPolicy>& stack, std::span<const float> q, std::span<const float> k,
            std::span<const float> v, dim_t tokens, dim_t position, dim_t first, dim_t count )
        {
            const Geometry& g = stack.shape;
            const auto qs = tokenRows( q, g.batches, tokens, g.heads * g.head_size, first, count );
            const auto ks = tokenRows( k, g.batches, tokens, g.kv_heads * g.head_size, first, count );
            const auto vs = tokenRows( v, g.batches, tokens, g.kv_heads * g.head_size, first, count );

            auto q_tensor = metalTensor<TPrecision>( qs, shape_t{ g.batches, count, g.heads * g.head_size } );
            auto k_tensor = metalTensor<TPrecision>( ks, shape_t{ g.batches, count, g.kv_heads * g.head_size } );
            auto v_tensor = metalTensor<TPrecision>( vs, shape_t{ g.batches, count, g.kv_heads * g.head_size } );

            stack.attention.getExecutionContext()->setDecodePosition( position + first );

            auto& output = stack.attention.decode( q_tensor, k_tensor, v_tensor, position + first );

            stack.attention.synchronize();

            return readFloats<TPrecision>( output );
        }

        /**
         * Two stacks given the same chunks, then `tokens` tokens from `position`: one at a time into the first, in one call into the
         * second. Every output equal, the one call's within the reference's bar, and a decode after them in each equal, which reads the
         * rows the one call wrote.
         */
        template<TensorDataType TPrecision, typename TKvPolicy = NoKvCompression>
        void expectOneCallEqualsOneAtATime( Geometry geometry, dim_t chunks, dim_t tokens )
        {
            geometry.decode_tokens = tokens;

            Stack<TPrecision, TKvPolicy> one_at_a_time( geometry );
            Stack<TPrecision, TKvPolicy> one_call( geometry );

            dim_t position = 0;

            // The chunks that write the history are held to the reference too, so a wrong cache cannot pass as two equal ones.
            for ( dim_t c = 0; c < chunks; ++c )
            {
                EXPECT_LE( one_at_a_time.errorOfStep( geometry.chunk, position, false ), kAttentionBar<TPrecision> ) << "chunk " << c;
                EXPECT_LE( one_call.errorOfStep( geometry.chunk, position, false ), kAttentionBar<TPrecision> ) << "chunk " << c;
                position += geometry.chunk;
            }

            const auto q_count = static_cast<std::size_t>( geometry.batches * tokens * geometry.heads * geometry.head_size );
            const auto kv_count = static_cast<std::size_t>( geometry.batches * tokens * geometry.kv_heads * geometry.head_size );
            const std::vector<float> q = randomValues( q_count, 501, 0.5f );
            const std::vector<float> k = randomValues( kv_count, 502, 0.5f );
            const std::vector<float> v = randomValues( kv_count, 503, 0.5f );

            const auto stored_k = asStored<TPrecision>( std::span<const float>( k ) );
            const auto stored_v = asStored<TPrecision>( std::span<const float>( v ) );

            one_at_a_time.remember( stored_k, stored_v, tokens, position );
            one_call.remember( stored_k, stored_v, tokens, position );

            const dim_t width = geometry.heads * geometry.head_size;
            std::vector<float> expected( q_count );

            for ( dim_t t = 0; t < tokens; ++t )
            {
                const std::vector<float> row = decodeCall( one_at_a_time, q, k, v, tokens, position, t, 1 );

                for ( dim_t b = 0; b < geometry.batches; ++b )
                {
                    std::copy_n( row.begin() + static_cast<std::ptrdiff_t>( b * width ), width,
                        expected.begin() + static_cast<std::ptrdiff_t>( ( b * tokens + t ) * width ) );
                }
            }

            const std::vector<float> produced = decodeCall( one_call, q, k, v, tokens, position, 0, tokens );

            ASSERT_EQ( produced.size(), expected.size() );

            std::size_t differing = 0;

            for ( std::size_t i = 0; i < expected.size(); ++i )
            {
                differing += produced[i] != expected[i] ? 1 : 0;
            }

            EXPECT_EQ( differing, 0u ) << "of " << expected.size() << " outputs";

            const auto stored_q = asStored<TPrecision>( std::span<const float>( q ) );

            EXPECT_LE( relativeError( produced, one_call.reference( stored_q, tokens, position ) ), kAttentionBar<TPrecision> );

            const double after_one_at_a_time = one_at_a_time.errorOfStep( 1, position + tokens, true );
            const double after_one_call = one_call.errorOfStep( 1, position + tokens, true );

            EXPECT_EQ( after_one_call, after_one_at_a_time ) << "the decode after them read other rows";
            EXPECT_LE( after_one_call, kAttentionBar<TPrecision> );
        }
    }

    template<typename TPrecisionConstant>
    class GqaMetalDecodeTokens : public ::testing::Test
    {
    };

    TYPED_TEST_SUITE( GqaMetalDecodeTokens, Precisions, PrecisionNames );

    TYPED_TEST( GqaMetalDecodeTokens, ARoundOfEveryLengthUpToEight )
    {
        for ( const dim_t tokens : { dim_t{ 1 }, dim_t{ 2 }, dim_t{ 5 }, dim_t{ 8 } } )
        {
            SCOPED_TRACE( std::format( "{} tokens", tokens ) );
            expectOneCallEqualsOneAtATime<TypeParam::value>( Geometry{ .context = 128 }, 3, tokens );
        }
    }

    TYPED_TEST( GqaMetalDecodeTokens, TheModelsHeadSizesAndGroupings )
    {
        // Gemma 4's global layer, 16 query heads to one KV head at 512 channels; its sliding layer, 2 to a KV head at 256; Llama's 128.
        expectOneCallEqualsOneAtATime<TypeParam::value>( Geometry{ .heads = 16, .kv_heads = 1, .head_size = 512, .context = 96, .chunk = 32 }, 2, 5 );
        expectOneCallEqualsOneAtATime<TypeParam::value>( Geometry{ .heads = 4, .kv_heads = 2, .head_size = 256, .context = 96, .chunk = 32 }, 2, 5 );
        expectOneCallEqualsOneAtATime<TypeParam::value>( Geometry{ .heads = 8, .kv_heads = 2, .head_size = 128, .context = 96, .chunk = 32 }, 2, 8 );
    }

    TYPED_TEST( GqaMetalDecodeTokens, MoreThanOneBatch )
    {
        expectOneCallEqualsOneAtATime<TypeParam::value>( Geometry{ .batches = 2, .context = 96 }, 2, 4 );
    }

    TYPED_TEST( GqaMetalDecodeTokens, AWindowEachTokenCutsAtItsOwnPosition )
    {
        expectOneCallEqualsOneAtATime<TypeParam::value>( Geometry{ .context = 96, .window = 7 }, 3, 6 );
    }

    TYPED_TEST( GqaMetalDecodeTokens, TheRingBeforeAcrossAndAfterItsWrap )
    {
        // The ring holds one fused tile, 64 rows, past its 13 + 16 - 1: a round from 16 sits before the wrap, from 64 at it, from 96 after.
        for ( const dim_t chunks : { dim_t{ 1 }, dim_t{ 4 }, dim_t{ 6 } } )
        {
            SCOPED_TRACE( std::format( "after {} chunks", chunks ) );
            expectOneCallEqualsOneAtATime<TypeParam::value, SlidingWindowKvCache>(
                Geometry{ .heads = 4, .kv_heads = 2, .head_size = 256, .context = 128, .chunk = 16, .window = 13 }, chunks, 8 );
        }
    }

    TYPED_TEST( GqaMetalDecodeTokens, AnFp8Cache )
    {
        expectOneCallEqualsOneAtATime<TypeParam::value, PerTokenKvFp8<>>( Geometry{ .heads = 16, .kv_heads = 1, .head_size = 512, .context = 96, .chunk = 32 }, 2, 4 );
    }

    TYPED_TEST( GqaMetalDecodeTokens, RefusesMoreTokensThanBuiltForOrTheRingKeeps )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;
        using TensorType = Tensor<kPrecision, MetalMemoryResource>;

        Stack<kPrecision> stack( Geometry{ .decode_tokens = 4 } );

        stack.errorOfStep( 16, 0, false );

        TensorType q( Device::Metal( 0 ), shape_t{ 1, 5, 64 } );
        TensorType k( Device::Metal( 0 ), shape_t{ 1, 5, 32 } );
        TensorType v( Device::Metal( 0 ), shape_t{ 1, 5, 32 } );

        stack.attention.getExecutionContext()->setDecodePosition( 16 );

        EXPECT_THROW( stack.attention.decode( q, k, v, 16 ), std::invalid_argument );

        // A ring of 4 + 4 - 1 = 7 rows keeps a window of 4 for 4 tokens; 5 would write over a key the first still attends.
        Stack<kPrecision, SlidingWindowKvCache> ring( Geometry{ .context = 64, .chunk = 4, .window = 4, .decode_tokens = 5 } );

        ring.errorOfStep( 4, 0, false );
        ring.attention.getExecutionContext()->setDecodePosition( 4 );

        EXPECT_THROW( ring.attention.decode( q, k, v, 4 ), std::invalid_argument );
    }

    TYPED_TEST( GqaMetalDecodeTokens, AsksForScratchForEveryTokensPartials )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        const Geometry geometry{ .batches = 2, .decode_tokens = 5 };
        Stack<kPrecision> stack( geometry );

        const std::size_t partials = static_cast<std::size_t>( geometry.batches * geometry.decode_tokens * geometry.heads ) * 32
            * static_cast<std::size_t>( geometry.head_size + 2 ) * sizeof( float );

        EXPECT_EQ( stack.attention.getMemoryStats().device_scratch_bytes, occupiedDeviceBytes( partials, allocationGranularity( Device::Metal( 0 ) ) ) );
    }

    TYPED_TEST( GqaMetalDecodeTokens, ARingRewindsOnlyAsFarAsItsRowsStillHoldTheWindow )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;

        // 16 + 16 - 1 = 31 rows in a context of 256: 15 positions past the window may be dropped, not 16.
        Stack<kPrecision, SlidingWindowKvCache> ring( Geometry{ .context = 256, .chunk = 16, .window = 16 } );

        ring.errorOfStep( 16, 0, false );
        ring.errorOfStep( 16, 16, false );

        EXPECT_TRUE( ring.attention.rewindKvCache( 17, 32 ) );
        EXPECT_FALSE( ring.attention.rewindKvCache( 16, 32 ) );

        // A ring as long as the context holds every position.
        Stack<kPrecision, SlidingWindowKvCache> whole( Geometry{ .context = 64, .chunk = 16, .window = 64 } );

        whole.errorOfStep( 16, 0, false );
        whole.errorOfStep( 16, 16, false );

        EXPECT_TRUE( whole.attention.rewindKvCache( 0, 32 ) );
    }

    // -----------------------------------------------------------------------------------------------------------------
    // A reader of another layer's cache (MetalBackend.md section 4.6, Gemma4Mtp.md 4.3): a query at position p over the keys through
    // p - 1, as the reference computes for that query at p - 1, and nothing written.
    // -----------------------------------------------------------------------------------------------------------------

    namespace
    {
        template<TensorDataType TPrecision, typename TKvPolicy = NoKvCompression>
        void expectAReaderAttendsTheKeysBeforeIt( const Geometry& geometry, dim_t chunks, dim_t decodes )
        {
            using TensorType = Tensor<TPrecision, MetalMemoryResource>;
            using ReaderType = Metal::Gqa::MetalSharedKvAttentionOp<TPrecision>;

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

            const dim_t width = geometry.heads * geometry.head_size;
            const GqaConfig config = GqaConfig( width, geometry.heads, geometry.kv_heads ).withWindow( geometry.window );

            ReaderType reader( stack.attention.getExecutionContext(), config );
            reader.build( BuildContext( shape_t{ geometry.batches, geometry.context, width }, RuntimeMode::Inference, false ) );

            const auto q_count = static_cast<std::size_t>( geometry.batches * width );
            const std::vector<float> q = randomValues( q_count, 601, 0.5f );

            TensorType q_tensor = metalTensor<TPrecision>( q, shape_t{ geometry.batches, 1, width } );
            TensorType output( Device::Metal( 0 ), shape_t{ geometry.batches, 1, width } );

            // The reader's query sits at the next position, which holds no key yet.
            stack.attention.getExecutionContext()->setDecodePosition( position );
            reader.decode( q_tensor, stack.attention.getKvCacheView(), output );
            stack.attention.synchronize();

            const auto stored_q = asStored<TPrecision>( std::span<const float>( q ) );

            EXPECT_LE( relativeError( readFloats<TPrecision>( output ), stack.reference( stored_q, 1, position - 1 ) ), kAttentionBar<TPrecision> );

            // It wrote nothing: the next decode of the layer that owns the cache still matches the reference.
            EXPECT_LE( stack.errorOfStep( 1, position, true ), kAttentionBar<TPrecision> );
        }
    }

    template<typename TPrecisionConstant>
    class GqaMetalSharedKv : public ::testing::Test
    {
    };

    TYPED_TEST_SUITE( GqaMetalSharedKv, Precisions, PrecisionNames );

    TYPED_TEST( GqaMetalSharedKv, AttendsTheKeysBeforeItsPosition )
    {
        expectAReaderAttendsTheKeysBeforeIt<TypeParam::value>( Geometry{ .context = 96 }, 2, 3 );
    }

    TYPED_TEST( GqaMetalSharedKv, TheDrafterOfGemmasLayers )
    {
        // The drafter's global layer: 16 query heads over the target's one KV head at 512; its sliding layers, 2 to a KV head at 256,
        // over the ring before and after it wraps.
        expectAReaderAttendsTheKeysBeforeIt<TypeParam::value>( Geometry{ .heads = 16, .kv_heads = 1, .head_size = 512, .context = 96, .chunk = 32 }, 2, 1 );
        expectAReaderAttendsTheKeysBeforeIt<TypeParam::value, SlidingWindowKvCache>(
            Geometry{ .heads = 4, .kv_heads = 2, .head_size = 256, .context = 128, .chunk = 16, .window = 13 }, 1, 2 );
        expectAReaderAttendsTheKeysBeforeIt<TypeParam::value, SlidingWindowKvCache>(
            Geometry{ .heads = 4, .kv_heads = 2, .head_size = 256, .context = 128, .chunk = 16, .window = 13 }, 4, 3 );
    }

    TYPED_TEST( GqaMetalSharedKv, AnFp8Cache )
    {
        expectAReaderAttendsTheKeysBeforeIt<TypeParam::value, PerTokenKvFp8<>>( Geometry{ .heads = 16, .kv_heads = 1, .head_size = 512, .context = 96, .chunk = 32 }, 2, 2 );
    }

    TYPED_TEST( GqaMetalSharedKv, RefusesACacheOfAnotherGeometry )
    {
        constexpr TensorDataType kPrecision = TypeParam::value;
        using TensorType = Tensor<kPrecision, MetalMemoryResource>;

        Stack<kPrecision> stack( Geometry{} );

        stack.errorOfStep( 16, 0, false );

        Metal::Gqa::MetalSharedKvAttentionOp<kPrecision> reader( stack.attention.getExecutionContext(), GqaConfig( 64, 4, 4 ) );
        reader.build( BuildContext( shape_t{ 1, 64, 64 }, RuntimeMode::Inference, false ) );

        TensorType q( Device::Metal( 0 ), shape_t{ 1, 1, 64 } );
        TensorType output( Device::Metal( 0 ), shape_t{ 1, 1, 64 } );

        EXPECT_THROW( reader.decode( q, stack.attention.getKvCacheView(), output ), std::invalid_argument );
    }

    TEST( GqaMetalSharedKvRows, TheReaderExistsAtFp16AndBf16Only )
    {
        EXPECT_TRUE( ( OperationSupported<OperationType::SharedKvAttentionOp, DeviceType::Metal, TensorDataType::FP16, void> ) );
        EXPECT_TRUE( ( OperationSupported<OperationType::SharedKvAttentionOp, DeviceType::Metal, TensorDataType::BF16, void> ) );
        EXPECT_FALSE( ( OperationSupported<OperationType::SharedKvAttentionOp, DeviceType::Metal, TensorDataType::FP32, void> ) );
    }
}
