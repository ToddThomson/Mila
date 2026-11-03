/**
 * @file CudaGqaOp.KeyBounds.Cuda.cpp
 * @brief Prefill with per-row key bounds, so the tokens of one image attend each other in both directions.
 *
 * The bounds travel on the execution context (Gemma4Modality.md 5.4). Each case runs a chunked prefill and decode
 * against exact double attention whose upper key limit is the row's bound, on the two kernels Gemma's prefill reaches:
 * the packed kernel over the sliding ring (head size 256) and the wide-head kernel over the global cache (512).
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

import Mila;
import Compute.CudaGqaOp;

namespace Mila::Tests::Dnn::Components::Attention::GQA::KeyBounds
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        using GlobalOp = Compute::Cuda::Gqa::CudaGqaOp<TensorDataType::BF16, false>;
        using GlobalFp8Op = Compute::Cuda::Gqa::CudaGqaOp<TensorDataType::BF16, false, true>;
        using RingOp = Compute::Cuda::Gqa::CudaGqaOp<TensorDataType::BF16, true>;
        using RingFp8Op = Compute::Cuda::Gqa::CudaGqaOp<TensorDataType::BF16, true, true>;
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        using DeviceBf16 = Tensor<TensorDataType::BF16, CudaDeviceMemoryResource>;

        // The same factor CudaGqaOp.Fp8Cache.Cuda.cpp holds a second arithmetic to: bounded attention computed as
        // exactly, against its own exact reference, as causal attention is against its own.
        constexpr double kMeanErrorRatioBound = 1.25;

        /// A run of tokens attending each other in both directions: positions [first, end).
        using Span = std::pair<int, int>;

        struct Geometry
        {
            int batch;
            int heads;
            int kv_heads;
            int head_size;
            int context;
            int chunk;
            int prefill_tokens;
            int decode_steps;

            /// Sliding window; 0 attends the whole context.
            int window{ 0 };

            /// Each inside one chunk and no longer than the window, as the network guarantees.
            std::vector<Span> spans{};
        };

        struct Sequence
        {
            std::vector<float> q, k, v;
            int tokens;
        };

        /// The last key position `t` attends: the end of its span, or itself.
        int keyBound( const Geometry& g, int t )
        {
            for ( const auto& [first, end] : g.spans )
            {
                if ( t >= first && t < end )
                {
                    return end - 1;
                }
            }

            return t;
        }

        class CudaGqaKeyBoundsTests : public ::testing::Test
        {
        protected:
            void SetUp() override
            {
                if ( getDeviceCount( DeviceType::Cuda ) == 0 )
                {
                    GTEST_SKIP() << "CUDA device not available";
                }

                context_ = createExecutionContext( Device::Cuda( 0 ) );
            }

            DeviceBf16 toDevice( const std::vector<float>& values, const shape_t& shape )
            {
                HostFp32 host( Device::Cpu(), shape );
                std::copy( values.begin(), values.end(), host.data() );

                DeviceBf16 device( Device::Cuda( 0 ), shape );
                copy( host, device, context_.get() );
                context_->synchronize();

                return device;
            }

            std::vector<float> toFloat( const DeviceBf16& device )
            {
                auto host = toHost<TensorDataType::FP32>( device, context_.get() );
                context_->synchronize();

                return std::vector<float>( host.data(), host.data() + host.size() );
            }

            /// Random BF16-exact inputs: drawn in FP32, rounded through the device once.
            Sequence randomSequence( const Geometry& g, unsigned seed )
            {
                std::mt19937 rng( seed );
                std::normal_distribution<float> normal( 0.0f, 1.0f );
                const int tokens = g.prefill_tokens + g.decode_steps;

                const auto draw = [&]( int width )
                {
                    std::vector<float> values( static_cast<std::size_t>( g.batch ) * tokens * width );

                    for ( float& value : values )
                    {
                        value = normal( rng );
                    }

                    return toFloat( toDevice( values, shape_t{ g.batch, tokens, width } ) );
                };

                Sequence sequence;
                sequence.q = draw( g.heads * g.head_size );
                sequence.k = draw( g.kv_heads * g.head_size );
                sequence.v = draw( g.kv_heads * g.head_size );
                sequence.tokens = tokens;

                return sequence;
            }

            /// Chunked prefill with each chunk's bounds set when `bounded`, then decode; every output row.
            template<typename TOp>
            std::vector<float> run( const Geometry& g, const Sequence& s, bool bounded )
            {
                const int model_dim = g.heads * g.head_size;
                const int kv_width = g.kv_heads * g.head_size;
                const int packed = ( g.heads + 2 * g.kv_heads ) * g.head_size;

                TOp op( context_.get(), GqaConfig( model_dim, g.heads, g.kv_heads ).withWindow( g.window ) );
                op.build( BuildContext( shape_t{ g.batch, g.context, packed }, RuntimeMode::Inference, false )
                    .withPrefillSize( g.chunk ) );
                op.initializeKvCache( g.batch, g.context );
                op.setUseFlashPrefill( true );
                op.setUseFlashDecode( true );

                std::vector<float> outputs( static_cast<std::size_t>( g.batch ) * s.tokens * model_dim );

                const auto slice = [&]( const std::vector<float>& all, int width, int first, int count )
                {
                    std::vector<float> part( static_cast<std::size_t>( g.batch ) * count * width );

                    for ( int b = 0; b < g.batch; ++b )
                    {
                        std::copy_n( all.begin() + ( static_cast<std::ptrdiff_t>( b ) * s.tokens + first ) * width,
                            static_cast<std::ptrdiff_t>( count ) * width,
                            part.begin() + static_cast<std::ptrdiff_t>( b ) * count * width );
                    }

                    return part;
                };

                const auto keep = [&]( const std::vector<float>& part, int first, int count )
                {
                    for ( int b = 0; b < g.batch; ++b )
                    {
                        std::copy_n( part.begin() + static_cast<std::ptrdiff_t>( b ) * count * model_dim,
                            static_cast<std::ptrdiff_t>( count ) * model_dim,
                            outputs.begin() + ( static_cast<std::ptrdiff_t>( b ) * s.tokens + first ) * model_dim );
                    }
                };

                for ( int first = 0; first < g.prefill_tokens; first += g.chunk )
                {
                    const int count = std::min( g.chunk, g.prefill_tokens - first );

                    DeviceBf16 q = toDevice( slice( s.q, model_dim, first, count ), shape_t{ g.batch, count, model_dim } );
                    DeviceBf16 k = toDevice( slice( s.k, kv_width, first, count ), shape_t{ g.batch, count, kv_width } );
                    DeviceBf16 v = toDevice( slice( s.v, kv_width, first, count ), shape_t{ g.batch, count, kv_width } );
                    DeviceBf16 out( Device::Cuda( 0 ), shape_t{ g.batch, count, model_dim } );

                    if ( bounded )
                    {
                        std::vector<dim_t> bounds( static_cast<std::size_t>( count ) );

                        for ( int row = 0; row < count; ++row )
                        {
                            bounds[ static_cast<std::size_t>( row ) ] = keyBound( g, first + row );
                        }

                        context_->setPrefillKeyBounds( first, bounds );
                    }

                    op.prefill( q, k, v, out, first );
                    context_->clearPrefillKeyBounds();
                    keep( toFloat( out ), first, count );
                }

                for ( int position = g.prefill_tokens; position < s.tokens; ++position )
                {
                    DeviceBf16 q = toDevice( slice( s.q, model_dim, position, 1 ), shape_t{ g.batch, 1, model_dim } );
                    DeviceBf16 k = toDevice( slice( s.k, kv_width, position, 1 ), shape_t{ g.batch, 1, kv_width } );
                    DeviceBf16 v = toDevice( slice( s.v, kv_width, position, 1 ), shape_t{ g.batch, 1, kv_width } );
                    DeviceBf16 out( Device::Cuda( 0 ), shape_t{ g.batch, 1, model_dim } );

                    context_->setDecodePosition( position );
                    op.decode( q, k, v, out, position );
                    keep( toFloat( out ), position, 1 );
                }

                return outputs;
            }

            /// Exact attention in double: row t attends from its window's start to its bound, or to t when unbounded.
            static std::vector<double> reference( const Geometry& g, const Sequence& s, bool bounded )
            {
                const int model_dim = g.heads * g.head_size;
                const int kv_width = g.kv_heads * g.head_size;
                const int group = g.heads / g.kv_heads;
                const double scale = 1.0 / std::sqrt( static_cast<double>( g.head_size ) );

                std::vector<double> outputs( static_cast<std::size_t>( g.batch ) * s.tokens * model_dim );

                for ( int b = 0; b < g.batch; ++b )
                {
                    for ( int t = 0; t < s.tokens; ++t )
                    {
                        const int first_key = g.window > 0 ? std::max( 0, t - g.window + 1 ) : 0;
                        const int last_key = bounded && t < g.prefill_tokens ? keyBound( g, t ) : t;

                        for ( int h = 0; h < g.heads; ++h )
                        {
                            const int kv = h / group;
                            const float* q = s.q.data() + ( static_cast<std::size_t>( b ) * s.tokens + t ) * model_dim + h * g.head_size;

                            std::vector<double> scores( static_cast<std::size_t>( last_key + 1 ), 0.0 );
                            double peak = -1e300;

                            for ( int j = first_key; j <= last_key; ++j )
                            {
                                const float* key = s.k.data() + ( static_cast<std::size_t>( b ) * s.tokens + j ) * kv_width + kv * g.head_size;
                                double dot = 0.0;

                                for ( int d = 0; d < g.head_size; ++d )
                                {
                                    dot += static_cast<double>( q[ d ] ) * key[ d ];
                                }

                                scores[ static_cast<std::size_t>( j ) ] = dot * scale;
                                peak = std::max( peak, dot * scale );
                            }

                            double total = 0.0;

                            for ( int j = first_key; j <= last_key; ++j )
                            {
                                double& score = scores[ static_cast<std::size_t>( j ) ];
                                score = std::exp( score - peak );
                                total += score;
                            }

                            double* out = outputs.data() + ( static_cast<std::size_t>( b ) * s.tokens + t ) * model_dim + h * g.head_size;

                            for ( int j = first_key; j <= last_key; ++j )
                            {
                                const float* value = s.v.data() + ( static_cast<std::size_t>( b ) * s.tokens + j ) * kv_width + kv * g.head_size;
                                const double weight = scores[ static_cast<std::size_t>( j ) ] / total;

                                for ( int d = 0; d < g.head_size; ++d )
                                {
                                    out[ d ] += weight * value[ d ];
                                }
                            }
                        }
                    }
                }

                return outputs;
            }

            static double meanError( const std::vector<float>& actual, const std::vector<double>& exact )
            {
                double sum = 0.0;

                for ( std::size_t i = 0; i < actual.size(); ++i )
                {
                    sum += std::fabs( actual[ i ] - exact[ i ] );
                }

                return sum / static_cast<double>( actual.size() );
            }

            /// Bounded attention against its exact reference, as close as causal attention is to its own.
            template<typename TOp>
            void expectAsExactAsCausal( const Geometry& g, unsigned seed )
            {
                const Sequence s = randomSequence( g, seed );
                const double bounded = meanError( run<TOp>( g, s, true ), reference( g, s, true ) );
                const double causal = meanError( run<TOp>( g, s, false ), reference( g, s, false ) );

                std::cout << std::format( "  HS {} heads {}/{} window {}: bounded mean error {:.3e}, causal {:.3e}\n",
                    g.head_size, g.heads, g.kv_heads, g.window, bounded, causal );

                EXPECT_LE( bounded, kMeanErrorRatioBound * causal );
            }

            /// Changing a span's last token moves its first token's output, and leaves every row before the span and
            /// every row of an unbounded run exactly where they were.
            template<typename TOp>
            void expectSpanAttendsForward( const Geometry& g, unsigned seed )
            {
                const auto [first, end] = g.spans.front();
                const int model_dim = g.heads * g.head_size;
                const int kv_width = g.kv_heads * g.head_size;

                const Sequence s = randomSequence( g, seed );
                Sequence changed = s;

                for ( int b = 0; b < g.batch; ++b )
                {
                    const std::size_t row = ( static_cast<std::size_t>( b ) * s.tokens + end - 1 ) * kv_width;

                    for ( int d = 0; d < kv_width; ++d )
                    {
                        changed.k[ row + d ] = -changed.k[ row + d ];
                        changed.v[ row + d ] = -changed.v[ row + d ];
                    }
                }

                const auto rowDifference = [&]( const std::vector<float>& a, const std::vector<float>& b, int t )
                {
                    double largest = 0.0;

                    for ( int batch = 0; batch < g.batch; ++batch )
                    {
                        const std::size_t base = ( static_cast<std::size_t>( batch ) * s.tokens + t ) * model_dim;

                        for ( int d = 0; d < model_dim; ++d )
                        {
                            largest = std::max( largest, static_cast<double>( std::fabs( a[ base + d ] - b[ base + d ] ) ) );
                        }
                    }

                    return largest;
                };

                const std::vector<float> bounded = run<TOp>( g, s, true );
                const std::vector<float> bounded_changed = run<TOp>( g, changed, true );
                const std::vector<float> causal = run<TOp>( g, s, false );
                const std::vector<float> causal_changed = run<TOp>( g, changed, false );

                EXPECT_GT( rowDifference( bounded, bounded_changed, first ), 1e-2 )
                    << "the span's first token does not see its last";
                EXPECT_EQ( rowDifference( causal, causal_changed, first ), 0.0 )
                    << "without bounds the first token saw a later one";

                for ( int t = 0; t < first; ++t )
                {
                    EXPECT_EQ( rowDifference( bounded, bounded_changed, t ), 0.0 ) << "row " << t << " before the span moved";
                }
            }

            /// K and V an FP8 cache holds exactly (CudaGqaOp.Fp8Cache.Cuda.cpp): the two caches' bounded outputs agree
            /// bit for bit, so the FP8 kernels apply the bounds as the BF16 kernels do.
            template<typename TFp8Op, typename TBf16Op>
            void expectFp8MatchesBf16BitForBit( const Geometry& g, unsigned seed )
            {
                Sequence s = randomSequence( g, seed );
                std::mt19937 rng( seed + 1 );
                std::normal_distribution<double> normal( 0.0, 120.0 );
                std::uniform_int_distribution<int> exponent( -9, -5 );

                const auto roundE4m3 = []( double x )
                {
                    const double magnitude = std::min( std::fabs( x ), 448.0 );

                    if ( magnitude == 0.0 )
                    {
                        return 0.0;
                    }

                    const int e = std::max( static_cast<int>( std::floor( std::log2( magnitude ) ) ), -6 );
                    const double step = std::ldexp( 1.0, e - 3 );

                    return std::copysign( std::min( std::nearbyint( magnitude / step ) * step, 448.0 ), x );
                };

                for ( std::vector<float>* rows : { &s.k, &s.v } )
                {
                    for ( std::size_t row = 0; row < rows->size(); row += static_cast<std::size_t>( g.head_size ) )
                    {
                        const double scale = std::ldexp( 1.0, exponent( rng ) );

                        for ( int d = 0; d < g.head_size; ++d )
                        {
                            const double code = d == 0 ? ( normal( rng ) < 0.0 ? -448.0 : 448.0 ) : roundE4m3( normal( rng ) );
                            ( *rows )[ row + static_cast<std::size_t>( d ) ] = static_cast<float>( code * scale );
                        }
                    }
                }

                const std::vector<float> fp8 = run<TFp8Op>( g, s, true );
                const std::vector<float> bf16 = run<TBf16Op>( g, s, true );

                double between = 0.0;

                for ( std::size_t i = 0; i < fp8.size(); ++i )
                {
                    between = std::max( between, static_cast<double>( std::fabs( fp8[ i ] - bf16[ i ] ) ) );
                }

                EXPECT_EQ( between, 0.0 ) << "the FP8 kernels bound their rows differently from the BF16 kernels";
            }

            /// `action` throws a logic_error that names the key bounds -- not one of the op's other refusals.
            template<typename TAction>
            static void expectBoundsRefusal( TAction&& action )
            {
                try
                {
                    action();
                    ADD_FAILURE() << "the prefill was not refused";
                }
                catch ( const std::logic_error& e )
                {
                    EXPECT_NE( std::string( e.what() ).find( "key bounds" ), std::string::npos ) << e.what();
                }
            }

            std::unique_ptr<IExecutionContext> context_;
        };

        // Gemma 4's global layers: MQA, head size 512, batch 2. Spans at a chunk's start, inside one, filling one to its
        // end, and a one-token span, which is causal.
        const Geometry kGlobal{ 2, 16, 1, 512, 160, 32, 96, 32, 0, { { 0, 9 }, { 20, 21 }, { 40, 57 }, { 64, 96 } } };

        // Gemma 4's sliding layers: head size 256, group 2, window 64 with 32-row chunks -- a 95-row ring that the
        // prefill and decode wrap more than twice. One span per chunk, the last at the end of the prefill.
        const Geometry kSliding{ 1, 16, 8, 256, 224, 32, 160, 64, 64, { { 3, 30 }, { 33, 64 }, { 100, 120 }, { 140, 160 } } };
    }

    TEST_F( CudaGqaKeyBoundsTests, GlobalGeometry_AsExactAsCausal )
    {
        expectAsExactAsCausal<GlobalOp>( kGlobal, 31u );
    }

    TEST_F( CudaGqaKeyBoundsTests, SlidingRing_AsExactAsCausal )
    {
        expectAsExactAsCausal<RingOp>( kSliding, 37u );
    }

    TEST_F( CudaGqaKeyBoundsTests, GlobalGeometry_SpanAttendsForward )
    {
        Geometry g = kGlobal;
        g.spans = { { 40, 57 } };

        expectSpanAttendsForward<GlobalOp>( g, 41u );
    }

    TEST_F( CudaGqaKeyBoundsTests, SlidingRing_SpanAttendsForward )
    {
        Geometry g = kSliding;
        g.spans = { { 100, 120 } };

        expectSpanAttendsForward<RingOp>( g, 43u );
    }

    TEST_F( CudaGqaKeyBoundsTests, Fp8Cache_MatchesTheBf16CacheBitForBit )
    {
        expectFp8MatchesBf16BitForBit<GlobalFp8Op, GlobalOp>( kGlobal, 47u );
        expectFp8MatchesBf16BitForBit<RingFp8Op, RingOp>( kSliding, 53u );
    }

    // ====================================================================
    // Refusals: bounds for another chunk, bounds the context cannot hold, and every path that does not read them.
    // ====================================================================

    TEST_F( CudaGqaKeyBoundsTests, BoundsForAnotherChunk_AreRefused )
    {
        const Geometry g{ 1, 16, 1, 512, 64, 32, 32, 0 };
        GlobalOp op( context_.get(), GqaConfig( 16 * 512, 16, 1 ) );
        op.build( BuildContext( shape_t{ 1, g.context, 18 * 512 }, RuntimeMode::Inference, false ).withPrefillSize( g.chunk ) );
        op.initializeKvCache( 1, g.context );
        op.setUseFlashPrefill( true );

        DeviceBf16 q( Device::Cuda( 0 ), shape_t{ 1, 32, 16 * 512 } );
        DeviceBf16 k( Device::Cuda( 0 ), shape_t{ 1, 32, 512 } );
        DeviceBf16 out( Device::Cuda( 0 ), shape_t{ 1, 32, 16 * 512 } );

        std::vector<dim_t> bounds( 32 );

        for ( int row = 0; row < 32; ++row )
        {
            bounds[ static_cast<std::size_t>( row ) ] = 32 + row;
        }

        context_->setPrefillKeyBounds( 32, bounds );

        expectBoundsRefusal( [&] { op.prefill( q, k, k, out, 0 ); } );

        context_->clearPrefillKeyBounds();
    }

    TEST_F( CudaGqaKeyBoundsTests, BoundsTheContextCannotHold_AreRefused )
    {
        // Below the row's own position.
        EXPECT_THROW( context_->setPrefillKeyBounds( 10, std::vector<dim_t>{ 10, 10, 11 } ), std::invalid_argument );

        // Decreasing.
        EXPECT_THROW( context_->setPrefillKeyBounds( 0, std::vector<dim_t>{ 2, 2, 2, 3, 2 } ), std::invalid_argument );

        // Past the chunk.
        EXPECT_THROW( context_->setPrefillKeyBounds( 0, std::vector<dim_t>{ 0, 3, 3 } ), std::invalid_argument );

        EXPECT_NO_THROW( context_->setPrefillKeyBounds( 0, std::vector<dim_t>{ 0, 2, 2, 3 } ) );

        context_->clearPrefillKeyBounds();
    }

    TEST_F( CudaGqaKeyBoundsTests, PrefillOffTheFlashKernels_IsRefused )
    {
        GlobalOp op( context_.get(), GqaConfig( 16 * 512, 16, 1 ) );
        op.build( BuildContext( shape_t{ 1, 64, 18 * 512 }, RuntimeMode::Inference, false ).withPrefillSize( 32 ) );
        op.initializeKvCache( 1, 64 );
        op.setUseFlashPrefill( false );

        DeviceBf16 q( Device::Cuda( 0 ), shape_t{ 1, 4, 16 * 512 } );
        DeviceBf16 k( Device::Cuda( 0 ), shape_t{ 1, 4, 512 } );
        DeviceBf16 out( Device::Cuda( 0 ), shape_t{ 1, 4, 16 * 512 } );

        context_->setPrefillKeyBounds( 0, std::vector<dim_t>{ 3, 3, 3, 3 } );

        expectBoundsRefusal( [&] { op.prefill( q, k, k, out, 0 ); } );

        context_->clearPrefillKeyBounds();
    }
}
