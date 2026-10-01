/**
 * @file CudaGqaOp.Fp8Cache.Cuda.cpp
 * @brief The FP8 KV cache (CudaGqaOp<BF16, false, true>, PerTokenKvFp8) against the BF16 cache and an exact reference.
 *
 * On values the FP8 cache holds exactly, the two ops agree bit for bit, through chunked prefill and decode. On random
 * values, each op's mean error against exact double attention over the values its cache holds -- the host rounding K
 * and V by the cache's rule for the FP8 op -- stays within a small factor of the BF16 op's (Quantization.md, Part III).
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
#include <vector>

import Mila;
import Compute.CudaGqaOp;

namespace Mila::Tests::Dnn::Components::Attention::GQA::Fp8Cache
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        using Fp8Op = Compute::Cuda::Gqa::CudaGqaOp<TensorDataType::BF16, false, true>;
        using Bf16Op = Compute::Cuda::Gqa::CudaGqaOp<TensorDataType::BF16, false>;

        // The bounded sliding-window ring, of FP8 codes (SlidingWindowKvFp8) and of BF16 (SlidingWindowKvCache).
        using Fp8RingOp = Compute::Cuda::Gqa::CudaGqaOp<TensorDataType::BF16, true, true>;
        using Bf16RingOp = Compute::Cuda::Gqa::CudaGqaOp<TensorDataType::BF16, true>;
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        using DeviceBf16 = Tensor<TensorDataType::BF16, CudaDeviceMemoryResource>;

        // Set after the first run, which showed the bound written before it (1e-2 absolute, on the premise that
        // outputs stay below 1) was wrong: with normal inputs outputs reach 3 to 4, where one BF16 step is 0.016 to
        // 0.031, and the largest error over ~1e5 outputs moves with the data alone. The FP8 op must compute the
        // attention over what its cache holds as exactly as the BF16 op computes it over its own: mean error against
        // each op's exact reference within this factor. Measured 1.13 to 1.14 at the three geometries below.
        constexpr double kMeanErrorRatioBound = 1.25;

        /// E4M3 round to nearest even, saturating at 448; subnormals below 2^-6 in steps of 2^-9.
        double roundE4m3( double x )
        {
            const double magnitude = std::min( std::fabs( x ), 448.0 );

            if ( magnitude == 0.0 )
            {
                return 0.0;
            }

            const int exponent = std::max( static_cast<int>( std::floor( std::log2( magnitude ) ) ), -6 );
            const double step = std::ldexp( 1.0, exponent - 3 );
            const double rounded = std::min( std::nearbyint( magnitude / step ) * step, 448.0 );

            return std::copysign( rounded, x );
        }

        /// A row as the cache holds it: codes times the row's scale.
        std::vector<double> quantizedRow( const float* row, int length )
        {
            float absmax = 0.0f;

            for ( int i = 0; i < length; ++i )
            {
                absmax = std::max( absmax, std::fabs( row[ i ] ) );
            }

            const float scale = absmax / 448.0f;
            const float inverse = absmax > 0.0f ? 1.0f / scale : 0.0f;
            std::vector<double> values( static_cast<std::size_t>( length ) );

            for ( int i = 0; i < length; ++i )
            {
                values[ static_cast<std::size_t>( i ) ] = roundE4m3( row[ i ] * inverse ) * scale;
            }

            return values;
        }

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
        };

        struct Sequence
        {
            // Every token's Q, K and V as the device holds them (BF16), [batch, tokens, heads * head_size].
            std::vector<float> q, k, v;
            int tokens;
        };

        class CudaGqaFp8CacheTests : public ::testing::Test
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

            /// Prefill in chunks, then decode one token at a time; every output row, [batch, tokens, heads * HS].
            template<typename TOp>
            std::vector<float> run( const Geometry& g, const Sequence& s )
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

                    op.prefill( q, k, v, out, first );
                    keep( toFloat( out ), first, count );
                }

                for ( int position = g.prefill_tokens; position < s.tokens; ++position )
                {
                    DeviceBf16 q = toDevice( slice( s.q, model_dim, position, 1 ), shape_t{ g.batch, 1, model_dim } );
                    DeviceBf16 k = toDevice( slice( s.k, kv_width, position, 1 ), shape_t{ g.batch, 1, kv_width } );
                    DeviceBf16 v = toDevice( slice( s.v, kv_width, position, 1 ), shape_t{ g.batch, 1, kv_width } );
                    DeviceBf16 out( Device::Cuda( 0 ), shape_t{ g.batch, 1, model_dim } );

                    // An op decoded outside a network is given the position the network would write.
                    context_->setDecodePosition( position );
                    op.decode( q, k, v, out, position );
                    keep( toFloat( out ), position, 1 );
                }

                return outputs;
            }

            /// Causal attention in double, within the window when there is one, over K and V rounded as the FP8 cache
            /// rounds them.
            static std::vector<double> reference( const Geometry& g, const Sequence& s, bool quantized )
            {
                const int model_dim = g.heads * g.head_size;
                const int kv_width = g.kv_heads * g.head_size;
                const int group = g.heads / g.kv_heads;
                const double scale = 1.0 / std::sqrt( static_cast<double>( g.head_size ) );

                std::vector<double> outputs( static_cast<std::size_t>( g.batch ) * s.tokens * model_dim );

                for ( int b = 0; b < g.batch; ++b )
                {
                    std::vector<std::vector<double>> keys, values;

                    for ( int t = 0; t < s.tokens; ++t )
                    {
                        for ( int kv = 0; kv < g.kv_heads; ++kv )
                        {
                            const std::size_t row = ( static_cast<std::size_t>( b ) * s.tokens + t ) * kv_width + kv * g.head_size;
                            keys.push_back( quantized ? quantizedRow( s.k.data() + row, g.head_size )
                                : std::vector<double>( s.k.data() + row, s.k.data() + row + g.head_size ) );
                            values.push_back( quantized ? quantizedRow( s.v.data() + row, g.head_size )
                                : std::vector<double>( s.v.data() + row, s.v.data() + row + g.head_size ) );
                        }
                    }

                    for ( int t = 0; t < s.tokens; ++t )
                    {
                        for ( int h = 0; h < g.heads; ++h )
                        {
                            const int kv = h / group;
                            const float* q = s.q.data() + ( static_cast<std::size_t>( b ) * s.tokens + t ) * model_dim + h * g.head_size;

                            std::vector<double> scores( static_cast<std::size_t>( t + 1 ), 0.0 );
                            double peak = -1e300;
                            const int first_key = g.window > 0 ? std::max( 0, t - g.window + 1 ) : 0;

                            for ( int j = first_key; j <= t; ++j )
                            {
                                const std::vector<double>& key = keys[ static_cast<std::size_t>( j ) * g.kv_heads + kv ];
                                double dot = 0.0;

                                for ( int d = 0; d < g.head_size; ++d )
                                {
                                    dot += q[ d ] * key[ static_cast<std::size_t>( d ) ];
                                }

                                scores[ static_cast<std::size_t>( j ) ] = dot * scale;
                                peak = std::max( peak, dot * scale );
                            }

                            double total = 0.0;

                            for ( int j = first_key; j <= t; ++j )
                            {
                                double& score = scores[ static_cast<std::size_t>( j ) ];
                                score = std::exp( score - peak );
                                total += score;
                            }

                            double* out = outputs.data() + ( static_cast<std::size_t>( b ) * s.tokens + t ) * model_dim + h * g.head_size;

                            for ( int j = first_key; j <= t; ++j )
                            {
                                const std::vector<double>& value = values[ static_cast<std::size_t>( j ) * g.kv_heads + kv ];
                                const double weight = scores[ static_cast<std::size_t>( j ) ] / total;

                                for ( int d = 0; d < g.head_size; ++d )
                                {
                                    out[ d ] += weight * value[ static_cast<std::size_t>( d ) ];
                                }
                            }
                        }
                    }
                }

                return outputs;
            }

            template<typename TFp8Op = Fp8Op, typename TBf16Op = Bf16Op>
            void expectAsExactAsBf16( const Geometry& g, unsigned seed )
            {
                const Sequence s = randomSequence( g, seed );
                const std::vector<float> fp8 = run<TFp8Op>( g, s );
                const std::vector<float> bf16 = run<TBf16Op>( g, s );
                const std::vector<double> exact_fp8 = reference( g, s, true );
                const std::vector<double> exact_bf16 = reference( g, s, false );

                // Each op against the exact attention over the cache it holds: the largest and the mean error.
                double fp8_max = 0.0, bf16_max = 0.0, fp8_sum = 0.0, bf16_sum = 0.0, against_bf16 = 0.0, cost_sum = 0.0;

                for ( std::size_t i = 0; i < fp8.size(); ++i )
                {
                    const double fp8_error = std::fabs( fp8[ i ] - exact_fp8[ i ] );
                    const double bf16_error = std::fabs( bf16[ i ] - exact_bf16[ i ] );

                    fp8_max = std::max( fp8_max, fp8_error );
                    bf16_max = std::max( bf16_max, bf16_error );
                    fp8_sum += fp8_error;
                    bf16_sum += bf16_error;
                    against_bf16 = std::max( against_bf16, static_cast<double>( std::fabs( fp8[ i ] - bf16[ i ] ) ) );
                    cost_sum += std::fabs( exact_fp8[ i ] - exact_bf16[ i ] );
                }

                const double count = static_cast<double>( fp8.size() );

                std::cout << std::format( "  HS {} heads {}/{} batch {}: FP8 op against its exact max {:.3e} mean {:.3e}; "
                    "BF16 op against its exact max {:.3e} mean {:.3e}; FP8 against BF16 cache max {:.3e}, exact quantization "
                    "cost mean {:.3e}\n", g.head_size, g.heads, g.kv_heads, g.batch, fp8_max, fp8_sum / count, bf16_max,
                    bf16_sum / count, against_bf16, cost_sum / count );

                EXPECT_LE( fp8_sum, kMeanErrorRatioBound * bf16_sum );
            }

            /// K and V the FP8 cache holds without loss: each row a power-of-two scale times E4M3 codes, one of them
            /// +-448, so the row's scale is that power of two and every code is exact -- and exact in BF16 too.
            Sequence losslessSequence( const Geometry& g, unsigned seed )
            {
                Sequence s = randomSequence( g, seed );
                std::mt19937 rng( seed + 1 );
                std::normal_distribution<double> normal( 0.0, 120.0 );
                std::uniform_int_distribution<int> exponent( -9, -5 );

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

                return s;
            }

            template<typename TFp8Op = Fp8Op, typename TBf16Op = Bf16Op>
            void expectLosslessMatchesBf16( const Geometry& g, unsigned seed )
            {
                const Sequence s = losslessSequence( g, seed );
                const std::vector<float> fp8 = run<TFp8Op>( g, s );
                const std::vector<float> bf16 = run<TBf16Op>( g, s );
                const std::vector<double> exact = reference( g, s, false );

                double fp8_error = 0.0, bf16_error = 0.0, between = 0.0;

                for ( std::size_t i = 0; i < fp8.size(); ++i )
                {
                    fp8_error = std::max( fp8_error, std::fabs( fp8[ i ] - exact[ i ] ) );
                    bf16_error = std::max( bf16_error, std::fabs( bf16[ i ] - exact[ i ] ) );
                    between = std::max( between, static_cast<double>( std::fabs( fp8[ i ] - bf16[ i ] ) ) );
                }

                std::cout << std::format( "  lossless HS {} heads {}/{}: FP8 op against exact {:.3e}, BF16 op against exact "
                    "{:.3e}, between them {:.3e}\n", g.head_size, g.heads, g.kv_heads, fp8_error, bf16_error, between );

                EXPECT_EQ( between, 0.0 ) << "the FP8 read path is not the BF16 path's arithmetic on identical values";
            }

            std::unique_ptr<IExecutionContext> context_;
        };
    }

    // With K and V the FP8 cache holds exactly, both caches hold the same values, and every scale is a power of two,
    // which factors out of the dot products without rounding: the two ops must agree bit for bit.
    TEST_F( CudaGqaFp8CacheTests, LosslessValues_MatchTheBf16CacheBitForBit )
    {
        expectLosslessMatchesBf16( Geometry{ 1, 8, 2, 128, 256, 64, 192, 64 }, 7u );
        expectLosslessMatchesBf16( Geometry{ 2, 16, 1, 512, 160, 32, 96, 64 }, 11u );
        expectLosslessMatchesBf16( Geometry{ 1, 12, 2, 256, 192, 64, 128, 40 }, 13u );
    }

    // Llama 3.1 8B's group (4) and head size; three prefill chunks, then decode past 64 so split-K engages.
    TEST_F( CudaGqaFp8CacheTests, LlamaGeometry_AsExactAsTheBf16Cache )
    {
        expectAsExactAsBf16( Geometry{ 1, 8, 2, 128, 256, 64, 192, 64 }, 7u );
    }

    // Gemma 4 global layers: MQA, head size 512; batch 2 for the batch strides.
    TEST_F( CudaGqaFp8CacheTests, GemmaGlobalGeometry_AsExactAsTheBf16Cache )
    {
        expectAsExactAsBf16( Geometry{ 2, 16, 1, 512, 160, 32, 96, 64 }, 11u );
    }

    // Qwen 3.8 full attention's head size 256, group 6.
    TEST_F( CudaGqaFp8CacheTests, QwenGeometry_AsExactAsTheBf16Cache )
    {
        expectAsExactAsBf16( Geometry{ 1, 12, 2, 256, 192, 64, 128, 40 }, 13u );
    }

    // ====================================================================
    // The bounded ring of FP8 codes (SlidingWindowKvFp8). Window 64 with 32-row chunks gives a 95-row ring, which a
    // 160-token prefill and 64 decode steps wrap more than twice.
    // ====================================================================

    // Gemma 4's sliding layers: head size 256, group 2.
    TEST_F( CudaGqaFp8CacheTests, Ring_LosslessValues_MatchTheBf16RingBitForBit )
    {
        expectLosslessMatchesBf16<Fp8RingOp, Bf16RingOp>( Geometry{ 1, 16, 8, 256, 224, 32, 160, 64, 64 }, 17u );
        expectLosslessMatchesBf16<Fp8RingOp, Bf16RingOp>( Geometry{ 2, 8, 2, 128, 224, 32, 160, 64, 64 }, 19u );
    }

    TEST_F( CudaGqaFp8CacheTests, Ring_GemmaSlidingGeometry_AsExactAsTheBf16Ring )
    {
        expectAsExactAsBf16<Fp8RingOp, Bf16RingOp>( Geometry{ 1, 16, 8, 256, 224, 32, 160, 64, 64 }, 17u );
    }

    TEST_F( CudaGqaFp8CacheTests, Ring_RequiredStateMemory_EqualsTheBuild )
    {
        Fp8RingOp op( context_.get(), GqaConfig( 16 * 256, 16, 8 ).withWindow( 64 ) );
        const auto build = BuildContext( shape_t{ 1, 512, 32 * 256 }, RuntimeMode::Inference, false )
            .withPrefillSize( 32 )
            .withAllocationGranularity( allocationGranularity( Device::Cuda( 0 ) ) );

        const std::size_t required = op.getRequiredStateMemorySize( build );
        op.build( build );

        EXPECT_EQ( op.getStateMemorySize(), required );
        EXPECT_EQ( op.getCacheCapacity(), 64 + 32 - 1 );
    }

    // The footprint the planner reads is the build's: codes plus one scale per row.
    TEST_F( CudaGqaFp8CacheTests, RequiredStateMemory_EqualsTheBuild )
    {
        Fp8Op op( context_.get(), GqaConfig( 8 * 128, 8, 2 ) );
        const auto build = BuildContext( shape_t{ 1, 512, 12 * 128 }, RuntimeMode::Inference, false )
            .withPrefillSize( 64 )
            .withAllocationGranularity( allocationGranularity( Device::Cuda( 0 ) ) );

        const std::size_t required = op.getRequiredStateMemorySize( build );
        op.build( build );

        std::cout << std::format( "  required {} bytes, built {} bytes\n", required, op.getStateMemorySize() );

        EXPECT_EQ( op.getStateMemorySize(), required );
        EXPECT_GE( required, std::size_t( 2 ) * 2 * 512 * 128 + std::size_t( 2 ) * 2 * 512 * 4 );
    }

    // Nothing but the fused kernels reads an FP8 cache.
    TEST_F( CudaGqaFp8CacheTests, UnsupportedHeadSize_IsRefusedAtBuild )
    {
        Fp8Op op( context_.get(), GqaConfig( 8 * 64, 8, 2 ) );

        EXPECT_THROW( op.build( BuildContext( shape_t{ 1, 128, 12 * 64 }, RuntimeMode::Inference, false ).withPrefillSize( 64 ) ),
            std::invalid_argument );
    }

    TEST_F( CudaGqaFp8CacheTests, PrefillWithFlashOff_IsRefused )
    {
        Fp8Op op( context_.get(), GqaConfig( 8 * 128, 8, 2 ) );
        op.build( BuildContext( shape_t{ 1, 128, 12 * 128 }, RuntimeMode::Inference, false ).withPrefillSize( 64 ) );
        op.initializeKvCache( 1, 128 );

        DeviceBf16 q( Device::Cuda( 0 ), shape_t{ 1, 16, 8 * 128 } );
        DeviceBf16 k( Device::Cuda( 0 ), shape_t{ 1, 16, 2 * 128 } );
        DeviceBf16 v( Device::Cuda( 0 ), shape_t{ 1, 16, 2 * 128 } );
        DeviceBf16 out( Device::Cuda( 0 ), shape_t{ 1, 16, 8 * 128 } );

        EXPECT_THROW( op.prefill( q, k, v, out, 0 ), std::logic_error );
    }
}
