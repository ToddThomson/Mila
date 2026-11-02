/**
 * @file CudaGqaOp.DecodeTokens.Cuda.cpp
 * @brief A decode of several tokens in one call equals the same tokens decoded one at a time (Gemma4Mtp.md 4.7).
 *
 * Two ops share a history; one decodes R tokens from position p one call at a time, the other in one call. The
 * caches they leave must be equal bit for bit -- the write is the same kernel over the same rows -- and each token's
 * attention output equal within FP32 summation order: the one-call kernel cuts its splits over the tokens' union band,
 * so a token's sums run in another order than its one-token decode's. A structural error -- a token attending past
 * itself, a window start taken from the wrong token, a pair mapped to the wrong head -- moves an output by the size of
 * the output.
 *
 * Compiled only under MILA_ENABLE_CUDA; each test skips if no device is present.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

import Mila;
import Compute.CudaGqaOp;
import Compute.GqaState;
import Compute.KvCacheView;

namespace Mila::Tests::Dnn::Components::Attention
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        using DeviceBf16 = Tensor<TensorDataType::BF16, CudaDeviceMemoryResource>;

        struct Geometry
        {
            int batch;
            int heads;
            int kv_heads;
            int head_dim;
            int window;
            int context;
            int prefill_chunk;
        };

        // Two BF16 steps at the outputs' magnitude (|values| < 1): each side rounds its FP32 result once.
        constexpr float kTolerance = 2e-2f;
    }

    class CudaGqaDecodeTokens : public ::testing::Test
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
                context_ = nullptr;
            }

            if ( !context_ )
                GTEST_SKIP() << "CUDA device not available";
        }

        HostFp32 randomHost( const shape_t& shape, std::mt19937& generator )
        {
            std::uniform_real_distribution<float> distribution( -1.0f, 1.0f );
            HostFp32 host( Device::Cpu(), shape );

            for ( std::size_t i = 0; i < host.size(); ++i )
                host.data()[ i ] = distribution( generator );

            return host;
        }

        DeviceBf16 toDevice( const HostFp32& host )
        {
            DeviceBf16 device( Device::Cuda( 0 ), host.shape() );
            copy( host, device, context_.get() );
            context_->synchronize();

            return device;
        }

        std::vector<float> toVector( const DeviceBf16& device )
        {
            auto host = toHost<TensorDataType::FP32>( device, context_.get() );
            context_->synchronize();

            return std::vector<float>( host.data(), host.data() + host.size() );
        }

        /// Rows [first, first + count) of a [B, T, width] host tensor.
        static HostFp32 rows( const HostFp32& source, int first, int count )
        {
            const auto& shape = source.shape();
            const int64_t batch = shape[ 0 ];
            const int64_t length = shape[ 1 ];
            const int64_t width = shape[ 2 ];
            HostFp32 slice( Device::Cpu(), shape_t{ batch, count, width } );

            for ( int64_t b = 0; b < batch; ++b )
            {
                std::copy_n( source.data() + ( b * length + first ) * width, count * width,
                    slice.data() + b * count * width );
            }

            return slice;
        }

        template<typename TOp>
        std::unique_ptr<TOp> makeOp( const Geometry& geometry, bool fused = true, int decode_tokens = 8 )
        {
            const int model_dim = geometry.heads * geometry.head_dim;
            const int packed = ( geometry.heads + 2 * geometry.kv_heads ) * geometry.head_dim;

            auto op = std::make_unique<TOp>( context_.get(),
                GqaConfig( model_dim, geometry.heads, geometry.kv_heads ).withWindow( geometry.window ) );
            op->build( BuildContext( shape_t{ geometry.batch, geometry.context, packed }, RuntimeMode::Inference, false )
                .withPrefillSize( geometry.prefill_chunk ).withDecodeTokens( decode_tokens ) );
            op->initializeKvCache( geometry.batch, geometry.context );
            op->setUseFlashDecode( fused );

            // Read only by the cuBLASLt decode, which a fused op does not take.
            q_permute_ = std::make_unique<DeviceBf16>( Device::Cuda( 0 ),
                shape_t{ geometry.batch, geometry.heads, 1, geometry.head_dim } );
            scores_ = std::make_unique<DeviceBf16>( Device::Cuda( 0 ),
                shape_t{ geometry.batch, geometry.heads, 1, geometry.context } );
            v_out_ = std::make_unique<DeviceBf16>( Device::Cuda( 0 ),
                shape_t{ geometry.batch, geometry.heads, 1, geometry.head_dim } );

            GqaState state;
            state.q_permute = q_permute_.get();
            state.preatt_decode = scores_.get();
            state.att_decode = scores_.get();
            state.v_out_decode = v_out_.get();
            op->setState( state );

            return op;
        }

        /// One decode call of `count` tokens from `first`; returns its output rows.
        template<typename TOp>
        std::vector<float> decode( TOp& op, const Geometry& geometry, const HostFp32& q, const HostFp32& k,
            const HostFp32& v, int first, int count )
        {
            DeviceBf16 dq = toDevice( rows( q, first, count ) );
            DeviceBf16 dk = toDevice( rows( k, first, count ) );
            DeviceBf16 dv = toDevice( rows( v, first, count ) );
            DeviceBf16 out( Device::Cuda( 0 ), shape_t{ geometry.batch, count, geometry.heads * geometry.head_dim } );

            context_->setDecodePosition( first );
            op.decode( dq, dk, dv, out, first );

            return toVector( out );
        }

        template<typename TOp>
        static std::vector<uint8_t> cacheBytes( const TOp& op, const Geometry& geometry )
        {
            const KvCacheView view = op.getKvCacheView();
            const std::size_t element = view.fp8 ? 1 : 2;
            const std::size_t bytes = static_cast<std::size_t>( geometry.batch ) * view.num_kv_heads * view.capacity
                * view.head_size * element;
            const std::size_t scales = view.fp8
                ? static_cast<std::size_t>( geometry.batch ) * view.num_kv_heads * view.capacity * sizeof( float )
                : 0;

            std::vector<uint8_t> all( 2 * ( bytes + scales ) );
            cudaMemcpy( all.data(), view.keys, bytes, cudaMemcpyDeviceToHost );
            cudaMemcpy( all.data() + bytes, view.values, bytes, cudaMemcpyDeviceToHost );

            if ( view.fp8 )
            {
                cudaMemcpy( all.data() + 2 * bytes, view.key_scales, scales, cudaMemcpyDeviceToHost );
                cudaMemcpy( all.data() + 2 * bytes + scales, view.value_scales, scales, cudaMemcpyDeviceToHost );
            }

            return all;
        }

        /**
         * `depth` tokens of history decoded one at a time into both ops, then `tokens` more: one at a time into the
         * first, in one call into the second.
         */
        template<typename TOp>
        void expectOneCallEqualsOneAtATime( const Geometry& geometry, int depth, int tokens, unsigned seed )
        {
            std::mt19937 generator( seed );
            const int total = depth + tokens;
            const HostFp32 q = randomHost( shape_t{ geometry.batch, total, geometry.heads * geometry.head_dim }, generator );
            const HostFp32 k = randomHost( shape_t{ geometry.batch, total, geometry.kv_heads * geometry.head_dim }, generator );
            const HostFp32 v = randomHost( shape_t{ geometry.batch, total, geometry.kv_heads * geometry.head_dim }, generator );

            auto one_at_a_time = makeOp<TOp>( geometry );
            auto one_call = makeOp<TOp>( geometry );

            for ( int t = 0; t < depth; ++t )
            {
                decode( *one_at_a_time, geometry, q, k, v, t, 1 );
                decode( *one_call, geometry, q, k, v, t, 1 );
            }

            const int width = geometry.heads * geometry.head_dim;
            std::vector<float> expected( static_cast<std::size_t>( geometry.batch ) * tokens * width );

            for ( int r = 0; r < tokens; ++r )
            {
                const std::vector<float> row = decode( *one_at_a_time, geometry, q, k, v, depth + r, 1 );

                for ( int b = 0; b < geometry.batch; ++b )
                {
                    std::copy_n( row.data() + b * width, width, expected.data() + ( b * tokens + r ) * width );
                }
            }

            const std::vector<float> produced = decode( *one_call, geometry, q, k, v, depth, tokens );
            ASSERT_EQ( produced.size(), expected.size() );

            float largest = 0.0f;
            int failures = 0;

            for ( std::size_t i = 0; i < expected.size(); ++i )
            {
                const float difference = std::abs( produced[ i ] - expected[ i ] );
                largest = std::max( largest, difference );

                if ( difference > kTolerance && ++failures <= 8 )
                {
                    const std::size_t token_row = i / width;
                    ADD_FAILURE() << "batch " << token_row / tokens << ", token " << token_row % tokens << ", element "
                        << i % width << ": one call " << produced[ i ] << ", one at a time " << expected[ i ];
                }
            }

            EXPECT_EQ( failures, 0 );
            EXPECT_TRUE( cacheBytes( *one_call, geometry ) == cacheBytes( *one_at_a_time, geometry ) )
                << "the caches the two leave differ";

            std::printf( "  depth %d, %d tokens: largest difference %.2e\n", depth, tokens, largest );
        }

        std::unique_ptr<IExecutionContext> context_;
        std::unique_ptr<DeviceBf16> q_permute_;
        std::unique_ptr<DeviceBf16> scores_;
        std::unique_ptr<DeviceBf16> v_out_;
    };

    namespace
    {
        using UnboundedOp = Compute::Cuda::Gqa::CudaGqaOp<TensorDataType::BF16, false>;
        using BoundedOp = Compute::Cuda::Gqa::CudaGqaOp<TensorDataType::BF16, true>;
        using Fp8Op = Compute::Cuda::Gqa::CudaGqaOp<TensorDataType::BF16, false, true>;

        // Gemma 4's global layer: one KV head for 16 query heads, so 5 tokens are 80 pairs over 5 row tiles; batch 2.
        constexpr Geometry kGemmaGlobal{ 2, 16, 1, 512, 0, 512, 32 };

        // Gemma 4's sliding layer at test scale: 8 KV heads of 2, window 24, a ring of 24 + 32 - 1 = 55 rows.
        constexpr Geometry kGemmaSliding{ 1, 16, 8, 256, 24, 83, 32 };

        // A Llama shape: 4 query heads a KV head, so 8 tokens are 32 pairs over 2 row tiles.
        constexpr Geometry kLlama{ 1, 8, 2, 128, 0, 256, 32 };
    }

    // Depth 200 puts the union band past one split, so the partials and their merge run per (token, head) pair.
    TEST_F( CudaGqaDecodeTokens, GemmaGlobalAcrossSplits )
    {
        expectOneCallEqualsOneAtATime<UnboundedOp>( kGemmaGlobal, 200, 5, 41u );
    }

    TEST_F( CudaGqaDecodeTokens, GemmaGlobalInOneSplit )
    {
        expectOneCallEqualsOneAtATime<UnboundedOp>( kGemmaGlobal, 7, 5, 42u );
    }

    // Before the window: every token attends from position 0.
    TEST_F( CudaGqaDecodeTokens, GemmaSlidingBeforeTheWindow )
    {
        expectOneCallEqualsOneAtATime<BoundedOp>( kGemmaSliding, 3, 5, 43u );
    }

    // Past the window and the ring's wrap: each token's band starts at its own position less the window.
    TEST_F( CudaGqaDecodeTokens, GemmaSlidingPastTheWindowAcrossTheWrap )
    {
        expectOneCallEqualsOneAtATime<BoundedOp>( kGemmaSliding, 60, 8, 44u );
    }

    TEST_F( CudaGqaDecodeTokens, LlamaAcrossRowTilesAndSplits )
    {
        expectOneCallEqualsOneAtATime<UnboundedOp>( kLlama, 150, 8, 45u );
    }

    TEST_F( CudaGqaDecodeTokens, Fp8Cache )
    {
        expectOneCallEqualsOneAtATime<Fp8Op>( kGemmaGlobal, 100, 4, 46u );
    }

    // The cuBLASLt decode's plans are built for one query token: several are refused before the cache is written.
    TEST_F( CudaGqaDecodeTokens, SeveralTokensWithoutTheFusedKernelAreRefused )
    {
        std::mt19937 generator( 47u );
        const HostFp32 q = randomHost( shape_t{ 1, 4, kLlama.heads * kLlama.head_dim }, generator );
        const HostFp32 k = randomHost( shape_t{ 1, 4, kLlama.kv_heads * kLlama.head_dim }, generator );
        const HostFp32 v = randomHost( shape_t{ 1, 4, kLlama.kv_heads * kLlama.head_dim }, generator );

        auto op = makeOp<UnboundedOp>( kLlama, false );
        const std::vector<uint8_t> before = cacheBytes( *op, kLlama );

        EXPECT_THROW( decode( *op, kLlama, q, k, v, 0, 4 ), std::logic_error );
        EXPECT_TRUE( cacheBytes( *op, kLlama ) == before );
    }

    // On the ring, writing the tokens must not evict a key the first token still attends: window + tokens - 1 rows.
    TEST_F( CudaGqaDecodeTokens, MoreTokensThanTheRingKeepsAreRefused )
    {
        const int tokens = 33;
        std::mt19937 generator( 48u );
        const HostFp32 q = randomHost( shape_t{ 1, tokens, kGemmaSliding.heads * kGemmaSliding.head_dim }, generator );
        const HostFp32 k = randomHost( shape_t{ 1, tokens, kGemmaSliding.kv_heads * kGemmaSliding.head_dim }, generator );
        const HostFp32 v = randomHost( shape_t{ 1, tokens, kGemmaSliding.kv_heads * kGemmaSliding.head_dim }, generator );

        auto op = makeOp<BoundedOp>( kGemmaSliding, true, tokens );

        EXPECT_THROW( decode( *op, kGemmaSliding, q, k, v, 0, tokens ), std::invalid_argument );
    }

    // Its scratch reservation holds the tokens the op was built for; more would grow the buffer under a recording.
    TEST_F( CudaGqaDecodeTokens, MoreTokensThanTheOpWasBuiltForAreRefused )
    {
        std::mt19937 generator( 49u );
        const HostFp32 q = randomHost( shape_t{ 1, 5, kLlama.heads * kLlama.head_dim }, generator );
        const HostFp32 k = randomHost( shape_t{ 1, 5, kLlama.kv_heads * kLlama.head_dim }, generator );
        const HostFp32 v = randomHost( shape_t{ 1, 5, kLlama.kv_heads * kLlama.head_dim }, generator );

        auto op = makeOp<UnboundedOp>( kLlama, true, 4 );

        EXPECT_THROW( decode( *op, kLlama, q, k, v, 0, 5 ), std::invalid_argument );
        EXPECT_NO_THROW( decode( *op, kLlama, q, k, v, 0, 4 ) );
    }
}
