/**
 * @file CudaGqaOp.Cuda.cpp
 * @brief Operation-level tests for CudaGqaOp<P, kBounded> — the KV-cache decode
 *        path and the bounded sliding-window ring (SlidingWindowKvCache.md Phase 1).
 *
 * The GroupedQueryAttention component test (GroupedQueryAttention.Cuda.cpp) covers
 * only the component surface; per its scope note the attention NUMERICS and the
 * GqaState / cache lifecycle belong to an operation-level test. This is that test.
 *
 * The bounded ring is validated against the unbounded NoKvCompression op as the
 * ORACLE (SlidingWindowKvCache.md D8): identical random q/k/v are decoded
 * token-by-token through both ops and the outputs compared.
 *   - seq <= window  : no eviction; the ring uses slots in natural order, so the
 *                      result is (near) bit-identical to the full cache.
 *   - seq  > capacity: the ring wraps; the slot->absolute-position softmask must
 *                      still reproduce the windowed full-cache result (summation
 *                      order differs, hence a tolerance rather than equality).
 *
 * Compiled only under MILA_ENABLE_CUDA (Tests/CMakeLists.txt CUDA block); SetUp()
 * skips when no device is present.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

import Mila;
// CudaGqaOp is not on the public umbrella; this suite tests the operation directly.
import Compute.CudaGqaOp;
import Compute.GqaState;

namespace Mila::Tests::Dnn::Components::Attention::GQA::Op
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        // Geometry: GQA with NKV < NH, head_dim large enough that cuBLASLt is happy.
        constexpr int kBatch = 1;
        constexpr int kNumHeads = 8;
        constexpr int kNumKvHeads = 2;
        constexpr int kHeadDim = 64;
        constexpr int kModelDim = kNumHeads * kHeadDim;            // 512
        constexpr int kPackedQkv = ( kNumHeads + 2 * kNumKvHeads ) * kHeadDim; // 768

        constexpr int kContext = 64;        // T_  (max sequence length)
        constexpr int kPrefillChunk = 16;   // prefill size threaded via BuildContext
        constexpr int kWindow = 8;          // sliding window
        // Bounded ring capacity = min(T, window + chunk - 1) = min(64, 23) = 23.
        constexpr int kExpectedCapacity = kWindow + kPrefillChunk - 1;

        struct Fp32Precision
        {
            static constexpr TensorDataType value = TensorDataType::FP32;
            static constexpr float atol = 1e-3f;
            static constexpr size_t bytes = 4;
            static constexpr const char* name = "Fp32";
        };

        struct Bf16Precision
        {
            static constexpr TensorDataType value = TensorDataType::BF16;
            static constexpr float atol = 3e-2f;
            static constexpr size_t bytes = 2;
            static constexpr const char* name = "Bf16";
        };

        using GqaOpPrecisions = ::testing::Types<Fp32Precision, Bf16Precision>;

        class PrecisionNames
        {
        public:
            template<typename TPrecisionTag>
            static std::string GetName( int )
            {
                return TPrecisionTag::name;
            }
        };
    }

    template<typename TPrecisionTag>
    class CudaGqaOpTests : public ::testing::Test
    {
    protected:
        static constexpr TensorDataType P = TPrecisionTag::value;
        static constexpr float kAtol = TPrecisionTag::atol;
        static constexpr size_t kBytes = TPrecisionTag::bytes;

        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        using DeviceTensor = Tensor<P, CudaDeviceMemoryResource>;
        using UnboundedOp = Compute::Cuda::Gqa::CudaGqaOp<P, false>;
        using BoundedOp = Compute::Cuda::Gqa::CudaGqaOp<P, true>;

        void SetUp() override
        {
            try
            {
                cuda_context_ = createExecutionContext( Device::Cuda( 0 ) );
            }
            catch ( const std::exception& )
            {
                cuda_context_ = nullptr;
            }

            if ( !cuda_context_ )
            {
                GTEST_SKIP() << "CUDA device not available";
            }
        }

        GqaConfig config( int window )
        {
            return GqaConfig( kModelDim, kNumHeads, kNumKvHeads ).withWindow( window );
        }

        // Random per-position projections, identical across both ops.
        HostFp32 randomHost( const shape_t& shape, std::mt19937& rng )
        {
            std::uniform_real_distribution<float> dist( -1.0f, 1.0f );
            HostFp32 host( Device::Cpu(), shape );

            for ( size_t i = 0; i < host.size(); ++i )
                host.data()[ i ] = dist( rng );

            return host;
        }

        DeviceTensor toDevice( const HostFp32& host )
        {
            DeviceTensor device( Device::Cuda( 0 ), host.shape() );
            copy( host, device, cuda_context_.get() );
            cuda_context_->synchronize();

            return device;
        }

        HostFp32 toFloat( const DeviceTensor& device )
        {
            auto host = toHost<TensorDataType::FP32>( device, cuda_context_.get() );
            cuda_context_->synchronize();

            return host;
        }

        // Decode the supplied sequence token-by-token through `op`, returning the
        // per-position output rows (flattened [modelDim]). The op must already be
        // built and have its KV cache initialized.
        template<typename OpT>
        std::vector<std::vector<float>> runDecodeSequence(
            OpT& op,
            const std::vector<HostFp32>& qs,
            const std::vector<HostFp32>& ks,
            const std::vector<HostFp32>& vs )
        {
            // Shared transient scratch wired via setState. preatt/att decode buffers
            // are sized at the full context so the same allocation serves both the
            // bounded (capacity-prefix) and unbounded (full-T) ops.
            DeviceTensor q_permute( Device::Cuda( 0 ), shape_t{ kBatch, kNumHeads, 1, kHeadDim } );
            DeviceTensor preatt_decode( Device::Cuda( 0 ), shape_t{ kBatch, kNumHeads, 1, kContext } );
            DeviceTensor att_decode( Device::Cuda( 0 ), shape_t{ kBatch, kNumHeads, 1, kContext } );
            DeviceTensor v_out_decode( Device::Cuda( 0 ), shape_t{ kBatch, kNumHeads, 1, kHeadDim } );

            GqaState state;
            state.q_permute = &q_permute;
            state.preatt_decode = &preatt_decode;
            state.att_decode = &att_decode;
            state.v_out_decode = &v_out_decode;
            op.setState( state );

            std::vector<std::vector<float>> outputs;
            outputs.reserve( qs.size() );

            for ( size_t t = 0; t < qs.size(); ++t )
            {
                DeviceTensor q = toDevice( qs[ t ] );
                DeviceTensor k = toDevice( ks[ t ] );
                DeviceTensor v = toDevice( vs[ t ] );
                DeviceTensor out( Device::Cuda( 0 ), shape_t{ kBatch, 1, kModelDim } );

                op.decode( q, k, v, out, static_cast<dim_t>( t ) );

                HostFp32 host = toFloat( out );
                outputs.emplace_back( host.data(), host.data() + host.size() );
            }

            return outputs;
        }

        // Build oracle + bounded ops, decode `seq` identical random tokens through
        // both, and assert the max absolute output difference is within tolerance.
        void expectBoundedMatchesOracle( int seq )
        {
            std::mt19937 rng( 1234u );

            std::vector<HostFp32> qs, ks, vs;
            for ( int t = 0; t < seq; ++t )
            {
                qs.push_back( randomHost( shape_t{ kBatch, 1, kNumHeads * kHeadDim }, rng ) );
                ks.push_back( randomHost( shape_t{ kBatch, 1, kNumKvHeads * kHeadDim }, rng ) );
                vs.push_back( randomHost( shape_t{ kBatch, 1, kNumKvHeads * kHeadDim }, rng ) );
            }

            auto build = [&]( auto& op )
            {
                op.build( BuildContext( shape_t{ kBatch, kContext, kPackedQkv },
                    RuntimeMode::Inference, false ).withPrefillSize( kPrefillChunk ) );
                op.initializeKvCache( kBatch, kContext );
            };

            UnboundedOp oracle( this->cuda_context_.get(), this->config( kWindow ) );
            BoundedOp bounded( this->cuda_context_.get(), this->config( kWindow ) );
            build( oracle );
            build( bounded );

            ASSERT_EQ( oracle.getCacheCapacity(), kContext );
            ASSERT_EQ( bounded.getCacheCapacity(), std::min( kContext, kExpectedCapacity ) );

            auto out_oracle = runDecodeSequence( oracle, qs, ks, vs );
            auto out_bounded = runDecodeSequence( bounded, qs, ks, vs );

            ASSERT_EQ( out_oracle.size(), out_bounded.size() );

            float max_diff = 0.0f;
            for ( size_t t = 0; t < out_oracle.size(); ++t )
            {
                ASSERT_EQ( out_oracle[ t ].size(), out_bounded[ t ].size() );

                for ( size_t i = 0; i < out_oracle[ t ].size(); ++i )
                    max_diff = std::max( max_diff, std::fabs( out_oracle[ t ][ i ] - out_bounded[ t ][ i ] ) );
            }

            EXPECT_LT( max_diff, kAtol ) << "bounded decode diverged from full-cache oracle at seq=" << seq;
        }

        // Generate a prefill schedule: per-chunk random q/k/v for [0, seq) split into
        // chunks of `chunk_size` (the final chunk is partial when seq % chunk_size != 0).
        void genChunks(
            int seq, int chunk_size, std::mt19937& rng,
            std::vector<HostFp32>& qChunks, std::vector<HostFp32>& kChunks,
            std::vector<HostFp32>& vChunks, std::vector<int>& offsets )
        {
            for ( int off = 0; off < seq; off += chunk_size )
            {
                const int clen = std::min( chunk_size, seq - off );
                qChunks.push_back( randomHost( shape_t{ kBatch, clen, kNumHeads * kHeadDim }, rng ) );
                kChunks.push_back( randomHost( shape_t{ kBatch, clen, kNumKvHeads * kHeadDim }, rng ) );
                vChunks.push_back( randomHost( shape_t{ kBatch, clen, kNumKvHeads * kHeadDim }, rng ) );
                offsets.push_back( off );
            }
        }

        // Run a full session: chunked prefill followed by single-token decode steps.
        // All seven GqaState scratch slots are wired so prefill and decode share one op.
        // Returns the per-position output rows (flattened [modelDim]).
        template<typename OpT>
        std::vector<std::vector<float>> runSession(
            OpT& op,
            const std::vector<HostFp32>& qChunks, const std::vector<HostFp32>& kChunks,
            const std::vector<HostFp32>& vChunks, const std::vector<int>& offsets,
            const std::vector<HostFp32>& qDec, const std::vector<HostFp32>& kDec,
            const std::vector<HostFp32>& vDec, dim_t decodeStart )
        {
            DeviceTensor q_permute( Device::Cuda( 0 ), shape_t{ kBatch, kNumHeads, kPrefillChunk, kHeadDim } );
            DeviceTensor preatt( Device::Cuda( 0 ), shape_t{ kBatch, kNumHeads, kPrefillChunk, kContext } );
            DeviceTensor att( Device::Cuda( 0 ), shape_t{ kBatch, kNumHeads, kPrefillChunk, kContext } );
            DeviceTensor v_out( Device::Cuda( 0 ), shape_t{ kBatch, kNumHeads, kPrefillChunk, kHeadDim } );
            DeviceTensor preatt_decode( Device::Cuda( 0 ), shape_t{ kBatch, kNumHeads, 1, kContext } );
            DeviceTensor att_decode( Device::Cuda( 0 ), shape_t{ kBatch, kNumHeads, 1, kContext } );
            DeviceTensor v_out_decode( Device::Cuda( 0 ), shape_t{ kBatch, kNumHeads, 1, kHeadDim } );

            GqaState state;
            state.q_permute = &q_permute;
            state.preatt = &preatt;
            state.att = &att;
            state.v_out = &v_out;
            state.preatt_decode = &preatt_decode;
            state.att_decode = &att_decode;
            state.v_out_decode = &v_out_decode;
            op.setState( state );

            std::vector<std::vector<float>> outputs;

            for ( size_t c = 0; c < qChunks.size(); ++c )
            {
                const int clen = static_cast<int>( qChunks[ c ].shape()[ 1 ] );
                DeviceTensor q = toDevice( qChunks[ c ] );
                DeviceTensor k = toDevice( kChunks[ c ] );
                DeviceTensor v = toDevice( vChunks[ c ] );
                DeviceTensor out( Device::Cuda( 0 ), shape_t{ kBatch, clen, kModelDim } );

                op.prefill( q, k, v, out, offsets[ c ] );

                HostFp32 host = toFloat( out );
                for ( int t = 0; t < clen; ++t )
                    outputs.emplace_back( host.data() + t * kModelDim, host.data() + ( t + 1 ) * kModelDim );
            }

            for ( size_t d = 0; d < qDec.size(); ++d )
            {
                DeviceTensor q = toDevice( qDec[ d ] );
                DeviceTensor k = toDevice( kDec[ d ] );
                DeviceTensor v = toDevice( vDec[ d ] );
                DeviceTensor out( Device::Cuda( 0 ), shape_t{ kBatch, 1, kModelDim } );

                op.decode( q, k, v, out, decodeStart + static_cast<dim_t>( d ) );

                HostFp32 host = toFloat( out );
                outputs.emplace_back( host.data(), host.data() + host.size() );
            }

            return outputs;
        }

        // Build oracle + bounded ops, run an identical prefill (+ optional decode tail)
        // through both, and assert the max absolute output difference is within tolerance.
        void expectSessionMatchesOracle( int prefillSeq, int decodeCount )
        {
            std::mt19937 rng( 4242u );

            std::vector<HostFp32> qC, kC, vC;
            std::vector<int> offs;
            genChunks( prefillSeq, kPrefillChunk, rng, qC, kC, vC, offs );

            std::vector<HostFp32> qD, kD, vD;
            for ( int d = 0; d < decodeCount; ++d )
            {
                qD.push_back( randomHost( shape_t{ kBatch, 1, kNumHeads * kHeadDim }, rng ) );
                kD.push_back( randomHost( shape_t{ kBatch, 1, kNumKvHeads * kHeadDim }, rng ) );
                vD.push_back( randomHost( shape_t{ kBatch, 1, kNumKvHeads * kHeadDim }, rng ) );
            }

            auto build = [&]( auto& op )
            {
                op.build( BuildContext( shape_t{ kBatch, kContext, kPackedQkv },
                    RuntimeMode::Inference, false ).withPrefillSize( kPrefillChunk ) );
                op.initializeKvCache( kBatch, kContext );
            };

            UnboundedOp oracle( this->cuda_context_.get(), this->config( kWindow ) );
            BoundedOp bounded( this->cuda_context_.get(), this->config( kWindow ) );
            build( oracle );
            build( bounded );

            auto out_oracle = runSession( oracle, qC, kC, vC, offs, qD, kD, vD, prefillSeq );
            auto out_bounded = runSession( bounded, qC, kC, vC, offs, qD, kD, vD, prefillSeq );

            ASSERT_EQ( out_oracle.size(), out_bounded.size() );

            float max_diff = 0.0f;
            for ( size_t t = 0; t < out_oracle.size(); ++t )
            {
                ASSERT_EQ( out_oracle[ t ].size(), out_bounded[ t ].size() );

                for ( size_t i = 0; i < out_oracle[ t ].size(); ++i )
                    max_diff = std::max( max_diff, std::fabs( out_oracle[ t ][ i ] - out_bounded[ t ][ i ] ) );
            }

            EXPECT_LT( max_diff, kAtol )
                << "bounded session diverged from oracle (prefill=" << prefillSeq
                << ", decode=" << decodeCount << ")";
        }

        std::unique_ptr<IExecutionContext> cuda_context_;
    };

    TYPED_TEST_SUITE( CudaGqaOpTests, GqaOpPrecisions, PrecisionNames );

    // ====================================================================
    // Capacity resolution (SlidingWindowKvCache.md 7.1, D2)
    // ====================================================================

    TYPED_TEST( CudaGqaOpTests, UnboundedCapacity_EqualsContext )
    {
        typename TestFixture::UnboundedOp op( this->cuda_context_.get(), this->config( kWindow ) );
        op.build( BuildContext( shape_t{ kBatch, kContext, kPackedQkv },
            RuntimeMode::Inference, false ).withPrefillSize( kPrefillChunk ) );

        EXPECT_EQ( op.getCacheCapacity(), kContext );
    }

    TYPED_TEST( CudaGqaOpTests, BoundedCapacity_IsWindowPlusChunkMinusOne )
    {
        typename TestFixture::BoundedOp op( this->cuda_context_.get(), this->config( kWindow ) );
        op.build( BuildContext( shape_t{ kBatch, kContext, kPackedQkv },
            RuntimeMode::Inference, false ).withPrefillSize( kPrefillChunk ) );

        EXPECT_EQ( op.getCacheCapacity(), std::min( kContext, kExpectedCapacity ) );
    }

    TYPED_TEST( CudaGqaOpTests, BoundedCapacity_ClampsToContextWhenWindowLarge )
    {
        // window >= T: capacity clamps to the context length (ring degenerates).
        typename TestFixture::BoundedOp op( this->cuda_context_.get(), this->config( 2 * kContext ) );
        op.build( BuildContext( shape_t{ kBatch, kContext, kPackedQkv },
            RuntimeMode::Inference, false ).withPrefillSize( kPrefillChunk ) );

        EXPECT_EQ( op.getCacheCapacity(), kContext );
    }

    TYPED_TEST( CudaGqaOpTests, BoundedOp_RequiresPositiveWindow_Throws )
    {
        typename TestFixture::BoundedOp op( this->cuda_context_.get(), this->config( 0 ) );

        EXPECT_THROW(
            op.build( BuildContext( shape_t{ kBatch, kContext, kPackedQkv },
                RuntimeMode::Inference, false ).withPrefillSize( kPrefillChunk ) ),
            std::invalid_argument );
    }

    // ====================================================================
    // KV-cache footprint (the gate-2 payoff, asserted closed-form)
    //
    // getStateMemorySize() reports exactly the retained KV cache (k + v); the
    // transient scratch is external via setState. So the footprint is deterministic:
    //   2 (K+V) * B * NKV * cache_rows * HS * sizeof(dtype),
    // with cache_rows = T (unbounded) or min(T, window+chunk-1) (bounded ring).
    // ====================================================================

    TYPED_TEST( CudaGqaOpTests, StateMemory_MatchesClosedFormAndShrinks )
    {
        typename TestFixture::UnboundedOp oracle( this->cuda_context_.get(), this->config( kWindow ) );
        typename TestFixture::BoundedOp bounded( this->cuda_context_.get(), this->config( kWindow ) );

        const auto ctx = BuildContext( shape_t{ kBatch, kContext, kPackedQkv },
            RuntimeMode::Inference, false ).withPrefillSize( kPrefillChunk );
        oracle.build( ctx );
        bounded.build( ctx );

        const int capacity = std::min( kContext, kExpectedCapacity );
        const size_t bytes = TestFixture::kBytes;

        const size_t oracle_bytes =
            size_t( 2 ) * kBatch * kNumKvHeads * kContext * kHeadDim * bytes;
        const size_t bounded_bytes =
            size_t( 2 ) * kBatch * kNumKvHeads * capacity * kHeadDim * bytes;

        EXPECT_EQ( oracle.getStateMemorySize(), oracle_bytes );
        EXPECT_EQ( bounded.getStateMemorySize(), bounded_bytes );

        // The bounded ring must actually be smaller for a context past the window
        // working set (kContext 64 > capacity 23 here).
        EXPECT_LT( bounded.getStateMemorySize(), oracle.getStateMemorySize() );
    }

    // ====================================================================
    // KV-cache rewind (prompt-prefix reuse, PromptCaching.md)
    //
    // rewindKvCache moves only the logical fill position; device contents are
    // untouched. The unbounded cache accepts any position within the current
    // fill. The bounded ring must additionally refuse when the stale tail
    // [position, cached) has wrapped over the OLD rows [position - window,
    // position) a continuation would attend to:
    //   valid  <=>  cached - position <= capacity - window  (= chunk - 1)
    // ====================================================================

    TYPED_TEST( CudaGqaOpTests, RewindKvCache_UnboundedAcceptsAnyPositionWithinFill )
    {
        constexpr int kSeq = 20;
        std::mt19937 rng( 77u );

        std::vector<typename TestFixture::HostFp32> qs, ks, vs;
        for ( int t = 0; t < kSeq; ++t )
        {
            qs.push_back( this->randomHost( shape_t{ kBatch, 1, kNumHeads * kHeadDim }, rng ) );
            ks.push_back( this->randomHost( shape_t{ kBatch, 1, kNumKvHeads * kHeadDim }, rng ) );
            vs.push_back( this->randomHost( shape_t{ kBatch, 1, kNumKvHeads * kHeadDim }, rng ) );
        }

        typename TestFixture::UnboundedOp op( this->cuda_context_.get(), this->config( kWindow ) );
        op.build( BuildContext( shape_t{ kBatch, kContext, kPackedQkv },
            RuntimeMode::Inference, false ).withPrefillSize( kPrefillChunk ) );
        op.initializeKvCache( kBatch, kContext );

        this->runDecodeSequence( op, qs, ks, vs );

        EXPECT_FALSE( op.rewindKvCache( kSeq + 1 ) );  // beyond the fill
        EXPECT_TRUE( op.rewindKvCache( kSeq ) );       // no-op boundary
        EXPECT_TRUE( op.rewindKvCache( 5 ) );          // deep rewind: always valid unbounded
        EXPECT_FALSE( op.rewindKvCache( 6 ) );         // fill is now 5; forward rewinds refused
    }

    TYPED_TEST( CudaGqaOpTests, RewindKvCache_BoundedRingEnforcesWindowValidity )
    {
        // Decode past the ring capacity so the ring has wrapped before probing.
        constexpr int kSeq = 40;  // > capacity 23
        constexpr int kStaleTolerance = kExpectedCapacity - kWindow;  // chunk - 1 = 15
        std::mt19937 rng( 99u );

        std::vector<typename TestFixture::HostFp32> qs, ks, vs;
        for ( int t = 0; t < kSeq; ++t )
        {
            qs.push_back( this->randomHost( shape_t{ kBatch, 1, kNumHeads * kHeadDim }, rng ) );
            ks.push_back( this->randomHost( shape_t{ kBatch, 1, kNumKvHeads * kHeadDim }, rng ) );
            vs.push_back( this->randomHost( shape_t{ kBatch, 1, kNumKvHeads * kHeadDim }, rng ) );
        }

        typename TestFixture::BoundedOp bounded( this->cuda_context_.get(), this->config( kWindow ) );
        bounded.build( BuildContext( shape_t{ kBatch, kContext, kPackedQkv },
            RuntimeMode::Inference, false ).withPrefillSize( kPrefillChunk ) );
        bounded.initializeKvCache( kBatch, kContext );

        this->runDecodeSequence( bounded, qs, ks, vs );

        // Stale tail one past the tolerance: the window rows are gone -- refuse,
        // and the fill position must be unchanged by the refusal.
        EXPECT_FALSE( bounded.rewindKvCache( kSeq - kStaleTolerance - 1 ) );
        EXPECT_FALSE( bounded.rewindKvCache( kSeq + 1 ) );

        // Exactly at the tolerance: the oldest needed row is still resident.
        EXPECT_TRUE( bounded.rewindKvCache( kSeq - kStaleTolerance ) );

        // The fill is now kSeq - kStaleTolerance; positions past it are refused.
        EXPECT_FALSE( bounded.rewindKvCache( kSeq - kStaleTolerance + 1 ) );
    }

    // ====================================================================
    // Decode parity vs the full-cache oracle (D8)
    // ====================================================================

    TYPED_TEST( CudaGqaOpTests, Decode_NoEviction_MatchesOracle )
    {
        // seq <= window: every key stays resident; the ring uses natural-order slots.
        this->expectBoundedMatchesOracle( kWindow );
    }

    TYPED_TEST( CudaGqaOpTests, Decode_PastWindow_MatchesOracle )
    {
        // seq > capacity > window: the ring wraps and the slot->abs softmask must
        // reproduce the windowed full-cache result.
        this->expectBoundedMatchesOracle( kContext - kPrefillChunk );
    }

    // ====================================================================
    // Prefill parity vs the full-cache oracle (D8)
    // ====================================================================

    TYPED_TEST( CudaGqaOpTests, Prefill_SingleChunk_MatchesOracle )
    {
        // One chunk, chunk < capacity: no eviction; exercises intra-chunk causal
        // masking on the ring.
        this->expectSessionMatchesOracle( kPrefillChunk, 0 );
    }

    TYPED_TEST( CudaGqaOpTests, Prefill_MultiChunkAcrossWindow_MatchesOracle )
    {
        // 3 full chunks (48 > capacity 23 > window 8): the ring wraps across chunk
        // boundaries and evicts; each chunk's needed span must stay resident.
        this->expectSessionMatchesOracle( 3 * kPrefillChunk, 0 );
    }

    TYPED_TEST( CudaGqaOpTests, Prefill_PartialFinalChunk_MatchesOracle )
    {
        // 2 full chunks + a partial final chunk (40 = 16 + 16 + 8): exercises the
        // partial-chunk plan + ring softmask row pitch.
        this->expectSessionMatchesOracle( 2 * kPrefillChunk + kPrefillChunk / 2, 0 );
    }

    TYPED_TEST( CudaGqaOpTests, PrefillThenDecode_MatchesOracle )
    {
        // Realistic generation: chunked prefill over [0, 32) then decode [32, 48)
        // over the same shared ring buffer.
        this->expectSessionMatchesOracle( 2 * kPrefillChunk, kPrefillChunk );
    }

    // ====================================================================
    // FlashAttention prefill parity (GqaFlashAttention.md 10.1)
    //
    // Diffs the fused flash prefill against the cuBLASLt QK->softmax->AV path on
    // the SAME unbounded BF16 op, toggled per-run via setUseFlashPrefill(). This
    // is the direct oracle the bounded-vs-unbounded tests above cannot provide:
    // they use the unbounded op AS their reference, and only at window 8, so
    // flash's actual production config (the Gemma global layer) had no op-level
    // check -- an HS-cap bug in the kernel therefore surfaced only 48 layers deep
    // in the model parity test. The geometry here is that config exactly:
    // head_dim 512 (the global_head_dim, NOT the sliding 256), single MQA KV head,
    // window 0 (full causal), multi-chunk so later chunks attend historical cache
    // (position_offset > 0) plus a partial final chunk. head_dim 512 is the value
    // that overflowed the kernel's per-lane register arrays in Iteration 1 -- this
    // test is the standing regression guard for that class of bug.
    // ====================================================================

    namespace
    {
        constexpr int kGBatch = 1;
        constexpr int kGNumHeads = 16;     // Gemma 4 12B query heads
        constexpr int kGNumKvHeads = 1;    // global layers are MQA (single shared KV head)
        constexpr int kGHeadDim = 512;     // global_head_dim (NOT the sliding head_dim 256)
        constexpr int kGModelDim = kGNumHeads * kGHeadDim;                         // 8192
        constexpr int kGPackedQkv = ( kGNumHeads + 2 * kGNumKvHeads ) * kGHeadDim; // 9216
        // 83 (not 80) is deliberate: the 19-token tail chunk is NOT a multiple of the
        // WMMA kernel's 16-row query tile, so the ragged-tile masking is exercised, and
        // its final key tile runs past cache_capacity, exercising the cp.async OOB row
        // clamp. A multiple-of-16 context covers neither.
        constexpr int kGContext = 83;      // T_
        constexpr int kGPrefillChunk = 32; // 83 tokens -> chunks 32 + 32 + 19 (ragged tail)
        constexpr int kGWindow = 0;        // global / full causal

        // Flash keeps QK scores in FP32; the cuBLASLt reference rounds preatt to BF16
        // before softmax, so the two differ by that rounding, not by algorithm. A gross
        // kernel bug (the HS=512 register overflow this guards) diverges by O(1), an
        // order of magnitude above this bound.
        constexpr float kGFlashAtol = 3e-2f;
    }

    class CudaGqaFlashPrefillParity : public ::testing::Test
    {
    protected:
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        using DeviceBf16 = Tensor<TensorDataType::BF16, CudaDeviceMemoryResource>;
        using UnboundedBf16Op = Compute::Cuda::Gqa::CudaGqaOp<TensorDataType::BF16, false>;

        void SetUp() override
        {
            try
            {
                cuda_context_ = createExecutionContext( Device::Cuda( 0 ) );
            }
            catch ( const std::exception& )
            {
                cuda_context_ = nullptr;
            }

            if ( !cuda_context_ )
                GTEST_SKIP() << "CUDA device not available";
        }

        HostFp32 randomHost( const shape_t& shape, std::mt19937& rng )
        {
            std::uniform_real_distribution<float> dist( -1.0f, 1.0f );
            HostFp32 host( Device::Cpu(), shape );

            for ( size_t i = 0; i < host.size(); ++i )
                host.data()[ i ] = dist( rng );

            return host;
        }

        DeviceBf16 toDevice( const HostFp32& host )
        {
            DeviceBf16 device( Device::Cuda( 0 ), host.shape() );
            copy( host, device, cuda_context_.get() );
            cuda_context_->synchronize();

            return device;
        }

        HostFp32 toFloat( const DeviceBf16& device )
        {
            auto host = toHost<TensorDataType::FP32>( device, cuda_context_.get() );
            cuda_context_->synchronize();

            return host;
        }

        // Chunked prefill of kGContext tokens through a fresh unbounded BF16 op with the
        // flash path toggled on/off. The random q/k/v schedule is fixed by seed so both
        // runs see identical input; returns per-position output rows [kGModelDim].
        std::vector<std::vector<float>> runPrefill( bool useFlash )
        {
            std::mt19937 rng( 7u );

            std::vector<HostFp32> qC, kC, vC;
            std::vector<int> offs;
            for ( int off = 0; off < kGContext; off += kGPrefillChunk )
            {
                const int clen = std::min( kGPrefillChunk, kGContext - off );
                qC.push_back( randomHost( shape_t{ kGBatch, clen, kGNumHeads * kGHeadDim }, rng ) );
                kC.push_back( randomHost( shape_t{ kGBatch, clen, kGNumKvHeads * kGHeadDim }, rng ) );
                vC.push_back( randomHost( shape_t{ kGBatch, clen, kGNumKvHeads * kGHeadDim }, rng ) );
                offs.push_back( off );
            }

            UnboundedBf16Op op( cuda_context_.get(),
                GqaConfig( kGModelDim, kGNumHeads, kGNumKvHeads ).withWindow( kGWindow ) );
            op.build( BuildContext( shape_t{ kGBatch, kGContext, kGPackedQkv },
                RuntimeMode::Inference, false ).withPrefillSize( kGPrefillChunk ) );
            op.initializeKvCache( kGBatch, kGContext );
            op.setUseFlashPrefill( useFlash );

            // Prefill scratch: consumed by the cuBLASLt path, ignored by flash.
            DeviceBf16 q_permute( Device::Cuda( 0 ), shape_t{ kGBatch, kGNumHeads, kGPrefillChunk, kGHeadDim } );
            DeviceBf16 preatt( Device::Cuda( 0 ), shape_t{ kGBatch, kGNumHeads, kGPrefillChunk, kGContext } );
            DeviceBf16 att( Device::Cuda( 0 ), shape_t{ kGBatch, kGNumHeads, kGPrefillChunk, kGContext } );
            DeviceBf16 v_out( Device::Cuda( 0 ), shape_t{ kGBatch, kGNumHeads, kGPrefillChunk, kGHeadDim } );

            GqaState state;
            state.q_permute = &q_permute;
            state.preatt = &preatt;
            state.att = &att;
            state.v_out = &v_out;
            op.setState( state );

            std::vector<std::vector<float>> outputs;
            for ( size_t c = 0; c < qC.size(); ++c )
            {
                const int clen = static_cast<int>( qC[ c ].shape()[ 1 ] );
                DeviceBf16 q = toDevice( qC[ c ] );
                DeviceBf16 k = toDevice( kC[ c ] );
                DeviceBf16 v = toDevice( vC[ c ] );
                DeviceBf16 out( Device::Cuda( 0 ), shape_t{ kGBatch, clen, kGModelDim } );

                op.prefill( q, k, v, out, offs[ c ] );

                HostFp32 host = toFloat( out );
                for ( int t = 0; t < clen; ++t )
                    outputs.emplace_back( host.data() + t * kGModelDim, host.data() + ( t + 1 ) * kGModelDim );
            }

            return outputs;
        }

        std::unique_ptr<IExecutionContext> cuda_context_;
    };

    TEST_F( CudaGqaFlashPrefillParity, FlashMatchesCublasLt_GemmaGlobalConfig )
    {
        auto flashOut = runPrefill( true );
        auto cublasOut = runPrefill( false );

        ASSERT_EQ( flashOut.size(), cublasOut.size() );
        ASSERT_EQ( static_cast<int>( flashOut.size() ), kGContext );

        float max_diff = 0.0f;
        for ( size_t t = 0; t < flashOut.size(); ++t )
        {
            ASSERT_EQ( flashOut[ t ].size(), cublasOut[ t ].size() );

            for ( size_t i = 0; i < flashOut[ t ].size(); ++i )
                max_diff = std::max( max_diff, std::fabs( flashOut[ t ][ i ] - cublasOut[ t ][ i ] ) );
        }

        EXPECT_LT( max_diff, kGFlashAtol )
            << "flash prefill diverged from cuBLASLt at the Gemma global config "
               "(HS=512, NKV=1, window=0), max_diff=" << max_diff;
    }

    // ====================================================================
    // Bounded-ring FlashAttention prefill parity (GqaFlashAttention.md 10.1)
    //
    // Diffs the bounded flash prefill (kBoundedRing kernel variant: ring row
    // mapping in cp.async + band-start key loop) against the cuBLASLt ring
    // pipeline (QK -> prefill_softmax_ring -> AV) on the SAME bounded BF16 op,
    // toggled per-run via setUseFlashPrefill(). The geometry is the Gemma 4 12B
    // LOCAL sliding layer exactly: head_dim 256 (W=4 / hs_tile 64 / 8 n-tiles --
    // the 2-blocks-per-SM configuration), NKV 8 (GS=2), context 83 so the final
    // 19-token chunk exercises the ragged query tile. Two windows split the risk:
    //   - kLWindowWrap 24: capacity 55 < 83, the ring WRAPS mid-prefill -- the
    //     cp.async ring source math and the masked-alias argument are on trial.
    //   - kLWindowNoWrap 64: capacity clamps to 83, the ring mapping is the
    //     identity -- isolates the band-start loop bound and window masking.
    // PTX-change gate discipline: any edit to the kernel's address paths must
    // also run compute-sanitizer memcheck over this fixture (a generic-address
    // ldmatrix bug once passed the parity oracle while doing UB at scale).
    // ====================================================================

    namespace
    {
        constexpr int kLBatch = 1;
        constexpr int kLNumHeads = 16;     // Gemma 4 12B query heads
        constexpr int kLNumKvHeads = 8;    // local sliding layers are GQA (GS = 2)
        constexpr int kLHeadDim = 256;     // sliding head_dim (NOT the global 512)
        constexpr int kLModelDim = kLNumHeads * kLHeadDim;                          // 4096
        constexpr int kLPackedQkv = ( kLNumHeads + 2 * kLNumKvHeads ) * kLHeadDim;  // 8192
        constexpr int kLContext = 83;      // T_ (ragged 19-token tail chunk)
        constexpr int kLPrefillChunk = 32; // 83 tokens -> chunks 32 + 32 + 19
        constexpr int kLWindowWrap = 24;   // capacity min(83, 24+32-1) = 55 -> ring wraps
        constexpr int kLWindowNoWrap = 64; // capacity min(83, 64+32-1) = 83 -> identity ring
        constexpr float kLFlashAtol = 3e-2f;
    }

    class CudaGqaFlashRingPrefillParity : public ::testing::Test
    {
    protected:
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        using DeviceBf16 = Tensor<TensorDataType::BF16, CudaDeviceMemoryResource>;
        using BoundedBf16Op = Compute::Cuda::Gqa::CudaGqaOp<TensorDataType::BF16, true>;

        void SetUp() override
        {
            try
            {
                cuda_context_ = createExecutionContext( Device::Cuda( 0 ) );
            }
            catch ( const std::exception& )
            {
                cuda_context_ = nullptr;
            }

            if ( !cuda_context_ )
                GTEST_SKIP() << "CUDA device not available";
        }

        HostFp32 randomHost( const shape_t& shape, std::mt19937& rng )
        {
            std::uniform_real_distribution<float> dist( -1.0f, 1.0f );
            HostFp32 host( Device::Cpu(), shape );

            for ( size_t i = 0; i < host.size(); ++i )
                host.data()[ i ] = dist( rng );

            return host;
        }

        DeviceBf16 toDevice( const HostFp32& host )
        {
            DeviceBf16 device( Device::Cuda( 0 ), host.shape() );
            copy( host, device, cuda_context_.get() );
            cuda_context_->synchronize();

            return device;
        }

        HostFp32 toFloat( const DeviceBf16& device )
        {
            auto host = toHost<TensorDataType::FP32>( device, cuda_context_.get() );
            cuda_context_->synchronize();

            return host;
        }

        // Chunked prefill of kLContext tokens through a fresh bounded BF16 op with the
        // flash path toggled on/off. The random q/k/v schedule is fixed by seed so both
        // runs see identical input; returns per-position output rows [kLModelDim].
        std::vector<std::vector<float>> runPrefill( bool useFlash, int window )
        {
            std::mt19937 rng( 11u );

            std::vector<HostFp32> qC, kC, vC;
            std::vector<int> offs;
            for ( int off = 0; off < kLContext; off += kLPrefillChunk )
            {
                const int clen = std::min( kLPrefillChunk, kLContext - off );
                qC.push_back( randomHost( shape_t{ kLBatch, clen, kLNumHeads * kLHeadDim }, rng ) );
                kC.push_back( randomHost( shape_t{ kLBatch, clen, kLNumKvHeads * kLHeadDim }, rng ) );
                vC.push_back( randomHost( shape_t{ kLBatch, clen, kLNumKvHeads * kLHeadDim }, rng ) );
                offs.push_back( off );
            }

            BoundedBf16Op op( cuda_context_.get(),
                GqaConfig( kLModelDim, kLNumHeads, kLNumKvHeads ).withWindow( window ) );
            op.build( BuildContext( shape_t{ kLBatch, kLContext, kLPackedQkv },
                RuntimeMode::Inference, false ).withPrefillSize( kLPrefillChunk ) );
            op.initializeKvCache( kLBatch, kLContext );
            op.setUseFlashPrefill( useFlash );

            // Prefill scratch: consumed by the cuBLASLt ring path, ignored by flash.
            // preatt/att are context-wide, a superset of the capacity columns it uses.
            DeviceBf16 q_permute( Device::Cuda( 0 ), shape_t{ kLBatch, kLNumHeads, kLPrefillChunk, kLHeadDim } );
            DeviceBf16 preatt( Device::Cuda( 0 ), shape_t{ kLBatch, kLNumHeads, kLPrefillChunk, kLContext } );
            DeviceBf16 att( Device::Cuda( 0 ), shape_t{ kLBatch, kLNumHeads, kLPrefillChunk, kLContext } );
            DeviceBf16 v_out( Device::Cuda( 0 ), shape_t{ kLBatch, kLNumHeads, kLPrefillChunk, kLHeadDim } );

            GqaState state;
            state.q_permute = &q_permute;
            state.preatt = &preatt;
            state.att = &att;
            state.v_out = &v_out;
            op.setState( state );

            std::vector<std::vector<float>> outputs;
            for ( size_t c = 0; c < qC.size(); ++c )
            {
                const int clen = static_cast<int>( qC[ c ].shape()[ 1 ] );
                DeviceBf16 q = toDevice( qC[ c ] );
                DeviceBf16 k = toDevice( kC[ c ] );
                DeviceBf16 v = toDevice( vC[ c ] );
                DeviceBf16 out( Device::Cuda( 0 ), shape_t{ kLBatch, clen, kLModelDim } );

                op.prefill( q, k, v, out, offs[ c ] );

                HostFp32 host = toFloat( out );
                for ( int t = 0; t < clen; ++t )
                    outputs.emplace_back( host.data() + t * kLModelDim, host.data() + ( t + 1 ) * kLModelDim );
            }

            return outputs;
        }

        void expectFlashMatchesRing( int window )
        {
            auto flashOut = runPrefill( true, window );
            auto cublasOut = runPrefill( false, window );

            ASSERT_EQ( flashOut.size(), cublasOut.size() );
            ASSERT_EQ( static_cast<int>( flashOut.size() ), kLContext );

            float max_diff = 0.0f;
            for ( size_t t = 0; t < flashOut.size(); ++t )
            {
                ASSERT_EQ( flashOut[ t ].size(), cublasOut[ t ].size() );

                for ( size_t i = 0; i < flashOut[ t ].size(); ++i )
                    max_diff = std::max( max_diff, std::fabs( flashOut[ t ][ i ] - cublasOut[ t ][ i ] ) );
            }

            EXPECT_LT( max_diff, kLFlashAtol )
                << "bounded flash prefill diverged from the cuBLASLt ring path at the Gemma "
                   "local config (HS=256, NKV=8, window=" << window << "), max_diff=" << max_diff;
        }

        std::unique_ptr<IExecutionContext> cuda_context_;
    };

    TEST_F( CudaGqaFlashRingPrefillParity, FlashMatchesCublasLtRing_RingWraps )
    {
        expectFlashMatchesRing( kLWindowWrap );
    }

    TEST_F( CudaGqaFlashRingPrefillParity, FlashMatchesCublasLtRing_NoWrap )
    {
        expectFlashMatchesRing( kLWindowNoWrap );
    }

    // ====================================================================
    // Fused decode-attention parity
    //
    // Diffs the fused decode kernel (cuda_gqa_decode_attention_bf16: streaming
    // online-softmax over the live band, split-K across blocks with a fixup
    // merge) against the cuBLASLt QK -> softmax_decode -> AV pipeline on the
    // SAME op, toggled per-run via setUseFlashDecode(). Three geometries pin
    // the three production shapes, and the position schedules pin the risky
    // regimes: the splits==1 direct-write path (early positions), the split-K
    // partial + fixup path (band > 64 positions), pre-window decode
    // (position < window), and ring-wrapped reads (position past the bounded
    // capacity). Both legs share one op instance, so the KV cache state is
    // identical by construction -- each position's K/V write is idempotent.
    // ====================================================================

    namespace
    {
        // Same rounding argument as the flash prefill parities: the fused kernel
        // keeps scores/probabilities in FP32 while the cuBLASLt reference rounds
        // preatt and att to BF16 between stages.
        constexpr float kDecodeParityAtol = 3e-2f;
    }

    class CudaGqaDecodeParity : public ::testing::Test
    {
    protected:
        using HostFp32 = Tensor<TensorDataType::FP32, CpuMemoryResource>;
        using DeviceBf16 = Tensor<TensorDataType::BF16, CudaDeviceMemoryResource>;
        using UnboundedBf16Op = Compute::Cuda::Gqa::CudaGqaOp<TensorDataType::BF16, false>;
        using BoundedBf16Op = Compute::Cuda::Gqa::CudaGqaOp<TensorDataType::BF16, true>;

        void SetUp() override
        {
            try
            {
                cuda_context_ = createExecutionContext( Device::Cuda( 0 ) );
            }
            catch ( const std::exception& )
            {
                cuda_context_ = nullptr;
            }

            if ( !cuda_context_ )
                GTEST_SKIP() << "CUDA device not available";
        }

        HostFp32 randomHost( const shape_t& shape, std::mt19937& rng )
        {
            std::uniform_real_distribution<float> dist( -1.0f, 1.0f );
            HostFp32 host( Device::Cpu(), shape );

            for ( size_t i = 0; i < host.size(); ++i )
                host.data()[ i ] = dist( rng );

            return host;
        }

        DeviceBf16 toDevice( const HostFp32& host )
        {
            DeviceBf16 device( Device::Cuda( 0 ), host.shape() );
            copy( host, device, cuda_context_.get() );
            cuda_context_->synchronize();

            return device;
        }

        HostFp32 toFloat( const DeviceBf16& device )
        {
            auto host = toHost<TensorDataType::FP32>( device, cuda_context_.get() );
            cuda_context_->synchronize();

            return host;
        }

        // Decode `steps` identical random tokens from position 0 through a fresh op
        // with the fused decode toggled on/off; returns per-position output rows.
        template<typename OpT>
        std::vector<std::vector<float>> runDecode(
            bool useFused, int batch, int num_heads, int num_kv_heads, int head_dim,
            int window, int context, int prefill_chunk, int steps, unsigned seed )
        {
            const int model_dim = num_heads * head_dim;
            const int packed_qkv = ( num_heads + 2 * num_kv_heads ) * head_dim;

            std::mt19937 rng( seed );
            std::vector<HostFp32> qs, ks, vs;

            for ( int t = 0; t < steps; ++t )
            {
                qs.push_back( randomHost( shape_t{ batch, 1, num_heads * head_dim }, rng ) );
                ks.push_back( randomHost( shape_t{ batch, 1, num_kv_heads * head_dim }, rng ) );
                vs.push_back( randomHost( shape_t{ batch, 1, num_kv_heads * head_dim }, rng ) );
            }

            OpT op( cuda_context_.get(),
                GqaConfig( model_dim, num_heads, num_kv_heads ).withWindow( window ) );
            op.build( BuildContext( shape_t{ batch, context, packed_qkv },
                RuntimeMode::Inference, false ).withPrefillSize( prefill_chunk ) );
            op.initializeKvCache( batch, context );
            op.setUseFlashDecode( useFused );

            // Decode scratch: consumed by the cuBLASLt path, ignored by the fused
            // kernel (its split-K partials come from the shared context scratch).
            // preatt/att are context-wide, a superset of the bounded capacity columns.
            DeviceBf16 q_permute( Device::Cuda( 0 ), shape_t{ batch, num_heads, 1, head_dim } );
            DeviceBf16 preatt_decode( Device::Cuda( 0 ), shape_t{ batch, num_heads, 1, context } );
            DeviceBf16 att_decode( Device::Cuda( 0 ), shape_t{ batch, num_heads, 1, context } );
            DeviceBf16 v_out_decode( Device::Cuda( 0 ), shape_t{ batch, num_heads, 1, head_dim } );

            GqaState state;
            state.q_permute = &q_permute;
            state.preatt_decode = &preatt_decode;
            state.att_decode = &att_decode;
            state.v_out_decode = &v_out_decode;
            op.setState( state );

            std::vector<std::vector<float>> outputs;
            outputs.reserve( qs.size() );

            for ( int t = 0; t < steps; ++t )
            {
                DeviceBf16 q = toDevice( qs[ t ] );
                DeviceBf16 k = toDevice( ks[ t ] );
                DeviceBf16 v = toDevice( vs[ t ] );
                DeviceBf16 out( Device::Cuda( 0 ), shape_t{ batch, 1, model_dim } );

                op.decode( q, k, v, out, t );

                HostFp32 host = toFloat( out );
                outputs.emplace_back( host.data(), host.data() + host.size() );
            }

            return outputs;
        }

        static float maxDifference(
            const std::vector<std::vector<float>>& a,
            const std::vector<std::vector<float>>& b )
        {
            EXPECT_EQ( a.size(), b.size() );

            float max_diff = 0.0f;

            for ( size_t t = 0; t < a.size() && t < b.size(); ++t )
            {
                EXPECT_EQ( a[ t ].size(), b[ t ].size() );

                for ( size_t i = 0; i < a[ t ].size(); ++i )
                    max_diff = std::max( max_diff, std::fabs( a[ t ][ i ] - b[ t ][ i ] ) );
            }

            return max_diff;
        }

        std::unique_ptr<IExecutionContext> cuda_context_;
    };

    TEST_F( CudaGqaDecodeParity, FusedMatchesCublasLt_GemmaGlobalConfig )
    {
        // Gemma global layer exactly: MQA (NKV=1, GS=16), head_dim 512, full causal.
        // batch 2 exercises the batch strides in both the main kernel and the fixup.
        // 300 steps: positions 0..64 run splits==1 (direct write), later positions
        // engage the split-K partials + fixup merge (band > 64).
        auto fused = runDecode<UnboundedBf16Op>( true, 2, 16, 1, 512, 0, 512, 32, 300, 11u );
        auto reference = runDecode<UnboundedBf16Op>( false, 2, 16, 1, 512, 0, 512, 32, 300, 11u );

        const float max_diff = maxDifference( fused, reference );
        EXPECT_LT( max_diff, kDecodeParityAtol )
            << "fused decode diverged from cuBLASLt at the Gemma global config "
               "(HS=512, NKV=1, window=0), max_diff=" << max_diff;
    }

    TEST_F( CudaGqaDecodeParity, FusedMatchesCublasLtRing_GemmaLocalConfig )
    {
        // Gemma local layer at test scale: bounded ring, NKV=8 (GS=2), head_dim 256,
        // window 24, context 83, chunk 32 -> capacity 55 < 83. Decoding 0..82 covers
        // pre-window positions (band < window), post-window sliding, and ring-wrapped
        // reads once positions pass the 55-slot capacity.
        auto fused = runDecode<BoundedBf16Op>( true, 1, 16, 8, 256, 24, 83, 32, 83, 23u );
        auto reference = runDecode<BoundedBf16Op>( false, 1, 16, 8, 256, 24, 83, 32, 83, 23u );

        const float max_diff = maxDifference( fused, reference );
        EXPECT_LT( max_diff, kDecodeParityAtol )
            << "fused decode diverged from the cuBLASLt ring path at the Gemma local "
               "config (HS=256, NKV=8, window=24, ring wraps), max_diff=" << max_diff;
    }

    TEST_F( CudaGqaDecodeParity, FusedMatchesCublasLt_LlamaConfig )
    {
        // Llama shape at test scale: GS=4, head_dim 128, full causal. 200 steps so
        // the split-K path engages (band > 64) on the NKV=2 grid.
        auto fused = runDecode<UnboundedBf16Op>( true, 1, 8, 2, 128, 0, 256, 32, 200, 37u );
        auto reference = runDecode<UnboundedBf16Op>( false, 1, 8, 2, 128, 0, 256, 32, 200, 37u );

        const float max_diff = maxDifference( fused, reference );
        EXPECT_LT( max_diff, kDecodeParityAtol )
            << "fused decode diverged from cuBLASLt at the Llama config "
               "(HS=128, GS=4, window=0), max_diff=" << max_diff;
    }
}
