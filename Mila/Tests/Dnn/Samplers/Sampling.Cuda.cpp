/**
 * @file Sampling.Cuda.cpp
 * @brief Injected-r oracle for the device token sampler (CudaSamplingOp<FP32>).
 *
 * The TokenSampler facade owns its RNG, so the rigorous sampling checks live at the
 * op layer, where forward() takes the host-drawn uniform `r` as an explicit argument.
 * Rather than a fragile device-vs-double-host exact match (FP32 reduction order differs
 * from a double reference), these lock the algorithm by property: greedy = argmax
 * (exact), inverse-CDF boundaries, determinism, top-k / top-p support restriction, and
 * softcap monotonicity. Small, well-separated synthetic logits keep every expectation
 * unambiguous. See Specifications/TokenSampling.md section 7.
 *
 * The multi-block stochastic pipeline is additionally locked against the retained
 * single-block reference kernel (forwardReference) by token parity at a Gemma-scale
 * vocabulary (truncated cases, where the survivor sets are exactly comparable) and
 * by a host double-precision CDF bracket (full multinomial, where serial-vs-chunked
 * float summation makes token equality unattainable in flat CDF regions) — the
 * bounded-ring oracle methodology.
 *
 * Compiled only under MILA_ENABLE_CUDA; SetUp() skips if no device at runtime.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

import Mila;
import Compute.ExecutionContext;
import Compute.CudaSamplingOp;
import Dnn.Samplers.SamplingConfig;
import Dnn.GenerateParams;

namespace Mila::Tests::Dnn::Samplers
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Compute::Cuda::Sampling;

    class SamplingCudaTests : public ::testing::Test
    {
    protected:
        static constexpr int64_t kVocab = 8;
        // Gemma 4 vocabulary size — the shape the pipeline kernels actually run at.
        static constexpr int64_t kGemmaVocab = 262144;

        using OpType = CudaSamplingOp<TensorDataType::FP32>;
        using DeviceLogits = Tensor<TensorDataType::FP32, CudaDeviceMemoryResource>;
        using DeviceToken = Tensor<TensorDataType::INT32, CudaDeviceMemoryResource>;
        using HostToken = Tensor<TensorDataType::INT32, CpuMemoryResource>;

        void SetUp() override
        {
            try
            {
                ctx_ = createExecutionContext( Device::Cuda( 0 ) );
            }
            catch ( const std::exception& )
            {
                ctx_ = nullptr;
            }

            if ( !ctx_ )
            {
                GTEST_SKIP() << "CUDA device not available";
            }
        }

        std::unique_ptr<OpType> makeOp( float softcap = 0.0f, int64_t vocab = kVocab )
        {
            SamplingConfig config = SamplingConfig{}
                .withVocabularySize( vocab )
                .withFinalLogitSoftcap( softcap );

            return std::make_unique<OpType>( ctx_.get(), config );
        }

        DeviceLogits deviceLogits( const std::vector<float>& values )
        {
            const int64_t vocab = static_cast<int64_t>( values.size() );
            const shape_t shape{ 1, 1, vocab };

            Tensor<TensorDataType::FP32, CpuMemoryResource> host( Device::Cpu(), shape );
            for ( int64_t i = 0; i < vocab; ++i )
                host.data()[ i ] = values[ static_cast<size_t>( i ) ];

            DeviceLogits device( Device::Cuda( 0 ), shape );
            copy( host, device, ctx_.get() );
            ctx_->synchronize();

            return device;
        }

        // Runs the op and reads the sampled token back. The op samples on the default
        // stream; a context-free copy is a synchronous default-stream readback (mirrors
        // the TokenSampler facade), so the host value is valid without an extra sync.
        int32_t sample( OpType& op, const DeviceLogits& logits, const SamplingParams& params, float r )
        {
            DeviceToken token_device( Device::Cuda( 0 ), shape_t{ 1, 1 } );
            op.forward( logits, token_device, params, r );

            HostToken token_host( Device::Cpu(), shape_t{ 1, 1 } );
            copy( token_device, token_host );

            return token_host.data()[ 0 ];
        }

        // Same readback through the retained single-block reference kernel.
        int32_t sampleReference( OpType& op, const DeviceLogits& logits, const SamplingParams& params, float r )
        {
            DeviceToken token_device( Device::Cuda( 0 ), shape_t{ 1, 1 } );
            op.forwardReference( logits, token_device, params, r );

            HostToken token_host( Device::Cpu(), shape_t{ 1, 1 } );
            copy( token_device, token_host );

            return token_host.data()[ 0 ];
        }

        // Enqueued-path readback: the op samples on the context stream, copies the token
        // into its pinned slot asynchronously, and awaitToken() blocks on the recorded
        // event -- the decode-ahead half of the pipelined generation loop.
        int32_t sampleEnqueued( OpType& op, const DeviceLogits& logits, const SamplingParams& params, float r )
        {
            DeviceToken token_device( Device::Cuda( 0 ), shape_t{ 1, 1 } );
            op.enqueueForward( logits, token_device, params, r );

            return op.awaitToken();
        }

        // Deterministic pseudo-random logits, continuous-valued so truncation
        // boundaries fall between distinct values with probability ~1.
        std::vector<float> randomLogits( int64_t vocab, uint32_t seed )
        {
            std::mt19937 rng( seed );
            std::normal_distribution<float> dist( 0.0f, 4.0f );

            std::vector<float> values( static_cast<size_t>( vocab ) );
            for ( auto& v : values )
                v = dist( rng );

            return values;
        }

        std::unique_ptr<IExecutionContext> ctx_;
    };

    // Greedy (temperature <= 0) is an exact argmax.
    TEST_F( SamplingCudaTests, Greedy_PicksArgmax )
    {
        auto op = makeOp();
        auto logits = deviceLogits( { 1, 2, 3, 4, 5, 6, 7, 8 } );

        SamplingParams params;
        params.temperature = 0.0f;

        EXPECT_EQ( sample( *op, logits, params, 0.5f ), 7 );
    }

    // top_k == 1 is also greedy (argmax), independent of the max position.
    TEST_F( SamplingCudaTests, TopK1_PicksArgmax )
    {
        auto op = makeOp();
        auto logits = deviceLogits( { 8, 2, 3, 4, 5, 6, 7, 1 } );

        SamplingParams params;
        params.temperature = 1.0f;
        params.top_k = 1;

        EXPECT_EQ( sample( *op, logits, params, 0.99f ), 0 );
    }

    // Full-multinomial inverse-CDF boundaries: r -> 0 selects the lowest-index
    // positive-probability token; r -> 1 selects the last.
    TEST_F( SamplingCudaTests, FullMultinomial_BoundaryR )
    {
        auto op = makeOp();
        auto logits = deviceLogits( { 1, 2, 3, 4, 5, 6, 7, 8 } );

        SamplingParams params;
        params.temperature = 1.0f;
        params.top_k = 0;
        params.top_p = 1.0f;

        EXPECT_EQ( sample( *op, logits, params, 0.0f ), 0 );
        EXPECT_EQ( sample( *op, logits, params, 0.999999f ), 7 );
    }

    // Deterministic: identical (logits, params, r) -> identical token.
    TEST_F( SamplingCudaTests, Stochastic_Deterministic )
    {
        auto op = makeOp();
        auto logits = deviceLogits( { 1, 5, 2, 8, 3, 6, 4, 7 } );

        SamplingParams params;
        params.temperature = 0.8f;
        params.top_k = 0;
        params.top_p = 1.0f;

        const int32_t first = sample( *op, logits, params, 0.42f );
        const int32_t second = sample( *op, logits, params, 0.42f );

        EXPECT_EQ( first, second );
    }

    // Top-k restricts the support: with the top-2 logits far above the rest, only those
    // two indices carry non-negligible probability across an r sweep.
    TEST_F( SamplingCudaTests, TopK_RestrictsSupport )
    {
        auto op = makeOp();
        auto logits = deviceLogits( { 1, 2, 3, 4, 5, 6, 70, 80 } );

        SamplingParams params;
        params.temperature = 1.0f;
        params.top_k = 2;
        params.top_p = 1.0f;

        for ( int i = 0; i < 20; ++i )
        {
            const float r = static_cast<float>( i ) / 20.0f;
            const int32_t token = sample( *op, logits, params, r );

            EXPECT_TRUE( token == 6 || token == 7 )
                << "top_k=2 selected out-of-support token " << token << " at r=" << r;
        }
    }

    // Top-p (nucleus) restricts the support: with one dominant token the 0.5 nucleus is
    // just that token, so every r maps to it.
    TEST_F( SamplingCudaTests, TopP_RestrictsSupport )
    {
        auto op = makeOp();
        auto logits = deviceLogits( { 0, 0, 0, 0, 0, 0, 0, 20 } );

        SamplingParams params;
        params.temperature = 1.0f;
        params.top_k = 0;
        params.top_p = 0.5f;

        for ( int i = 0; i < 20; ++i )
        {
            const float r = static_cast<float>( i ) / 20.0f;
            EXPECT_EQ( sample( *op, logits, params, r ), 7 )
                << "top_p=0.5 should select only the dominant token";
        }
    }

    // Softcap squashes a huge outlier but is monotonic, so greedy argmax is preserved.
    TEST_F( SamplingCudaTests, Softcap_PreservesArgmax )
    {
        auto op = makeOp( 30.0f );
        auto logits = deviceLogits( { 1, 2, 3, 4, 5, 6, 7, 1000 } );

        SamplingParams params;
        params.temperature = 0.0f;

        EXPECT_EQ( sample( *op, logits, params, 0.5f ), 7 );
    }

    // Pipeline-vs-reference token parity at the Gemma vocabulary across the
    // truncated filter matrix and an r grid. Continuous random logits keep
    // truncation boundaries between distinct values, where both implementations
    // provably keep the same survivor set, and the truncated CDF has few enough
    // steps that float summation order cannot move the crossing.
    //
    // The FULL multinomial is deliberately absent: its index-order CDF is a 262k
    // float sum, where serial (reference) vs chunked (pipeline) accumulation
    // diverge by ~1e-5 relative — in the flat tail that shifts the crossing by
    // hundreds of indices while moving negligible mass (observed: 732 indices at
    // r = 0.999). That case is locked by the host-double bracket test below.
    TEST_F( SamplingCudaTests, Pipeline_MatchesReference_Truncated_AtGemmaVocab )
    {
        auto op = makeOp( 0.0f, kGemmaVocab );
        auto logits = deviceLogits( randomLogits( kGemmaVocab, 42 ) );

        struct FilterCase { int top_k; float top_p; };
        const FilterCase cases[] = { { 64, 1.0f }, { 0, 0.9f }, { 64, 0.9f } };
        const float r_grid[] = { 0.0f, 0.1f, 0.37f, 0.5f, 0.73f, 0.9f, 0.999f };

        for ( const auto& filter : cases )
        {
            for ( const float r : r_grid )
            {
                SamplingParams params;
                params.temperature = 0.8f;
                params.top_k = filter.top_k;
                params.top_p = filter.top_p;

                EXPECT_EQ(
                    sample( *op, logits, params, r ),
                    sampleReference( *op, logits, params, r ) )
                    << "top_k=" << filter.top_k << " top_p=" << filter.top_p << " r=" << r;
            }
        }
    }

    // Full multinomial at the Gemma vocabulary: both implementations must place
    // the sampled token so that its cumulative interval [cum - p, cum] contains
    // r * total, within a slack covering FP32 summation error over 262k terms.
    // Checked against an independent double-precision host CDF — stronger than
    // mutual token equality, which float summation order makes unattainable in
    // flat CDF regions (see the truncated parity test's comment).
    TEST_F( SamplingCudaTests, FullMultinomial_HostCdfBracket_AtGemmaVocab )
    {
        constexpr float kTemperature = 0.8f;
        const std::vector<float> values = randomLogits( kGemmaVocab, 42 );

        auto op = makeOp( 0.0f, kGemmaVocab );
        auto logits = deviceLogits( values );

        // Host double-precision inclusive CDF of the scaled distribution.
        double max_scaled = -1e300;
        for ( const float v : values )
            max_scaled = std::max( max_scaled, static_cast<double>( v ) / kTemperature );

        std::vector<double> cumulative( values.size() );
        double running = 0.0;
        for ( size_t i = 0; i < values.size(); ++i )
        {
            running += std::exp( static_cast<double>( values[i] ) / kTemperature - max_scaled );
            cumulative[i] = running;
        }
        const double total = running;
        const double slack = 5e-4 * total;

        SamplingParams params;
        params.temperature = kTemperature;
        params.top_k = 0;
        params.top_p = 1.0f;

        const float r_grid[] = { 0.0f, 0.1f, 0.37f, 0.5f, 0.73f, 0.9f, 0.999f };

        for ( const float r : r_grid )
        {
            const double target = static_cast<double>( r ) * total;

            const int32_t tokens[] = {
                sample( *op, logits, params, r ),
                sampleReference( *op, logits, params, r ) };
            const char* names[] = { "pipeline", "reference" };

            for ( int which = 0; which < 2; ++which )
            {
                const int32_t token = tokens[which];
                ASSERT_GE( token, 0 );
                ASSERT_LT( token, kGemmaVocab );

                const double cum_inclusive = cumulative[static_cast<size_t>( token )];
                const double cum_exclusive = ( token > 0 )
                    ? cumulative[static_cast<size_t>( token ) - 1] : 0.0;

                EXPECT_GE( cum_inclusive, target - slack )
                    << names[which] << " token " << token << " ends before the target at r=" << r;
                EXPECT_LE( cum_exclusive, target + slack )
                    << names[which] << " token " << token << " starts past the target at r=" << r;
            }
        }
    }

    // Same parity with the Gemma final-logit softcap engaged (config-time property).
    TEST_F( SamplingCudaTests, Pipeline_MatchesReference_WithSoftcap )
    {
        auto op = makeOp( 30.0f, kGemmaVocab );
        auto logits = deviceLogits( randomLogits( kGemmaVocab, 7 ) );

        const float r_grid[] = { 0.1f, 0.5f, 0.9f };

        for ( const float r : r_grid )
        {
            SamplingParams params;
            params.temperature = 0.7f;
            params.top_k = 64;
            params.top_p = 0.95f;

            EXPECT_EQ(
                sample( *op, logits, params, r ),
                sampleReference( *op, logits, params, r ) )
                << "softcap parity at r=" << r;
        }
    }

    // Inverse-CDF boundary properties survive the chunked walk at the Gemma
    // vocabulary: uniform logits make the CDF exact in FP32 (integer partial sums),
    // so r -> 0 is token 0 and r -> 1 is the last token.
    TEST_F( SamplingCudaTests, Pipeline_BoundaryR_AtGemmaVocab )
    {
        auto op = makeOp( 0.0f, kGemmaVocab );
        auto logits = deviceLogits( std::vector<float>( kGemmaVocab, 0.0f ) );

        SamplingParams params;
        params.temperature = 1.0f;
        params.top_k = 0;
        params.top_p = 1.0f;

        EXPECT_EQ( sample( *op, logits, params, 0.0f ), 0 );
        EXPECT_EQ( sample( *op, logits, params, 0.999999f ), kGemmaVocab - 1 );
    }

    // The top-k refinement is integer-count based, so the full pipeline is
    // run-deterministic under top-k at scale.
    TEST_F( SamplingCudaTests, Pipeline_Deterministic_TopK_AtGemmaVocab )
    {
        auto op = makeOp( 0.0f, kGemmaVocab );
        auto logits = deviceLogits( randomLogits( kGemmaVocab, 1234 ) );

        SamplingParams params;
        params.temperature = 0.8f;
        params.top_k = 64;
        params.top_p = 1.0f;

        const int32_t first = sample( *op, logits, params, 0.42f );
        const int32_t second = sample( *op, logits, params, 0.42f );

        EXPECT_EQ( first, second );
    }

    // ------------------------------------------------------------------
    // Enqueued path (decode-ahead pipeline): enqueueForward + awaitToken
    // ------------------------------------------------------------------

    // The enqueued path is the same kernel dispatch on a different stream with an
    // async readback; greedy argmax must match the synchronous path exactly.
    TEST_F( SamplingCudaTests, Enqueued_MatchesForward_Greedy )
    {
        auto op = makeOp();
        auto logits = deviceLogits( { 3, 9, 1, 7, 5, 2, 8, 4 } );

        SamplingParams params;
        params.temperature = 0.0f;

        EXPECT_EQ( sampleEnqueued( *op, logits, params, 0.5f ), 1 );
        EXPECT_EQ( sampleEnqueued( *op, logits, params, 0.5f ), sample( *op, logits, params, 0.5f ) );
    }

    // Token parity between the synchronous and enqueued paths across the truncated
    // filter matrix at the Gemma vocabulary -- same kernels, same injected r, so
    // equality is exact.
    TEST_F( SamplingCudaTests, Enqueued_MatchesForward_Truncated_AtGemmaVocab )
    {
        auto op = makeOp( 0.0f, kGemmaVocab );
        auto logits = deviceLogits( randomLogits( kGemmaVocab, 42 ) );

        struct FilterCase { int top_k; float top_p; };
        const FilterCase cases[] = { { 64, 1.0f }, { 0, 0.9f }, { 64, 0.9f } };
        const float r_grid[] = { 0.0f, 0.37f, 0.73f, 0.999f };

        for ( const auto& filter : cases )
        {
            for ( const float r : r_grid )
            {
                SamplingParams params;
                params.temperature = 0.8f;
                params.top_k = filter.top_k;
                params.top_p = filter.top_p;

                EXPECT_EQ(
                    sampleEnqueued( *op, logits, params, r ),
                    sample( *op, logits, params, r ) )
                    << "top_k=" << filter.top_k << " top_p=" << filter.top_p << " r=" << r;
            }
        }
    }

    // Back-to-back enqueue/await cycles reuse the single pinned slot + event; each
    // cycle must return its own token (the generation loop's steady-state pattern).
    TEST_F( SamplingCudaTests, Enqueued_BackToBack_SingleSlotReuse )
    {
        auto op = makeOp();
        auto logits_low = deviceLogits( { 9, 1, 1, 1, 1, 1, 1, 1 } );
        auto logits_high = deviceLogits( { 1, 1, 1, 1, 1, 1, 1, 9 } );

        SamplingParams params;
        params.temperature = 0.0f;

        EXPECT_EQ( sampleEnqueued( *op, logits_low, params, 0.5f ), 0 );
        EXPECT_EQ( sampleEnqueued( *op, logits_high, params, 0.5f ), 7 );
        EXPECT_EQ( sampleEnqueued( *op, logits_low, params, 0.5f ), 0 );
    }

    // Stream-ordering contract: logits written on the context stream with NO host
    // synchronize before enqueueForward -- the sampler kernels are ordered after the
    // producing copy on the same stream, mirroring the pipelined loop where no
    // network synchronize runs between forward and sample.
    TEST_F( SamplingCudaTests, Enqueued_OrderedAfterPriorStreamWork )
    {
        auto op = makeOp();

        const shape_t shape{ 1, 1, kVocab };
        Tensor<TensorDataType::FP32, CpuMemoryResource> host( Device::Cpu(), shape );
        for ( int64_t i = 0; i < kVocab; ++i )
            host.data()[ i ] = ( i == 5 ) ? 50.0f : 1.0f;

        DeviceLogits device( Device::Cuda( 0 ), shape );
        copy( host, device, ctx_.get() );

        SamplingParams params;
        params.temperature = 0.0f;

        DeviceToken token_device( Device::Cuda( 0 ), shape_t{ 1, 1 } );
        op->enqueueForward( device, token_device, params, 0.5f );

        EXPECT_EQ( op->awaitToken(), 5 );
    }

    // awaitToken() with no outstanding enqueueForward() is a caller bug, not UB.
    TEST_F( SamplingCudaTests, AwaitToken_WithoutEnqueue_Throws )
    {
        auto op = makeOp();

        EXPECT_THROW( (void)op->awaitToken(), std::logic_error );
    }
}
