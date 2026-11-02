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
 * The presence penalty and suppressed tokens are locked against the unadjusted draw over logits
 * adjusted on the host, and top-k ties by tokens planted on the cut.
 *
 * Compiled only under MILA_ENABLE_CUDA; SetUp() skips if no device at runtime.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>

import Mila;
import Compute.CudaSamplingOp;
import Compute.CudaExecutionContext;
import Dnn.Samplers.SamplingConfig;
import Dnn.Samplers.SpeculativeSampler;
import Dnn.Samplers.TokenSampler;
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
        using DeviceFlags = Tensor<TensorDataType::UINT8, CudaDeviceMemoryResource>;
        using HostFlags = Tensor<TensorDataType::UINT8, CpuMemoryResource>;

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

        std::unique_ptr<OpType> makeOp( float softcap = 0.0f, int64_t vocab = kVocab, std::vector<int32_t> suppressed = {} )
        {
            SamplingConfig config = SamplingConfig{}
                .withVocabularySize( vocab )
                .withFinalLogitSoftcap( softcap )
                .withSuppressedTokens( std::move( suppressed ) );

            return std::make_unique<OpType>( ctx_.get(), config );
        }

        // One UINT8 flag per token, set for each of `flagged`: the reply's produced tokens as the sampler reads them.
        DeviceFlags deviceFlags( int64_t vocab, const std::vector<int32_t>& flagged )
        {
            HostFlags host( Device::Cpu(), shape_t{ vocab } );
            std::fill( host.data(), host.data() + vocab, static_cast<uint8_t>( 0 ) );

            for ( const int32_t token : flagged )
                host.data()[ token ] = 1;

            DeviceFlags device( Device::Cuda( 0 ), shape_t{ vocab } );
            copy( host, device, ctx_.get() );
            ctx_->synchronize();

            return device;
        }

        std::vector<int32_t> flaggedTokens( const DeviceFlags& flags )
        {
            HostFlags host( Device::Cpu(), flags.shape() );
            copy( flags, host, ctx_.get() );
            ctx_->synchronize();

            std::vector<int32_t> flagged;

            for ( int64_t i = 0; i < flags.size(); ++i )
            {
                if ( host.data()[ i ] != 0 )
                    flagged.push_back( static_cast<int32_t>( i ) );
            }

            return flagged;
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

        // Runs the op and reads the sampled token back. The op samples on the context's stream, so the
        // readback copies on it and synchronizes it, as the TokenSampler facade does.
        int32_t sample( OpType& op, const DeviceLogits& logits, const SamplingParams& params, float r,
            ITensor* produced = nullptr )
        {
            DeviceToken token_device( Device::Cuda( 0 ), shape_t{ 1, 1 } );
            op.forward( logits, token_device, params, r, produced );

            return readToken( token_device );
        }

        // Same readback through the retained single-block reference kernel.
        int32_t sampleReference( OpType& op, const DeviceLogits& logits, const SamplingParams& params, float r,
            ITensor* produced = nullptr )
        {
            DeviceToken token_device( Device::Cuda( 0 ), shape_t{ 1, 1 } );
            op.forwardReference( logits, token_device, params, r, produced );

            return readToken( token_device );
        }

        int32_t readToken( const DeviceToken& token_device )
        {
            HostToken token_host( Device::Cpu(), shape_t{ 1, 1 } );
            copy( token_device, token_host, ctx_.get() );
            ctx_->synchronize();

            return token_host.data()[ 0 ];
        }

        // Enqueued-path readback: the op samples on the context stream, copies the token
        // into its pinned slot asynchronously, and awaitToken() blocks on the recorded
        // event -- the decode-ahead half of the pipelined generation loop.
        int32_t sampleEnqueued( OpType& op, const DeviceLogits& logits, const SamplingParams& params, float r,
            ITensor* produced = nullptr )
        {
            DeviceToken token_device( Device::Cuda( 0 ), shape_t{ 1, 1 } );
            op.enqueueForward( logits, token_device, params, r, produced );

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
    // Top-k ties: every token tied with the k-th survives (HuggingFace's TopKLogitsWarper)
    // ------------------------------------------------------------------

    // Three tokens tie at the 2nd-largest value, below a distinct largest one, so top_k = 2 keeps all four. The
    // reference's bisection used to drop the tie whole and keep the largest alone.
    TEST_F( SamplingCudaTests, TopK_KeepsEveryTokenTiedWithTheKth )
    {
        auto op = makeOp();
        auto logits = deviceLogits( { 1, 2, 3, 5, 5, 5, 5.5f, 0 } );

        SamplingParams params;
        params.temperature = 1.0f;
        params.top_k = 2;
        params.top_p = 1.0f;

        std::set<int32_t> pipeline_drawn;
        std::set<int32_t> reference_drawn;

        for ( int i = 0; i < 30; ++i )
        {
            const float r = static_cast<float>( i ) / 30.0f;
            pipeline_drawn.insert( sample( *op, logits, params, r ) );
            reference_drawn.insert( sampleReference( *op, logits, params, r ) );
        }

        EXPECT_EQ( pipeline_drawn, ( std::set<int32_t>{ 3, 4, 5, 6 } ) );
        EXPECT_EQ( reference_drawn, ( std::set<int32_t>{ 3, 4, 5, 6 } ) );
    }

    // At the Gemma vocabulary: 30 tokens at 11 and 100 tied at 10 under top_k = 64, so the 64th is one of the tie and
    // all 130 survive. A fine r sweep draws every one of them and nothing else.
    TEST_F( SamplingCudaTests, TopK_KeepsTiesAcrossTheCut_AtGemmaVocab )
    {
        auto op = makeOp( 0.0f, kGemmaVocab );

        std::vector<float> values( static_cast<size_t>( kGemmaVocab ), 0.0f );
        std::set<int32_t> above;
        std::set<int32_t> tied;

        for ( int32_t i = 0; i < 130; ++i )
        {
            const int32_t token = 17 + i * 1999;
            values[ static_cast<size_t>( token ) ] = ( i < 30 ) ? 11.0f : 10.0f;
            ( i < 30 ? above : tied ).insert( token );
        }

        std::set<int32_t> survivors = above;
        survivors.insert( tied.begin(), tied.end() );

        auto logits = deviceLogits( values );

        SamplingParams params;
        params.temperature = 1.0f;
        params.top_k = 64;
        params.top_p = 1.0f;

        std::set<int32_t> pipeline_drawn;
        std::set<int32_t> reference_drawn;

        for ( int i = 0; i < 2000; ++i )
        {
            const float r = ( static_cast<float>( i ) + 0.5f ) / 2000.0f;
            pipeline_drawn.insert( sample( *op, logits, params, r ) );

            if ( i % 4 == 0 )
                reference_drawn.insert( sampleReference( *op, logits, params, r ) );
        }

        EXPECT_EQ( pipeline_drawn, survivors );

        int tied_drawn = 0;

        for ( const int32_t token : reference_drawn )
        {
            EXPECT_TRUE( survivors.contains( token ) ) << "reference drew token " << token << " outside the top 64";
            tied_drawn += tied.contains( token ) ? 1 : 0;
        }

        EXPECT_GE( tied_drawn, 80 ) << "the reference keeps the tie";
    }

    // ------------------------------------------------------------------
    // Presence penalty (TokenSampling.md 3.3)
    // ------------------------------------------------------------------

    // Greedy takes the penalty: the produced maximum falls below the runner-up.
    TEST_F( SamplingCudaTests, PresencePenalty_Greedy_PassesOverAProducedMaximum )
    {
        auto op = makeOp();
        auto logits = deviceLogits( { 1, 2, 3, 9, 8.5f, 4, 5, 6 } );

        SamplingParams params;
        params.temperature = 0.0f;
        params.presence_penalty = 1.5f;

        // Each penalized draw marks its token, so each gets its own produced tokens.
        auto produced = deviceFlags( kVocab, { 3 } );
        EXPECT_EQ( sample( *op, logits, params, 0.5f, &produced ), 4 );

        auto produced_for_reference = deviceFlags( kVocab, { 3 } );
        EXPECT_EQ( sampleReference( *op, logits, params, 0.5f, &produced_for_reference ), 4 );

        params.presence_penalty = 0.0f;
        auto produced_unread = deviceFlags( kVocab, { 3 } );

        EXPECT_EQ( sample( *op, logits, params, 0.5f, &produced_unread ), 3 ) << "a zero penalty reads no flags";
    }

    // The penalty follows the softcap: 30 tanh(9 / 30) - 1.5 = 7.24 falls below 30 tanh(8.5 / 30) = 8.27.
    TEST_F( SamplingCudaTests, PresencePenalty_Greedy_FollowsTheSoftcap )
    {
        auto op = makeOp( 30.0f );
        auto logits = deviceLogits( { 1, 2, 3, 9, 8.5f, 4, 5, 6 } );
        auto produced = deviceFlags( kVocab, { 3 } );

        SamplingParams params;
        params.temperature = 0.0f;
        params.presence_penalty = 1.5f;

        EXPECT_EQ( sample( *op, logits, params, 0.5f, &produced ), 4 );
    }

    // The penalized draw equals the unpenalized draw over logits with the penalty subtracted on the host -- the
    // same float subtraction the kernel makes before temperature -- across the truncated filter matrix and the full
    // multinomial, in the pipeline and the reference.
    TEST_F( SamplingCudaTests, PresencePenalty_EqualsTheDrawOverPenalizedLogits_AtGemmaVocab )
    {
        constexpr float kPenalty = 1.5f;

        auto op = makeOp( 0.0f, kGemmaVocab );
        const std::vector<float> values = randomLogits( kGemmaVocab, 42 );

        // The largest 40 logits and 400 others are produced, so the penalty moves the top of the distribution.
        std::vector<int32_t> order( static_cast<size_t>( kGemmaVocab ) );
        std::iota( order.begin(), order.end(), 0 );
        std::partial_sort( order.begin(), order.begin() + 40, order.end(),
            [&]( int32_t a, int32_t b ) { return values[ static_cast<size_t>( a ) ] > values[ static_cast<size_t>( b ) ]; } );

        std::vector<int32_t> flagged( order.begin(), order.begin() + 40 );

        for ( int32_t i = 0; i < 400; ++i )
            flagged.push_back( order[ static_cast<size_t>( 1000 + i * 600 ) ] );

        std::vector<float> penalized_values = values;

        for ( const int32_t token : flagged )
            penalized_values[ static_cast<size_t>( token ) ] -= kPenalty;

        auto logits = deviceLogits( values );
        auto penalized_logits = deviceLogits( penalized_values );

        struct FilterCase { int top_k; float top_p; };
        const FilterCase cases[] = { { 20, 0.8f }, { 64, 1.0f }, { 0, 0.95f }, { 0, 1.0f } };
        const float r_grid[] = { 0.0f, 0.1f, 0.37f, 0.5f, 0.73f, 0.9f, 0.999f };

        for ( const auto& filter : cases )
        {
            for ( const float r : r_grid )
            {
                SamplingParams params;
                params.temperature = 0.7f;
                params.top_k = filter.top_k;
                params.top_p = filter.top_p;

                const int32_t expected = sample( *op, penalized_logits, params, r );
                const int32_t expected_reference = sampleReference( *op, penalized_logits, params, r );

                params.presence_penalty = kPenalty;

                // Each penalized draw marks its token, so each starts from the same produced tokens.
                auto produced = deviceFlags( kGemmaVocab, flagged );
                EXPECT_EQ( sample( *op, logits, params, r, &produced ), expected )
                    << "top_k=" << filter.top_k << " top_p=" << filter.top_p << " r=" << r;

                auto produced_for_reference = deviceFlags( kGemmaVocab, flagged );
                EXPECT_EQ( sampleReference( *op, logits, params, r, &produced_for_reference ), expected_reference )
                    << "reference top_k=" << filter.top_k << " top_p=" << filter.top_p << " r=" << r;
            }
        }
    }

    // Each draw marks its token on the device before the next is enqueued, with no host read between: the
    // decode-ahead loop's case. Three greedy draws over the same logits walk down the three largest.
    TEST_F( SamplingCudaTests, PresencePenalty_MarksEachDrawOnTheDevice )
    {
        auto op = makeOp();
        auto logits = deviceLogits( { 1, 2, 3, 9, 8.5f, 8, 0, 0 } );
        DeviceFlags produced( Device::Cuda( 0 ), shape_t{ kVocab } );
        op->enqueueClearProduced( produced );

        SamplingParams params;
        params.temperature = 0.0f;
        params.presence_penalty = 1.5f;

        DeviceToken tokens( Device::Cuda( 0 ), shape_t{ 3 } );

        for ( int64_t step = 0; step < 3; ++step )
        {
            auto slot = tokens.view( shape_t{ 1, 1 }, step );
            op->enqueueForwardOnDevice( logits, slot, params, 0.5f, &produced );
        }

        HostToken host( Device::Cpu(), shape_t{ 3 } );
        copy( tokens, host, ctx_.get() );
        ctx_->synchronize();

        EXPECT_EQ( std::vector<int32_t>( host.data(), host.data() + 3 ), ( std::vector<int32_t>{ 3, 4, 5 } ) );
        EXPECT_EQ( flaggedTokens( produced ), ( std::vector<int32_t>{ 3, 4, 5 } ) );
    }

    // With the penalty off nothing is marked, and the enqueued path marks as the others do.
    TEST_F( SamplingCudaTests, PresencePenalty_MarksOnlyWhenOn )
    {
        auto op = makeOp();
        auto logits = deviceLogits( { 1, 2, 3, 9, 8.5f, 8, 0, 0 } );
        auto produced = deviceFlags( kVocab, {} );

        SamplingParams params;
        params.temperature = 0.0f;

        EXPECT_EQ( sampleEnqueued( *op, logits, params, 0.5f, &produced ), 3 );
        EXPECT_TRUE( flaggedTokens( produced ).empty() );

        params.presence_penalty = 1.5f;

        EXPECT_EQ( sampleEnqueued( *op, logits, params, 0.5f, &produced ), 3 );
        EXPECT_EQ( flaggedTokens( produced ), ( std::vector<int32_t>{ 3 } ) );
    }

    // A penalty with no produced tokens to read, or a non-finite one, is refused before any launch.
    TEST_F( SamplingCudaTests, PresencePenalty_RefusesWhatItCannotApply )
    {
        auto op = makeOp();
        auto logits = deviceLogits( { 1, 2, 3, 4, 5, 6, 7, 8 } );
        auto too_small = deviceFlags( 4, {} );
        DeviceToken token( Device::Cuda( 0 ), shape_t{ 1, 1 } );

        SamplingParams params;
        params.presence_penalty = 1.5f;

        EXPECT_THROW( op->forward( logits, token, params, 0.5f ), std::invalid_argument );
        EXPECT_THROW( op->forward( logits, token, params, 0.5f, &too_small ), std::invalid_argument );

        auto produced = deviceFlags( kVocab, {} );
        params.presence_penalty = std::numeric_limits<float>::infinity();

        EXPECT_THROW( op->forward( logits, token, params, 0.5f, &produced ), std::invalid_argument );
    }

    // ------------------------------------------------------------------
    // Suppressed tokens: never chosen
    // ------------------------------------------------------------------

    TEST_F( SamplingCudaTests, Suppressed_GreedyPassesOverASuppressedMaximum )
    {
        auto op = makeOp( 30.0f, kVocab, { 7, 2 } );
        auto logits = deviceLogits( { 1, 2, 3, 4, 5, 6, 7, 8 } );

        SamplingParams params;
        params.temperature = 0.0f;

        EXPECT_EQ( sample( *op, logits, params, 0.5f ), 6 );
        EXPECT_EQ( sampleReference( *op, logits, params, 0.5f ), 6 );
    }

    // A suppressed token draws as one whose probability is exactly zero: the same token as the unsuppressed op over
    // logits that put the suppressed ones far below everything, in every filter case and in the reference.
    TEST_F( SamplingCudaTests, Suppressed_EqualsTheDrawWithThoseTokensRemoved_AtGemmaVocab )
    {
        const std::vector<float> values = randomLogits( kGemmaVocab, 99 );

        // The ten largest logits and some ordinary ones, so suppression moves the top of the distribution.
        std::vector<int32_t> order( static_cast<size_t>( kGemmaVocab ) );
        std::iota( order.begin(), order.end(), 0 );
        std::partial_sort( order.begin(), order.begin() + 10, order.end(),
            [&]( int32_t a, int32_t b ) { return values[ static_cast<size_t>( a ) ] > values[ static_cast<size_t>( b ) ]; } );

        std::vector<int32_t> suppressed( order.begin(), order.begin() + 10 );
        suppressed.push_back( 255999 );
        suppressed.push_back( 258884 );

        std::vector<float> removed_values = values;

        for ( const int32_t token : suppressed )
            removed_values[ static_cast<size_t>( token ) ] = -1.0e4f;

        auto suppressing_op = makeOp( 0.0f, kGemmaVocab, suppressed );
        auto plain_op = makeOp( 0.0f, kGemmaVocab );
        auto logits = deviceLogits( values );
        auto removed_logits = deviceLogits( removed_values );

        struct FilterCase { int top_k; float top_p; };
        const FilterCase cases[] = { { 64, 0.95f }, { 20, 1.0f }, { 0, 0.9f }, { 0, 1.0f } };
        const float r_grid[] = { 0.0f, 0.1f, 0.37f, 0.5f, 0.73f, 0.9f, 0.999f };

        for ( const auto& filter : cases )
        {
            for ( const float r : r_grid )
            {
                SamplingParams params;
                params.temperature = 1.0f;
                params.top_k = filter.top_k;
                params.top_p = filter.top_p;

                const int32_t token = sample( *suppressing_op, logits, params, r );

                EXPECT_FALSE( std::ranges::contains( suppressed, token ) ) << "drew suppressed token " << token;
                EXPECT_EQ( token, sample( *plain_op, removed_logits, params, r ) )
                    << "top_k=" << filter.top_k << " top_p=" << filter.top_p << " r=" << r;
                EXPECT_EQ( sampleReference( *suppressing_op, logits, params, r ),
                    sampleReference( *plain_op, removed_logits, params, r ) )
                    << "reference top_k=" << filter.top_k << " top_p=" << filter.top_p << " r=" << r;
            }
        }
    }

    // A suppressed token outside the vocabulary is a configuration error.
    TEST_F( SamplingCudaTests, Suppressed_RefusesATokenOutsideTheVocabulary )
    {
        EXPECT_THROW( makeOp( 0.0f, kVocab, { static_cast<int32_t>( kVocab ) } ), std::invalid_argument );
        EXPECT_THROW( makeOp( 0.0f, kVocab, { -1 } ), std::invalid_argument );
    }

    // ------------------------------------------------------------------
    // Enqueued path (decode-ahead pipeline): enqueueForward + awaitToken
    // ------------------------------------------------------------------

    // The enqueued path is the same kernel dispatch with an async readback; greedy
    // argmax must match the synchronous path exactly.
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

    // ------------------------------------------------------------------
    // Rows: every row of a verify sampled in one call (Gemma4Mtp.md 4.8)
    // ------------------------------------------------------------------

    class SamplingRowsCudaTests : public SamplingCudaTests
    {
    protected:
        static constexpr int64_t kRows = 5;

        // The most rows one call samples (kMaxSampleRows in Sampling.cuh): a verify of K = 7.
        static constexpr int64_t kMaxSampleRows = 8;

        std::unique_ptr<OpType> makeRowsOp( int64_t rows, float softcap = 0.0f, int64_t vocab = kGemmaVocab )
        {
            SamplingConfig config = SamplingConfig{}
                .withVocabularySize( vocab )
                .withFinalLogitSoftcap( softcap )
                .withMaximumRows( rows );

            return std::make_unique<OpType>( ctx_.get(), config );
        }

        // `rows` rows of random logits, each its own seed, contiguous as a verify's [1, rows, vocab].
        DeviceLogits deviceRows( int64_t rows, int64_t vocab, uint32_t seed )
        {
            const shape_t shape{ 1, rows, vocab };
            Tensor<TensorDataType::FP32, CpuMemoryResource> host( Device::Cpu(), shape );

            for ( int64_t row = 0; row < rows; ++row )
            {
                const std::vector<float> values = randomLogits( vocab, seed + static_cast<uint32_t>( row ) );

                for ( int64_t i = 0; i < vocab; ++i )
                    host.data()[ row * vocab + i ] = values[ static_cast<size_t>( i ) ];
            }

            DeviceLogits device( Device::Cuda( 0 ), shape );
            copy( host, device, ctx_.get() );
            ctx_->synchronize();

            return device;
        }

        std::vector<int32_t> readTokens( const DeviceToken& tokens, int64_t count )
        {
            HostToken host( Device::Cpu(), shape_t{ count } );
            auto first = tokens.view( shape_t{ count }, 0 );
            copy( first, host, ctx_.get() );
            ctx_->synchronize();

            return std::vector<int32_t>( host.data(), host.data() + count );
        }

        // Each row sampled alone, as the per-row loop sampled it: the one-row call on that row's view.
        std::vector<int32_t> sampleEachRow(
            OpType& op, const DeviceLogits& logits, int64_t rows, const SamplingParams& params,
            const std::vector<float>& uniforms )
        {
            std::vector<int32_t> tokens;

            for ( int64_t row = 0; row < rows; ++row )
            {
                auto view = logits.view( shape_t{ 1, 1, kGemmaVocab }, row * kGemmaVocab );
                DeviceToken token( Device::Cuda( 0 ), shape_t{ 1, 1 } );
                op.enqueueForwardOnDevice( view, token, params, uniforms[ static_cast<size_t>( row ) ] );
                tokens.push_back( readTokens( token, 1 )[ 0 ] );
            }

            return tokens;
        }

        static std::vector<float> uniformsFrom( uint32_t seed, int64_t count )
        {
            std::mt19937 rng( seed );
            std::uniform_real_distribution<float> dist( 0.0f, 1.0f );
            std::vector<float> uniforms( static_cast<size_t>( count ) );

            for ( auto& uniform : uniforms )
                uniform = dist( rng );

            return uniforms;
        }
    };

    // The rows call is the one-row pipeline per row: same token at every row, at Google's sampling and the truncated
    // filter matrix, with the Gemma softcap. Top-k counts are integer-exact; the top-p cases rely on the same
    // run-to-run stability the pipeline-vs-reference parity test above does.
    TEST_F( SamplingRowsCudaTests, Rows_MatchOneRowCalls_AtGemmaVocab )
    {
        auto rows_op = makeRowsOp( kRows, 30.0f );
        auto one_row_op = makeOp( 30.0f, kGemmaVocab );
        auto logits = deviceRows( kRows, kGemmaVocab, 100 );

        struct SettingsCase { float temperature; int top_k; float top_p; };
        const SettingsCase cases[] = { { 1.0f, 64, 0.95f }, { 0.8f, 40, 1.0f }, { 0.8f, 0, 0.9f }, { 1.0f, 0, 1.0f } };

        for ( uint32_t trial = 0; trial < 4; ++trial )
        {
            const std::vector<float> uniforms = uniformsFrom( 7 + trial, kRows );

            for ( const auto& settings : cases )
            {
                SamplingParams params;
                params.temperature = settings.temperature;
                params.top_k = settings.top_k;
                params.top_p = settings.top_p;

                DeviceToken tokens( Device::Cuda( 0 ), shape_t{ kRows } );
                rows_op->enqueueRowsOnDevice( logits, tokens, kRows, params, uniforms );

                EXPECT_EQ( readTokens( tokens, kRows ), sampleEachRow( *one_row_op, logits, kRows, params, uniforms ) )
                    << "temperature=" << settings.temperature << " top_k=" << settings.top_k
                    << " top_p=" << settings.top_p << " trial=" << trial;
            }
        }
    }

    // Greedy rows are each row's argmax, one block a row.
    TEST_F( SamplingRowsCudaTests, Rows_Greedy_IsEachRowsArgmax )
    {
        auto op = makeRowsOp( 3, 0.0f, kVocab );

        const shape_t shape{ 1, 3, kVocab };
        Tensor<TensorDataType::FP32, CpuMemoryResource> host( Device::Cpu(), shape );
        const float values[ 3 ][ kVocab ] = {
            { 1, 9, 2, 3, 4, 5, 6, 7 }, { 8, 1, 2, 3, 4, 5, 6, 7 }, { 1, 2, 3, 4, 5, 6, 7, 9 } };

        for ( int64_t row = 0; row < 3; ++row )
            for ( int64_t i = 0; i < kVocab; ++i )
                host.data()[ row * kVocab + i ] = values[ row ][ i ];

        DeviceLogits logits( Device::Cuda( 0 ), shape );
        copy( host, logits, ctx_.get() );

        SamplingParams params;
        params.temperature = 0.0f;

        DeviceToken tokens( Device::Cuda( 0 ), shape_t{ 3 } );
        const std::vector<float> uniforms( 3, 0.5f );
        op->enqueueRowsOnDevice( logits, tokens, 3, params, uniforms );

        EXPECT_EQ( readTokens( tokens, 3 ), ( std::vector<int32_t>{ 1, 0, 7 } ) );
    }

    // A call of more rows than the sampler was built for, or with too few uniforms, is refused before any launch.
    TEST_F( SamplingRowsCudaTests, Rows_RefusesMoreRowsThanBuilt )
    {
        auto op = makeRowsOp( 2, 0.0f, kVocab );
        auto logits = deviceRows( 3, kVocab, 1 );

        SamplingParams params;
        DeviceToken tokens( Device::Cuda( 0 ), shape_t{ 3 } );

        EXPECT_THROW( op->enqueueRowsOnDevice( logits, tokens, 3, params, std::vector<float>( 3, 0.5f ) ),
            std::invalid_argument );
        EXPECT_THROW( op->enqueueRowsOnDevice( logits, tokens, 2, params, std::vector<float>( 1, 0.5f ) ),
            std::invalid_argument );
    }

    // The walk keeps a draft while the row before it chose it: none, some and all kept, the result laid out as
    // [m, d_1 .. d_m, next] and the next token written into slot 0.
    TEST_F( SamplingRowsCudaTests, Accept_KeepsDraftsWhileTheRowBeforeChoseThem )
    {
        auto op = makeRowsOp( kRows, 0.0f, kVocab );

        struct AcceptCase
        {
            std::vector<int32_t> tokens;
            std::vector<int32_t> chosen;
            std::vector<int32_t> expected;
        };

        // tokens = [known, d_1 .. d_4]; chosen[r] is row r's draw.
        const AcceptCase cases[] = {
            { { 50, 11, 12, 13, 14 }, { 99, 12, 13, 14, 15 }, { 0, 99 } },
            { { 50, 11, 12, 13, 14 }, { 11, 12, 77, 14, 15 }, { 2, 11, 12, 77 } },
            { { 50, 11, 12, 13, 14 }, { 11, 12, 13, 14, 15 }, { 4, 11, 12, 13, 14, 15 } },
            { { 50, 11, 12, 13, 14 }, { 11, 99, 13, 14, 15 }, { 1, 11, 99 } } };

        for ( const auto& accept : cases )
        {
            HostToken host_tokens( Device::Cpu(), shape_t{ kRows } );
            HostToken host_chosen( Device::Cpu(), shape_t{ kRows } );

            for ( int64_t i = 0; i < kRows; ++i )
            {
                host_tokens.data()[ i ] = accept.tokens[ static_cast<size_t>( i ) ];
                host_chosen.data()[ i ] = accept.chosen[ static_cast<size_t>( i ) ];
            }

            DeviceToken tokens( Device::Cuda( 0 ), shape_t{ kRows } );
            DeviceToken chosen( Device::Cuda( 0 ), shape_t{ kRows } );
            DeviceToken result( Device::Cuda( 0 ), shape_t{ kRows + 1 } );
            copy( host_tokens, tokens, ctx_.get() );
            copy( host_chosen, chosen, ctx_.get() );

            op->enqueueAcceptOnDevice( tokens, chosen, kRows, result );

            const std::vector<int32_t> written = readTokens( result, static_cast<int64_t>( accept.expected.size() ) );

            EXPECT_EQ( written, accept.expected );
            EXPECT_EQ( readTokens( tokens, 1 )[ 0 ], accept.expected.back() ) << "slot 0 holds the next token";
        }
    }

    // The whole speculative sampler against the host walk over one-row draws: with the first j drafts set to the
    // tokens the rows draw and draft j + 1 set to something else, it keeps exactly j and returns row j's draw.
    TEST_F( SamplingRowsCudaTests, SpeculativeSampler_MatchesTheHostWalkOverOneRowDraws )
    {
        using Sampler = SpeculativeSampler<DeviceType::Cuda, TensorDataType::FP32>;

        SamplingConfig config = SamplingConfig{}
            .withVocabularySize( kGemmaVocab )
            .withFinalLogitSoftcap( 30.0f )
            .withMaximumRows( kRows );

        Sampler sampler( ctx_.get(), config );
        auto one_row_op = makeOp( 30.0f, kGemmaVocab );
        auto logits = deviceRows( kRows, kGemmaVocab, 300 );

        SamplingParams params;
        params.temperature = 1.0f;
        params.top_k = 64;
        params.top_p = 0.95f;

        const std::vector<float> uniforms = uniformsFrom( 11, kRows );
        const std::vector<int32_t> draws = sampleEachRow( *one_row_op, logits, kRows, params, uniforms );

        for ( int64_t kept_drafts = 0; kept_drafts < kRows; ++kept_drafts )
        {
            HostToken host_tokens( Device::Cpu(), shape_t{ kRows } );
            host_tokens.data()[ 0 ] = 1234;

            for ( int64_t i = 1; i < kRows; ++i )
            {
                const int32_t draw = draws[ static_cast<size_t>( i - 1 ) ];
                host_tokens.data()[ i ] = ( i <= kept_drafts ) ? draw : ( draw + 1 ) % static_cast<int32_t>( kGemmaVocab );
            }

            DeviceToken tokens( Device::Cuda( 0 ), shape_t{ kRows } );
            copy( host_tokens, tokens, ctx_.get() );

            sampler.enqueueRound( logits, tokens, kRows, params, uniforms );
            const auto kept = sampler.awaitRound();

            std::vector<int32_t> expected( draws.begin(), draws.begin() + kept_drafts + 1 );

            EXPECT_EQ( std::vector<int32_t>( kept.begin(), kept.end() ), expected ) << "kept_drafts=" << kept_drafts;
            EXPECT_EQ( readTokens( tokens, 1 )[ 0 ], expected.back() );
        }
    }

    // With a presence penalty, row r of a verify also penalizes the round's drafts 1..r: each row's draw equals the
    // one-row draw over that row with those drafts added to the produced tokens, greedy and sampled, softcap on.
    TEST_F( SamplingRowsCudaTests, Rows_PresencePenaltyCountsTheDraftsBeforeEachRow )
    {
        constexpr float kPenalty = 2.0f;

        auto rows_op = makeRowsOp( kRows, 30.0f );
        auto one_row_op = makeOp( 30.0f, kGemmaVocab );
        auto logits = deviceRows( kRows, kGemmaVocab, 700 );

        // Draft r is row r's own unpenalized top, so counting it moves that row's draw.
        SamplingParams greedy;
        greedy.temperature = 0.0f;

        const std::vector<int32_t> base_flagged = { 5, 77, 1234 };
        std::vector<int32_t> round( static_cast<size_t>( kRows ) );
        round[ 0 ] = 1234;

        for ( int64_t row = 1; row < kRows; ++row )
        {
            auto view = logits.view( shape_t{ 1, 1, kGemmaVocab }, row * kGemmaVocab );
            DeviceToken token( Device::Cuda( 0 ), shape_t{ 1, 1 } );
            one_row_op->enqueueForwardOnDevice( view, token, greedy, 0.5f );
            round[ static_cast<size_t>( row ) ] = readTokens( token, 1 )[ 0 ];
        }

        HostToken host_round( Device::Cpu(), shape_t{ kRows } );
        std::copy( round.begin(), round.end(), host_round.data() );
        DeviceToken round_tokens( Device::Cuda( 0 ), shape_t{ kRows } );
        copy( host_round, round_tokens, ctx_.get() );

        auto produced = deviceFlags( kGemmaVocab, base_flagged );

        SamplingParams sampled;
        sampled.temperature = 1.0f;
        sampled.top_k = 64;
        sampled.top_p = 0.95f;

        for ( SamplingParams params : { greedy, sampled } )
        {
            params.presence_penalty = kPenalty;

            const std::vector<float> uniforms = uniformsFrom( 21, kRows );
            DeviceToken tokens( Device::Cuda( 0 ), shape_t{ kRows } );
            rows_op->enqueueRowsOnDevice( logits, tokens, kRows, params, uniforms, &produced, &round_tokens );

            std::vector<int32_t> expected;

            for ( int64_t row = 0; row < kRows; ++row )
            {
                std::vector<int32_t> flagged = base_flagged;
                flagged.insert( flagged.end(), round.begin() + 1, round.begin() + 1 + row );
                auto row_produced = deviceFlags( kGemmaVocab, flagged );

                auto view = logits.view( shape_t{ 1, 1, kGemmaVocab }, row * kGemmaVocab );
                DeviceToken token( Device::Cuda( 0 ), shape_t{ 1, 1 } );
                one_row_op->enqueueForwardOnDevice( view, token, params, uniforms[ static_cast<size_t>( row ) ], &row_produced );
                expected.push_back( readTokens( token, 1 )[ 0 ] );
            }

            const std::vector<int32_t> drawn = readTokens( tokens, kRows );

            EXPECT_EQ( drawn, expected ) << "temperature=" << params.temperature;
            EXPECT_EQ( flaggedTokens( produced ), ( std::vector<int32_t>{ 5, 77, 1234 } ) ) << "rows mark nothing";

            if ( params.temperature == 0.0f )
            {
                for ( int64_t row = 1; row < kRows; ++row )
                    EXPECT_NE( drawn[ static_cast<size_t>( row ) ], round[ static_cast<size_t>( row ) ] )
                        << "row " << row << " drew the draft it counts";
            }
        }

        // Without the penalty, the round's drafts change nothing.
        const std::vector<float> uniforms = uniformsFrom( 21, kRows );
        DeviceToken with_round( Device::Cuda( 0 ), shape_t{ kRows } );
        DeviceToken without_round( Device::Cuda( 0 ), shape_t{ kRows } );
        rows_op->enqueueRowsOnDevice( logits, with_round, kRows, sampled, uniforms, &produced, &round_tokens );
        rows_op->enqueueRowsOnDevice( logits, without_round, kRows, sampled, uniforms );

        EXPECT_EQ( readTokens( with_round, kRows ), readTokens( without_round, kRows ) );
    }

    // The acceptance walk marks the drafts it keeps and the next token, never a rejected draft.
    TEST_F( SamplingRowsCudaTests, Accept_MarksTheKeptTokensProduced )
    {
        auto op = makeRowsOp( kRows, 0.0f, kVocab * 16 );

        HostToken host_tokens( Device::Cpu(), shape_t{ kRows } );
        HostToken host_chosen( Device::Cpu(), shape_t{ kRows } );
        const int32_t round[] = { 50, 11, 12, 13, 14 };
        const int32_t chosen_at[] = { 11, 12, 77, 14, 15 };

        for ( int64_t i = 0; i < kRows; ++i )
        {
            host_tokens.data()[ i ] = round[ i ];
            host_chosen.data()[ i ] = chosen_at[ i ];
        }

        DeviceToken tokens( Device::Cuda( 0 ), shape_t{ kRows } );
        DeviceToken chosen( Device::Cuda( 0 ), shape_t{ kRows } );
        DeviceToken result( Device::Cuda( 0 ), shape_t{ kRows + 1 } );
        copy( host_tokens, tokens, ctx_.get() );
        copy( host_chosen, chosen, ctx_.get() );

        auto produced = deviceFlags( kVocab * 16, { 50 } );
        op->enqueueAcceptOnDevice( tokens, chosen, kRows, result, &produced );

        EXPECT_EQ( readTokens( result, 4 ), ( std::vector<int32_t>{ 2, 11, 12, 77 } ) );
        EXPECT_EQ( flaggedTokens( produced ), ( std::vector<int32_t>{ 11, 12, 50, 77 } ) );
    }

    // Suppressed tokens are never drawn by a rows call either, greedy or sampled.
    TEST_F( SamplingRowsCudaTests, Rows_NeverDrawASuppressedToken )
    {
        auto logits = deviceRows( kRows, kGemmaVocab, 900 );
        auto plain_op = makeRowsOp( kRows );

        SamplingParams greedy;
        greedy.temperature = 0.0f;

        // Suppress each row's own greedy choice.
        DeviceToken greedy_tokens( Device::Cuda( 0 ), shape_t{ kRows } );
        plain_op->enqueueRowsOnDevice( logits, greedy_tokens, kRows, greedy, std::vector<float>( kRows, 0.5f ) );
        const std::vector<int32_t> suppressed = readTokens( greedy_tokens, kRows );

        SamplingConfig config = SamplingConfig{}
            .withVocabularySize( kGemmaVocab )
            .withMaximumRows( kRows )
            .withSuppressedTokens( suppressed );
        OpType op( ctx_.get(), config );

        SamplingParams sampled;
        sampled.temperature = 1.0f;
        sampled.top_k = 64;
        sampled.top_p = 0.95f;

        for ( const SamplingParams& params : { greedy, sampled } )
        {
            for ( uint32_t trial = 0; trial < 8; ++trial )
            {
                DeviceToken tokens( Device::Cuda( 0 ), shape_t{ kRows } );
                op.enqueueRowsOnDevice( logits, tokens, kRows, params, uniformsFrom( 40 + trial, kRows ) );

                for ( const int32_t token : readTokens( tokens, kRows ) )
                    EXPECT_FALSE( std::ranges::contains( suppressed, token ) ) << "drew suppressed token " << token;
            }
        }
    }

    // ------------------------------------------------------------------
    // TokenSampler: the reply's produced tokens across draws
    // ------------------------------------------------------------------

    // A reply begun with a penalty starts from nothing produced, each draw counts, and the next reply starts over.
    // The synchronous sample() reads its token back on the context's stream.
    TEST_F( SamplingCudaTests, TokenSampler_ProducedTokensLastOneReply )
    {
        TokenSampler<DeviceType::Cuda, TensorDataType::FP32> sampler(
            ctx_.get(), SamplingConfig{}.withVocabularySize( kVocab ) );
        auto logits = deviceLogits( { 1, 2, 3, 9, 8.5f, 8, 0, 0 } );
        DeviceToken token( Device::Cuda( 0 ), shape_t{ 1, 1 } );

        SamplingParams params;
        params.temperature = 0.0f;
        params.presence_penalty = 1.5f;

        EXPECT_THROW( sampler.sample( logits, token, params ), std::logic_error ) << "no reply begun with a penalty";

        sampler.beginReply( params );

        EXPECT_EQ( sampler.sample( logits, token, params ), 3 );

        sampler.enqueueSample( logits, token, params );
        EXPECT_EQ( sampler.awaitToken(), 4 );
        EXPECT_EQ( sampler.sample( logits, token, params ), 5 );

        sampler.beginReply( params );

        EXPECT_EQ( sampler.sample( logits, token, params ), 3 ) << "a new reply counts nothing produced";

        SamplingParams off = params;
        off.presence_penalty = 0.0f;
        sampler.beginReply( off );

        EXPECT_EQ( sampler.producedTokens(), nullptr );
        EXPECT_EQ( sampler.sample( logits, token, off ), 3 );
        EXPECT_THROW( sampler.sample( logits, token, params ), std::logic_error );
    }

    // Timing, not a gate: one rows call against K + 1 one-row calls at the Gemma vocabulary and Google's sampling,
    // DRAM-resident logits, events on the context's stream (Gemma4Mtp.md 4.8, "priced before it is built").
    TEST_F( SamplingRowsCudaTests, DISABLED_RowsRate_AtGemmaVocab )
    {
        constexpr int kRepeats = 50;

        auto rows_op = makeRowsOp( kMaxSampleRows, 30.0f );
        auto one_row_op = makeOp( 30.0f, kGemmaVocab );
        auto logits = deviceRows( kMaxSampleRows, kGemmaVocab, 500 );

        // Larger than L2, so each call reads its logits from DRAM as a verify's head output would be.
        Tensor<TensorDataType::FP32, CudaDeviceMemoryResource> flush( Device::Cuda( 0 ), shape_t{ 64 * 1024 * 1024 } );

        auto* context = dynamic_cast<CudaExecutionContext*>( ctx_.get() );
        ASSERT_NE( context, nullptr );
        cudaStream_t stream = context->getStream();

        const auto flushCache = [&]
        {
            cudaMemsetAsync( flush.rawData(), 0, flush.size() * sizeof( float ), stream );
        };

        const auto timeMs = [&]( auto&& call ) -> double
        {
            cudaEvent_t begin;
            cudaEvent_t end;
            cudaEventCreate( &begin );
            cudaEventCreate( &end );

            std::vector<double> samples;

            for ( int repeat = 0; repeat < kRepeats; ++repeat )
            {
                flushCache();
                cudaEventRecord( begin, stream );
                call();
                cudaEventRecord( end, stream );
                cudaEventSynchronize( end );

                float ms = 0.0f;
                cudaEventElapsedTime( &ms, begin, end );
                samples.push_back( ms );
            }

            cudaEventDestroy( begin );
            cudaEventDestroy( end );
            std::sort( samples.begin(), samples.end() );

            return samples[ samples.size() / 2 ];
        };

        struct SettingsCase { const char* name; float temperature; int top_k; float top_p; };
        const SettingsCase cases[] = { { "google 1.0/64/0.95", 1.0f, 64, 0.95f }, { "greedy", 0.0f, 0, 1.0f } };

        DeviceToken tokens( Device::Cuda( 0 ), shape_t{ kMaxSampleRows } );

        for ( const auto& settings : cases )
        {
            SamplingParams params;
            params.temperature = settings.temperature;
            params.top_k = settings.top_k;
            params.top_p = settings.top_p;

            for ( int64_t rows = 1; rows <= kMaxSampleRows; ++rows )
            {
                const std::vector<float> uniforms = uniformsFrom( 3, rows );

                const double one_by_one = timeMs( [&]
                {
                    for ( int64_t row = 0; row < rows; ++row )
                    {
                        auto view = logits.view( shape_t{ 1, 1, kGemmaVocab }, row * kGemmaVocab );
                        auto slot = tokens.view( shape_t{ 1, 1 }, row );
                        one_row_op->enqueueForwardOnDevice( view, slot, params, uniforms[ static_cast<size_t>( row ) ] );
                    }
                } );

                const double together = timeMs( [&]
                {
                    rows_op->enqueueRowsOnDevice( logits, tokens, rows, params, uniforms );
                } );

                std::printf( "[rows-rate] %-20s rows=%lld  one-by-one %.3f ms  rows call %.3f ms  saved %.3f ms\n",
                    settings.name, static_cast<long long>( rows ), one_by_one, together, one_by_one - together );
            }
        }
    }
}
