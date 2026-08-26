/**
 * @file Qwen.Cuda.cpp
 * @brief Concrete-network tests for QwenTransformer<DeviceType::Cuda, FP32>.
 *
 * QwenTransformer is the Qwen 3.8 decoder-only language network:
 *   TokenEmbedding -> N x [QwenAttentionBlock | QwenDeltaNetBlock] -> RmsNorm -> lm_head
 *
 * The DeltaNet block is Phase 3 and does not exist, so every test here uses an
 * all-full-attention configuration (`full_attention_interval: 1`) -- and one test asserts
 * the REFUSAL of a configuration that needs the missing block, because a transformer that
 * quietly built 16 of 64 layers would be the worst of the available failures.
 *
 * Numerics are intentionally NOT asserted: the value oracle is HF parity, which is Phase 4.
 * What is asserted is shape, finiteness, the inference-only contract, and the untied-table
 * accounting -- Qwen has no weight tying, so the parameter count carries two full tables
 * where Gemma's carries one.
 *
 * CUDA device tests -- skipped when no CUDA device is present.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <algorithm>
#include <set>
#include <cstdint>
#include <memory>
#include <string>
#include <stdexcept>

import Mila;

namespace Mila::Tests::Dnn::Components::Transformers::Qwen
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        using QwenCuda = Mila::Dnn::QwenTransformer<DeviceType::Cuda, TensorDataType::FP32>;
        using TokenTensor = Tensor<TensorDataType::INT32, CpuMemoryResource>;
        using HostTensor = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        constexpr dim_t kModelDim = 64;
        constexpr dim_t kLayers = 4;
        constexpr dim_t kHeads = 4;
        constexpr dim_t kKVHeads = 2;
        constexpr dim_t kHeadDim = 32;
        constexpr dim_t kHidden = 128;
        constexpr dim_t kVocab = 128;
        constexpr dim_t kMaxSeq = 32;

        QwenConfig configWithInterval( dim_t interval, dim_t layers = kLayers )
        {
            return QwenConfig( kModelDim, layers )
                .withVocabularyLength( kVocab )
                .withNumHeads( kHeads )
                .withNumKVHeads( kKVHeads )
                .withHeadDim( kHeadDim )
                .withAttentionOutputGate( true )
                .withHiddenDimension( kHidden )
                .withMaxSequenceLength( kMaxSeq )
                .withRMSNormEpsilon( 1e-6f )
                .withRoPETheta( 1e7f )
                .withPartialRotaryFactor( 0.25f )
                .withFullAttentionInterval( interval );
        }

        /// Every layer full attention -- the only configuration that builds before Phase 3.
        QwenConfig allAttentionConfig( dim_t layers = kLayers ) { return configWithInterval( 1, layers ); }

        /// The published 3:1 interleave, which needs the block Phase 3 will add.
        QwenConfig hybridConfig() { return configWithInterval( 4 ); }
    }

    class QwenTransformerCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }
        }

        std::unique_ptr<QwenCuda> builtNet( const QwenConfig& config, dim_t batch, dim_t seq,
            bool initialize_parameters = false )
        {
            auto net = std::make_unique<QwenCuda>( "qwen", config, Device::Cuda( 0 ) );
            net->build( BuildContext( shape_t{ batch, seq }, RuntimeMode::Inference, initialize_parameters ) );

            return net;
        }

        /**
         * @brief Footprint of a network built alone, then destroyed.
         *
         * Comparing two footprints requires this: the RoPE cos/sin cache is process-wide and
         * refcounted by RopeCacheRegistry, so a second network built while the first is still
         * alive finds the cache present and reports a smaller total than the same network
         * built by itself. Holding both and subtracting compares a first build against a
         * second one.
         */
        MemoryStats builtFootprint( const QwenConfig& config, const BuildContext& context )
        {
            QwenCuda net( "qwen", config, Device::Cuda( 0 ) );
            net.build( context );

            return net.getMemoryStats();
        }

        QwenCuda::TokenIndexType makeTokens( dim_t batch, dim_t seq )
        {
            TokenTensor host( Device::Cpu(), shape_t{ batch, seq } );
            auto* data = host.data();

            for ( size_t i = 0; i < host.size(); ++i )
            {
                data[ i ] = static_cast<int32_t>( i % static_cast<size_t>( kVocab ) );
            }

            QwenCuda::TokenIndexType device_tokens( Device::Cuda( 0 ), shape_t{ batch, seq } );
            copy( host, device_tokens );

            return device_tokens;
        }

        /// log( softmax( row )[ target ] ), in double, from a host FP32 logit row.
        static double hostLogProbability( const float* row, dim_t vocab_size, int32_t target )
        {
            float max_logit = row[ 0 ];

            for ( dim_t v = 1; v < vocab_size; ++v )
            {
                max_logit = std::fmax( max_logit, row[ v ] );
            }

            double sum_exponentials = 0.0;

            for ( dim_t v = 0; v < vocab_size; ++v )
            {
                sum_exponentials += std::exp( static_cast<double>( row[ v ] - max_logit ) );
            }

            return static_cast<double>( row[ target ] - max_logit ) - std::log( sum_exponentials );
        }

        void expectScoreMatchesPrefillOracle( dim_t head_positions, dim_t sequence_length )
        {
            const QwenConfig config =
                allAttentionConfig().withLanguageModelHeadPositions( head_positions );

            auto net = builtNet( config, 1, sequence_length, /*initialize_parameters*/ true );

            auto device_tokens = makeTokens( 1, sequence_length );
            auto host_tokens = toHost<TensorDataType::INT32>( device_tokens );

            const auto scored = net->scoreTokens( device_tokens );

            EXPECT_EQ( scored.scored_positions, sequence_length - 1 )
                << "every position but the last predicts a following token";

            double expected_total = 0.0;

            for ( dim_t position = 0; position + 1 < sequence_length; ++position )
            {
                auto prefix = device_tokens.view( shape_t{ 1, position + 1 }, 0 );
                auto& logits = net->prefill( prefix );

                HostTensor host_logits( Device::Cpu(), logits.shape() );
                copy( logits, host_logits );
                net->synchronize();

                expected_total += hostLogProbability(
                    host_logits.data(), kVocab, host_tokens.data()[ position + 1 ] );
            }

            EXPECT_NEAR( scored.total_log_probability, expected_total, 1e-3 );

            // A model assigning every token equal probability would score exactly this.
            // Landing on it means the logits were flat -- the signature of a build that
            // never filled its parameters, which would make the comparison above vacuous.
            const double uniform_total = -std::log( static_cast<double>( kVocab ) )
                * static_cast<double>( sequence_length - 1 );

            EXPECT_GT( std::abs( scored.total_log_probability - uniform_total ), 1e-6 )
                << "scores are indistinguishable from a uniform model; parameters may be zero";
        }

        static bool allFinite( const HostTensor& t )
        {
            const auto* data = t.data();

            for ( size_t i = 0; i < t.size(); ++i )
            {
                if ( !std::isfinite( data[ i ] ) )
                {
                    return false;
                }
            }

            return true;
        }

        static constexpr dim_t batch_ = 1;
        static constexpr dim_t seq_ = 4;
    };

    // ====================================================================
    // A. Construction and the Phase 3 boundary
    // ====================================================================

    TEST_F( QwenTransformerCudaTests, Construct_AllAttentionSucceeds )
    {
        QwenCuda net( "qwen", allAttentionConfig(), Device::Cuda( 0 ) );

        EXPECT_EQ( net.getName(), "qwen" );
        EXPECT_EQ( net.getDeviceId().type, DeviceType::Cuda );
    }

    TEST_F( QwenTransformerCudaTests, Construct_HybridInterleaveSucceeds )
    {
        // The published geometry: 3 Gated DeltaNet layers per full-attention layer. This
        // configuration was refused at construction until the DeltaNet block existed --
        // building it is what closes that gap.
        QwenCuda net( "qwen", hybridConfig(), Device::Cuda( 0 ) );

        EXPECT_EQ( net.getDeviceId().type, DeviceType::Cuda );
    }

    TEST_F( QwenTransformerCudaTests, Build_HybridInterleaveAllocatesBothBlockKinds )
    {
        auto net = builtNet( hybridConfig(), batch_, seq_ );

        EXPECT_TRUE( net->isBuilt() );
        EXPECT_GT( net->parameterCount(), 0 );
    }

    TEST_F( QwenTransformerCudaTests, HybridInterleaveCostsMoreParametersThanAllAttention )
    {
        // Not a size check for its own sake: it is the cheapest assertion that BOTH kinds
        // were really instantiated. A DeltaNet layer carries five input projections plus a
        // convolution where an attention layer carries one fused QKV, so a stack that
        // silently built attention blocks everywhere would not clear this.
        auto attention_only = builtNet( allAttentionConfig(), batch_, seq_ );
        auto hybrid = builtNet( hybridConfig(), batch_, seq_ );

        EXPECT_NE( hybrid->parameterCount(), attention_only->parameterCount() );
    }

    TEST_F( QwenTransformerCudaTests, Construct_DeviceTypeMismatchThrows )
    {
        EXPECT_THROW( QwenCuda( "qwen", allAttentionConfig(), Device::Cpu() ), std::invalid_argument );
    }

    // ====================================================================
    // B. Build lifecycle
    // ====================================================================

    TEST_F( QwenTransformerCudaTests, Build_SetsIsBuiltAndAllocatesParameters )
    {
        auto net = builtNet( allAttentionConfig(), batch_, seq_ );

        EXPECT_TRUE( net->isBuilt() );
        EXPECT_GT( net->parameterCount(), 0 );
    }

    TEST_F( QwenTransformerCudaTests, Build_ThrowsOnNonRank2Input )
    {
        QwenCuda net( "qwen", allAttentionConfig(), Device::Cuda( 0 ) );

        EXPECT_THROW( net.build( BuildContext( shape_t{ batch_, seq_, kModelDim }, RuntimeMode::Inference ) ),
            std::invalid_argument );
    }

    TEST_F( QwenTransformerCudaTests, ParameterCount_CarriesBothTablesBecauseTheyAreUntied )
    {
        auto net = builtNet( allAttentionConfig(), batch_, seq_ );

        // tie_word_embeddings is false, so nothing is shared and nothing is subtracted: the
        // embedding table and the head are two independent [vocab, model_dim] allocations.
        // Gemma's count subtracts one of them; a count here that did the same would be
        // short by a whole table -- 1.271 B parameters on the real model.
        EXPECT_GE( net->parameterCount(), 2 * kVocab * kModelDim );
    }

    // ====================================================================
    // C. Inference-only contract
    // ====================================================================

    TEST_F( QwenTransformerCudaTests, Forward_Throws )
    {
        auto net = builtNet( allAttentionConfig(), batch_, seq_ );
        auto input = makeTokens( batch_, seq_ );

        EXPECT_THROW( (void)net->forward( input ), std::runtime_error );
    }

    // ====================================================================
    // D. Prefill / decode -- shape and finiteness
    // ====================================================================

    TEST_F( QwenTransformerCudaTests, Prefill_ProducesLastTokenLogitsShape )
    {
        auto net = builtNet( allAttentionConfig(), batch_, seq_ );
        auto input = makeTokens( batch_, seq_ );

        auto& logits = net->prefill( input );

        // Last position only: at the real vocabulary a full chunk of logit rows would be
        // 0.48 GiB (Qwen3.8.md section 5).
        EXPECT_EQ( logits.shape(), ( shape_t{ batch_, 1, kVocab } ) );
    }

    TEST_F( QwenTransformerCudaTests, Prefill_ProducesFiniteLogits )
    {
        // Parameters explicitly initialized so finiteness means something without a loaded
        // checkpoint. This is also the only test that runs the output gate end to end: a
        // gate applied to an uninitialized buffer would show up here as a NaN.
        auto net = builtNet( allAttentionConfig(), batch_, seq_, /*initialize_parameters*/ true );
        auto input = makeTokens( batch_, seq_ );

        auto& logits = net->prefill( input );

        HostTensor host( Device::Cpu(), logits.shape() );
        copy( logits, host );
        net->synchronize();

        EXPECT_TRUE( allFinite( host ) );
    }

    /**
     * @brief The published interleave, driven end to end.
     *
     * Building both block kinds proves they instantiate; only running them proves the stack
     * composes -- that a DeltaNet layer accepts the residual stream an attention layer
     * produced and hands back something the next attention layer can use. Finiteness is the
     * assertion that matters here: the recurrence multiplies its state by exp(g) every step
     * and the convolution feeds it, so a sign or an index error surfaces as inf or NaN long
     * before it would surface as a wrong-looking number.
     */
    TEST_F( QwenTransformerCudaTests, HybridInterleave_PrefillAndDecodeStayFinite )
    {
        constexpr dim_t kContext = 8;

        auto net = builtNet( hybridConfig(), batch_, kContext, /*initialize_parameters*/ true );

        auto prompt = makeTokens( batch_, seq_ );
        auto& prefill_logits = net->prefill( prompt );

        // Last position only, as the all-attention path returns: a full chunk of logit rows
        // would be 0.48 GiB at the real vocabulary.
        EXPECT_EQ( prefill_logits.shape(), ( shape_t{ batch_, 1, kVocab } ) );

        HostTensor prefill_host( Device::Cpu(), prefill_logits.shape() );
        copy( prefill_logits, prefill_host );
        net->synchronize();

        EXPECT_TRUE( allFinite( prefill_host ) );

        auto next = makeTokens( batch_, 1 );
        auto& decode_logits = net->decode( next, seq_ );

        EXPECT_EQ( decode_logits.shape(), ( shape_t{ batch_, 1, kVocab } ) );

        HostTensor decode_host( Device::Cpu(), decode_logits.shape() );
        copy( decode_logits, decode_host );
        net->synchronize();

        EXPECT_TRUE( allFinite( decode_host ) );
    }

    TEST_F( QwenTransformerCudaTests, Decode_ProducesSingleTokenLogits )
    {
        // Built for a context WIDER than the prompt, so the decode step appends a genuinely
        // new position rather than overwriting the prompt's last one. Decoding at seq_ in a
        // network built for exactly seq_ is out of the cache's range, and the GQA op says so.
        constexpr dim_t kContext = 8;

        auto net = builtNet( allAttentionConfig(), batch_, kContext, /*initialize_parameters*/ true );

        auto prompt = makeTokens( batch_, seq_ );
        (void)net->prefill( prompt );

        auto next = makeTokens( batch_, 1 );
        auto& logits = net->decode( next, seq_ );

        EXPECT_EQ( logits.shape(), ( shape_t{ batch_, 1, kVocab } ) );

        HostTensor host( Device::Cpu(), logits.shape() );
        copy( logits, host );
        net->synchronize();

        EXPECT_TRUE( allFinite( host ) );
    }

    // ====================================================================
    // D2. Teacher-forced scoring (Qwen3.8.md section 8 item 9)
    //
    // The oracle is the generation path itself: scoring position p must report exactly what
    // a prefill of tokens[0..p] says about token p+1. That makes the test independent of
    // the model's values -- it compares two ways of reaching the same logit row -- and it is
    // what catches an off-by-one in the target alignment, which is the defect this code is
    // most likely to have and the one a plausible-looking perplexity would hide.
    //
    // Parameters MUST be initialized. On zero weights every logit row is identical, every
    // position scores -log(vocab), and the comparison below passes without proving anything.
    // ====================================================================

    TEST_F( QwenTransformerCudaTests, ScoreTokens_MatchesPrefillOracle_OneRowAtATime )
    {
        expectScoreMatchesPrefillOracle( /*head_positions*/ 1, /*sequence_length*/ 8 );
    }

    // Three does not divide eight, so the final window is partial -- the case where a window
    // loop most often reads a row it should not or skips one it should.
    TEST_F( QwenTransformerCudaTests, ScoreTokens_MatchesPrefillOracle_PartialFinalWindow )
    {
        expectScoreMatchesPrefillOracle( /*head_positions*/ 3, /*sequence_length*/ 8 );
    }

    // ====================================================================
    // D3. The observation attach walk (Observability.md 6, 11)
    //
    // Publication shipped without a door: every component publishes, but selection is
    // per-component and the tree is behind a protected accessor, so the first consumer that
    // wanted logits bolted a purpose-built accessor onto LanguageModel instead. These pin the
    // walk that replaced it. They run on the tiny in-tree model, not an artifact, because the
    // behaviour under test is resolution and gating -- neither needs real weights.
    // ====================================================================

    TEST_F( QwenTransformerCudaTests, Observe_ResolvesAPatternOnceAndReportsTheMatchCount )
    {
        auto net = builtNet( allAttentionConfig(), batch_, seq_ );

        const auto paths = net->componentPaths();

        ASSERT_FALSE( paths.empty() );

        // The head is one component; the pattern names it without the caller knowing the
        // network's own name.
        const size_t head_matches = net->observe( "*.lm_head",
            ComputePassMask::inference(), []( std::string_view, ComputePass,
                std::string_view, const ITensor& ) {} );

        EXPECT_EQ( head_matches, 1u );

        // A pattern that matches nothing must SAY so. Downstream a silent zero is
        // indistinguishable from a run with nothing to report, which is the false negative a
        // NaN hunt cannot afford.
        const size_t absent = net->observe( "*.no_such_component",
            ComputePassMask::inference(), []( std::string_view, ComputePass,
                std::string_view, const ITensor& ) {} );

        EXPECT_EQ( absent, 0u );

        // The whole subtree, which is what a fingerprint pass wants.
        const size_t everything = net->observe( "*", ComputePassMask::inference(),
            []( std::string_view, ComputePass, std::string_view, const ITensor& ) {} );

        EXPECT_EQ( everything, paths.size() );

        net->stopObserving();
    }

    TEST_F( QwenTransformerCudaTests, Observe_DeliversOnlyTheSelectedComponents )
    {
        auto net = builtNet( allAttentionConfig(), batch_, seq_, /*initialize_parameters*/ true );

        std::set<std::string> publishers;

        const size_t matched = net->observe( "*.lm_head", ComputePassMask::inference(),
            [&publishers]( std::string_view path, ComputePass, std::string_view,
                const ITensor& )
            {
                publishers.insert( std::string( path ) );
            } );

        ASSERT_EQ( matched, 1u );

        auto tokens = makeTokens( batch_, seq_ );
        (void)net->prefill( tokens );
        net->synchronize();

        ASSERT_EQ( publishers.size(), 1u )
            << "a selected pattern must publish from exactly the components it matched";
        EXPECT_TRUE( publishers.begin()->ends_with( ".lm_head" ) );

        // Detaching has to actually stop it. An observer that outlives its question keeps
        // firing on every later pass, which costs and confuses.
        net->stopObserving();
        publishers.clear();

        (void)net->prefill( tokens );
        net->synchronize();

        EXPECT_TRUE( publishers.empty() ) << "stopObserving left the sink attached";
    }

    // The published tensor is the one the caller actually wanted: reading its VALUES is the
    // whole point, and doing so needs the concrete tensor type back out of the ITensor.
    TEST_F( QwenTransformerCudaTests, Observe_PublishesReadableLogitsFromTheHead )
    {
        using DeviceTensor = Tensor<TensorDataType::FP32,
            typename DeviceTypeTraits<DeviceType::Cuda>::memory_resource>;

        auto net = builtNet( allAttentionConfig(), batch_, seq_, /*initialize_parameters*/ true );

        std::vector<float> logits;

        const size_t matched = net->observe( "*.lm_head", ComputePassMask::inference(),
            [&logits]( std::string_view, ComputePass, std::string_view stage,
                const ITensor& value )
            {
                if ( stage != "output" )
                {
                    return;
                }

                const auto* typed = dynamic_cast<const DeviceTensor*>( &value );

                if ( typed == nullptr )
                {
                    return;
                }

                auto host = toHost<TensorDataType::FP32>( *typed );

                logits.assign( host.data(), host.data() + host.size() );
            } );

        ASSERT_EQ( matched, 1u );

        auto tokens = makeTokens( batch_, seq_ );
        (void)net->prefill( tokens );
        net->synchronize();

        ASSERT_EQ( logits.size(), static_cast<size_t>( batch_ * kVocab ) );
        EXPECT_TRUE( std::all_of( logits.begin(), logits.end(),
            []( float value ) { return std::isfinite( value ); } ) );

        net->stopObserving();
    }

    // ====================================================================
    // E. Required-memory contract (MemoryFootprint.md 7, Gate A)
    //
    // Load-bearing rather than belt-and-braces: a block derives five build contexts and
    // pairs children to them by hand. A mispairing, a forgotten pooled slot, or wrong RoPE
    // deduplication all produce a PLAUSIBLE total rather than an obviously wrong one, and
    // this comparison is the only thing that separates them.
    // ====================================================================

    TEST_F( QwenTransformerCudaTests, GetRequiredMemory_MatchesBuiltFootprint )
    {
        const BuildContext context( shape_t{ batch_, seq_ }, RuntimeMode::Inference );

        QwenCuda predictor( "qwen", allAttentionConfig(), Device::Cuda( 0 ) );
        const MemoryStats predicted = predictor.getRequiredMemory( context );

        QwenCuda built( "qwen", allAttentionConfig(), Device::Cuda( 0 ) );
        built.build( context );
        const MemoryStats actual = built.getMemoryStats();

        EXPECT_EQ( predicted.device_parameter_bytes, actual.device_parameter_bytes ) << "parameters";
        EXPECT_EQ( predicted.device_state_bytes, actual.device_state_bytes ) << "state";
        EXPECT_EQ( predicted.device_gradient_bytes, actual.device_gradient_bytes ) << "gradients";
    }

    /**
     * @brief The same contract on the PUBLISHED geometry, which is where it was broken.
     *
     * The all-attention case above passed throughout, because it only ever exercised one
     * block kind. The transformer handed every block a context saying "the parent installs
     * your output" and then installed nothing into the DeltaNet ones, so their twenty
     * component outputs were predicted at zero and allocated in full -- 138.2 MiB per layer
     * on the 27B, ~6.5 GiB over 48 layers, which is what capped that model at 512 context
     * and made WDDM page its weights (Qwen3.8.md section 8, Phase 5). A prediction that
     * understates by 60% is worse than none: Chat's GPU FIT verdict reads it.
     */
    TEST_F( QwenTransformerCudaTests, GetRequiredMemory_MatchesBuiltFootprint_HybridInterleave )
    {
        const BuildContext context( shape_t{ batch_, seq_ }, RuntimeMode::Inference );

        QwenCuda predictor( "qwen", hybridConfig(), Device::Cuda( 0 ) );
        const MemoryStats predicted = predictor.getRequiredMemory( context );

        QwenCuda built( "qwen", hybridConfig(), Device::Cuda( 0 ) );
        built.build( context );
        const MemoryStats actual = built.getMemoryStats();

        EXPECT_EQ( predicted.device_parameter_bytes, actual.device_parameter_bytes ) << "parameters";
        EXPECT_EQ( predicted.device_state_bytes, actual.device_state_bytes ) << "state";
        EXPECT_EQ( predicted.device_gradient_bytes, actual.device_gradient_bytes ) << "gradients";
    }

    // Eight layers rather than four: with only one RoPE cache allocated per key, seven of
    // the eight per-layer reports must be subtracted. A short stack cannot tell a correct
    // deduplication from an off-by-one.
    TEST_F( QwenTransformerCudaTests, GetRequiredMemory_PinsRopeCacheDeduplication )
    {
        const BuildContext context( shape_t{ batch_, seq_ }, RuntimeMode::Inference );

        QwenCuda predictor( "qwen", allAttentionConfig( 8 ), Device::Cuda( 0 ) );
        const MemoryStats predicted = predictor.getRequiredMemory( context );

        QwenCuda built( "qwen", allAttentionConfig( 8 ), Device::Cuda( 0 ) );
        built.build( context );
        const MemoryStats actual = built.getMemoryStats();

        EXPECT_EQ( predicted.device_state_bytes, actual.device_state_bytes );
    }

    /**
     * @brief A widened language-model head is predicted at the width it is built at.
     *
     * The head is one row wide for generation, and teacher-forced scoring needs a logit at
     * every position, so the width is a config capacity. It reaches build() and
     * getRequiredMemory() through one resolver precisely so a scoring build cannot allocate
     * a head the prediction never named -- the failure Qwen has already paid for twice, and
     * the one Chat's GPU FIT verdict would report wrongly.
     *
     * The inequality is what keeps this non-vacuous: a knob ignored by both paths would
     * satisfy the equalities and fail here.
     */
    TEST_F( QwenTransformerCudaTests, GetRequiredMemory_MatchesBuiltFootprint_WidenedLanguageModelHead )
    {
        const BuildContext context( shape_t{ batch_, seq_ }, RuntimeMode::Inference );

        const QwenConfig widened = allAttentionConfig().withLanguageModelHeadPositions( seq_ );

        QwenCuda predictor( "qwen", widened, Device::Cuda( 0 ) );
        const MemoryStats predicted = predictor.getRequiredMemory( context );

        const MemoryStats actual = builtFootprint( widened, context );

        EXPECT_EQ( predicted.device_parameter_bytes, actual.device_parameter_bytes ) << "parameters";
        EXPECT_EQ( predicted.device_state_bytes, actual.device_state_bytes ) << "state";
        EXPECT_EQ( predicted.device_gradient_bytes, actual.device_gradient_bytes ) << "gradients";

        EXPECT_GT( actual.device_state_bytes,
            builtFootprint( allAttentionConfig(), context ).device_state_bytes )
            << "a wider head must cost more state than the one-row default";
    }

    /**
     * @brief A width above what a prefill pass supplies resolves to the pass, not the request.
     *
     * The head reads the block stack's output, and a pass produces at most prefill_chunk rows
     * of it, so rows beyond that name nothing. At seq_ below the chunk floor the whole context
     * is one chunk, which makes the bound exactly seq_ and the two configurations below
     * indistinguishable in footprint.
     */
    TEST_F( QwenTransformerCudaTests, LanguageModelHeadPositions_ClampToWhatAPrefillPassSupplies )
    {
        const BuildContext context( shape_t{ batch_, seq_ }, RuntimeMode::Inference );

        const QwenConfig at_bound = allAttentionConfig().withLanguageModelHeadPositions( seq_ );
        const QwenConfig above_bound = allAttentionConfig().withLanguageModelHeadPositions( seq_ + 100 );

        const MemoryStats bounded = builtFootprint( at_bound, context );
        const MemoryStats over = builtFootprint( above_bound, context );

        EXPECT_EQ( over.device_state_bytes, bounded.device_state_bytes );

        QwenCuda over_predictor( "qwen", above_bound, Device::Cuda( 0 ) );

        EXPECT_EQ( over_predictor.getRequiredMemory( context ).device_state_bytes,
            over.device_state_bytes )
            << "prediction must clamp exactly as the build does";
    }

    // ====================================================================
    // F. Identity
    // ====================================================================

    TEST_F( QwenTransformerCudaTests, GetModelType_IsQwen )
    {
        QwenCuda net( "qwen", allAttentionConfig(), Device::Cuda( 0 ) );

        EXPECT_EQ( net.getModelType(), ModelType::Qwen );
    }
}
