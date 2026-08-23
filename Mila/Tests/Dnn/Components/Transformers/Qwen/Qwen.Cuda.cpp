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

    // ====================================================================
    // F. Identity
    // ====================================================================

    TEST_F( QwenTransformerCudaTests, GetModelType_IsQwen )
    {
        QwenCuda net( "qwen", allAttentionConfig(), Device::Cuda( 0 ) );

        EXPECT_EQ( net.getModelType(), ModelType::Qwen );
    }
}
