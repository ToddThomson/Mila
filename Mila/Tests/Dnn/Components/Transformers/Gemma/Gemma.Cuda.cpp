/**
 * @file Gemma.Cuda.cpp
 * @brief Concrete-network tests for GemmaTransformer<DeviceType::Cuda, FP32>.
 *
 * GemmaTransformer is the Gemma 4 decoder-only language network:
 *   TokenEmbedding -> N x GemmaBlock (heterogeneous local/global) -> RmsNorm -> lm_head
 *
 * Like LlamaTransformer it is a LanguageNetwork, but INFERENCE-ONLY: forward()/
 * backward() throw, and generation is driven by prefill()/decode(). This suite tests
 * the GemmaTransformer DELTA over the base: construction/validation, the rank-2
 * [B, T] build contract, the inference-only contract, the heterogeneous layer list
 * (a local + a global GemmaBlock built into one network), logits shape, and type
 * identity. The constituent components are covered in their own suites.
 *
 * Execution (prefill/decode logits shape + finiteness) is asserted on an ALL-LOCAL
 * config, which exercises the validated compact-NKV GQA path. The heterogeneous
 * (with-global) case is exercised for BUILD only: the global K=V / head_dim-512
 * attention kernels are not yet HF-validated (BACKLOG: confirm hand-written GQA
 * kernels at the global geometry), so running them is deferred to the Step 5f
 * parity oracle rather than asserted blind here.
 *
 * Numerics are intentionally NOT asserted -- the value oracle is HF parity (5f).
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

namespace Mila::Tests::Dnn::Components::Transformers::Gemma
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;

    namespace
    {
        using GemmaCuda = Mila::Dnn::GemmaTransformer<DeviceType::Cuda, TensorDataType::FP32>;
        using TokenTensor = Tensor<TensorDataType::INT32, CpuMemoryResource>;
        using HostTensor = Tensor<TensorDataType::FP32, CpuMemoryResource>;

        constexpr int64_t kModelDim      = 64;
        constexpr int64_t kLayers        = 2;
        constexpr int64_t kHeads         = 4;
        constexpr int64_t kKVHeads       = 2;
        constexpr int64_t kHeadDim       = 32;
        constexpr int64_t kGlobalHeadDim = 64;
        constexpr int64_t kHidden        = 128;
        constexpr int64_t kVocab         = 128;
        constexpr int64_t kMaxSeq        = 32;

        // pattern 6 over 2 layers -> both layers local (validated GQA path).
        // pattern 2 over 2 layers -> layer 1 global (heterogeneous build coverage).
        GemmaConfig configWithPattern( int64_t pattern )
        {
            return GemmaConfig( kModelDim, kLayers )
                .withVocabularyLength( kVocab )
                .withNumHeads( kHeads )
                .withNumKVHeads( kKVHeads )
                .withHeadDim( kHeadDim )
                .withGlobalHeadDim( kGlobalHeadDim )
                .withNumGlobalKVHeads( 1 )
                .withKeyEqualsValue( true )
                .withHiddenDimension( kHidden )
                .withMaxSequenceLength( kMaxSeq )
                .withRMSNormEpsilon( 1e-6f )
                .withWindow( 8 )
                .withSlidingWindowPattern( pattern )
                .withGlobalRotaryDim( 32 )
                .withRoPETheta( 10000.0f )
                .withGlobalRoPETheta( 1000000.0f )
                .withFinalLogitSoftcapping( 30.0f );
        }

        GemmaConfig allLocalConfig() { return configWithPattern( 6 ); }
        GemmaConfig heterogeneousConfig() { return configWithPattern( 2 ); }
    }

    class GemmaTransformerCudaTests : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            if ( getDeviceCount( DeviceType::Cuda ) == 0 )
            {
                GTEST_SKIP() << "No CUDA device available";
            }
        }

        std::unique_ptr<GemmaCuda> builtNet( const GemmaConfig& config, int64_t batch, int64_t seq )
        {
            auto net = std::make_unique<GemmaCuda>( "gemma", config, Device::Cuda( 0 ) );
            net->build( BuildContext( shape_t{ batch, seq }, RuntimeMode::Inference ) );

            return net;
        }

        GemmaCuda::TokenIndexType makeTokens( int64_t batch, int64_t seq )
        {
            TokenTensor host( Device::Cpu(), shape_t{ batch, seq } );
            auto* data = host.data();

            for ( size_t i = 0; i < host.size(); ++i )
            {
                data[ i ] = static_cast<int32_t>( i % static_cast<size_t>( kVocab ) );
            }

            GemmaCuda::TokenIndexType device_tokens( Device::Cuda( 0 ), shape_t{ batch, seq } );
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

        static constexpr int64_t batch_ = 1;
        static constexpr int64_t seq_ = 4;
    };

    // ====================================================================
    // A. Construction & Validation
    // ====================================================================

    TEST_F( GemmaTransformerCudaTests, Construct_StandaloneSucceeds )
    {
        GemmaCuda net( "gemma", allLocalConfig(), Device::Cuda( 0 ) );

        EXPECT_EQ( net.getName(), "gemma" );
        EXPECT_EQ( net.getDeviceId().type, DeviceType::Cuda );
    }

    TEST_F( GemmaTransformerCudaTests, Construct_DeviceTypeMismatchThrows )
    {
        EXPECT_THROW( GemmaCuda( "gemma", allLocalConfig(), Device::Cpu() ), std::invalid_argument );
    }

    // ====================================================================
    // B. Build Lifecycle
    // ====================================================================

    TEST_F( GemmaTransformerCudaTests, Build_SetsIsBuilt )
    {
        auto net = builtNet( allLocalConfig(), batch_, seq_ );

        EXPECT_TRUE( net->isBuilt() );
    }

    TEST_F( GemmaTransformerCudaTests, Build_AllocatesParameters )
    {
        auto net = builtNet( allLocalConfig(), batch_, seq_ );

        EXPECT_GT( net->parameterCount(), 0u );
    }

    TEST_F( GemmaTransformerCudaTests, Build_ThrowsOnNonRank2Input )
    {
        GemmaCuda net( "gemma", allLocalConfig(), Device::Cuda( 0 ) );

        // Rank-3 input violates the [B, T] token-index contract.
        EXPECT_THROW( net.build( BuildContext( shape_t{ batch_, seq_, kModelDim }, RuntimeMode::Inference ) ),
            std::invalid_argument );
    }

    // ====================================================================
    // C. Heterogeneous layer list — a local + a global GemmaBlock in one net
    // ====================================================================

    TEST_F( GemmaTransformerCudaTests, HeterogeneousConfig_BuildsBothBlockKinds )
    {
        // pattern 2 makes layer 1 global, so build constructs a LocalBlock AND a
        // GlobalBlock instantiation into the same network. Build only -- the global
        // GQA execution path is deferred to the 5f parity oracle (see file note).
        auto net = builtNet( heterogeneousConfig(), batch_, seq_ );

        EXPECT_TRUE( net->isBuilt() );
        EXPECT_GT( net->parameterCount(), 0u );
        EXPECT_EQ( net->getComponents().size(), static_cast<size_t>( kLayers + 3 ) );
    }

    // ====================================================================
    // D. Inference-only contract
    // ====================================================================

    TEST_F( GemmaTransformerCudaTests, Forward_ThrowsInferenceOnly )
    {
        auto net = builtNet( allLocalConfig(), batch_, seq_ );
        auto input = makeTokens( batch_, seq_ );

        EXPECT_THROW( net->forward( input ), std::runtime_error );
    }

    TEST_F( GemmaTransformerCudaTests, Backward_ThrowsInferenceOnly )
    {
        auto net = builtNet( allLocalConfig(), batch_, seq_ );
        auto input = makeTokens( batch_, seq_ );
        GemmaCuda::TensorType output_grad( Device::Cuda( 0 ), shape_t{ batch_, seq_, kVocab } );

        EXPECT_THROW( net->backward( input, output_grad ), std::runtime_error );
    }

    TEST_F( GemmaTransformerCudaTests, BuiltForInference_IsInferenceMode )
    {
        auto net = builtNet( allLocalConfig(), batch_, seq_ );

        EXPECT_TRUE( net->isInferenceMode() );
        EXPECT_FALSE( net->isTrainingMode() );
    }

    // ====================================================================
    // E. Prefill / Decode (logits shape + finiteness; all-local path)
    // ====================================================================

    TEST_F( GemmaTransformerCudaTests, Prefill_ProducesLastTokenLogitsShape )
    {
        auto net = builtNet( allLocalConfig(), batch_, seq_ );
        auto input = makeTokens( batch_, seq_ );

        auto& logits = net->prefill( input );

        // prefill returns logits for the last position only: [B, 1, vocab].
        EXPECT_EQ( logits.shape(), ( shape_t{ batch_, 1, kVocab } ) );
    }

    TEST_F( GemmaTransformerCudaTests, Prefill_ProducesFiniteLogits )
    {
        // Build with parameters explicitly initialized so finiteness is meaningful
        // without a loaded checkpoint (independent of the inference-mode init default).
        auto net = std::make_unique<GemmaCuda>( "gemma", allLocalConfig(), Device::Cuda( 0 ) );
        net->build( BuildContext( shape_t{ batch_, seq_ }, RuntimeMode::Inference, true ) );

        auto input = makeTokens( batch_, seq_ );

        auto& logits = net->prefill( input );

        HostTensor host( Device::Cpu(), logits.shape() );
        copy( logits, host );
        net->synchronize();

        EXPECT_TRUE( allFinite( host ) );
    }

    TEST_F( GemmaTransformerCudaTests, Decode_ProducesLogitsShape )
    {
        auto net = builtNet( allLocalConfig(), batch_, seq_ );

        // Prefill the prompt, then decode one token at the next position.
        auto prompt = makeTokens( batch_, seq_ );
        net->prefill( prompt );

        auto step = makeTokens( batch_, 1 );

        // REVIEW: These static_casts are a smell that the public decode() interface
        // is not well-aligned with the internal block contract (int vs int64_t).
        // The public interface should be fixed to take int64_t (dim_t) and avoid these.

        auto& logits = net->decode( step, static_cast<int>( seq_ - 1) );

        EXPECT_EQ( logits.shape(), ( shape_t{ batch_, 1, kVocab } ) );
    }

    // ====================================================================
    // F. Execution Context
    // ====================================================================

    TEST_F( GemmaTransformerCudaTests, Synchronize_Succeeds )
    {
        auto net = builtNet( allLocalConfig(), batch_, seq_ );

        EXPECT_NO_THROW( net->synchronize() );
    }

    // ====================================================================
    // G. Parameters & Components
    // ====================================================================

    TEST_F( GemmaTransformerCudaTests, GetComponents_ReturnsLayersPlusThree )
    {
        auto net = builtNet( allLocalConfig(), batch_, seq_ );

        // token_embedding + kLayers blocks + final_rmsnorm + lm_head.
        EXPECT_EQ( net->getComponents().size(), static_cast<size_t>( kLayers + 3 ) );
    }

    // ====================================================================
    // I. Diagnostics
    // ====================================================================

    TEST_F( GemmaTransformerCudaTests, ToString_ContainsExpectedFields )
    {
        GemmaCuda net( "my_gemma", allLocalConfig(), Device::Cuda( 0 ) );

        const std::string text = net.toString();

        EXPECT_NE( text.find( "Gemma Network" ), std::string::npos );
        EXPECT_NE( text.find( "my_gemma" ), std::string::npos );
    }

    // ====================================================================
    // K. KV-cache policy wiring (SlidingWindowKvCache.md Phase 3)
    //
    // Compile-time proof that the transformer routes the sliding-window policy to
    // the LOCAL (sliding) layers only; GLOBAL (full-attention) layers always use
    // the full-context cache regardless of the policy. This pins the per-block-kind
    // selection in Gemma.ixx (the actual byte footprint of each op is asserted
    // closed-form in CudaGqaOp.Cuda.cpp::StateMemory_MatchesClosedFormAndShrinks).
    // ====================================================================

    TEST_F( GemmaTransformerCudaTests, KvPolicy_RoutesBoundedRingToLocalLayersOnly )
    {
        using Bounded = Mila::Dnn::GemmaTransformer<
            DeviceType::Cuda, TensorDataType::FP32,
            Mila::Dnn::Quant::Weight::NoWeightQuant,
            Mila::Dnn::Quant::KvCache::SlidingWindowKvCache>;

        // Assert the routed POLICY (kKvCompressed = TKvPolicy::kIsActive), not the
        // resolved op: naming OpType would complete CudaGqaOp and drag its build()
        // body (and ExecutionContext<Cuda>) into this TU. Phase 0's dispatch test
        // already pins policy -> bounded op; here we pin the transformer's routing.

        // Sliding policy -> local layers carry the active (bounded) policy; global
        // layers stay full regardless (hardwired NoKvCompression in Gemma.ixx).
        static_assert( Bounded::LocalBlockType::AttentionType::kKvCompressed,
            "SlidingWindowKvCache must reach the local (sliding) layers" );
        static_assert( !Bounded::GlobalBlockType::AttentionType::kKvCompressed,
            "global (full-attention) layers must never be compressed/bounded" );

        // Default policy (NoKvCompression) -> both layer kinds full.
        static_assert( !GemmaCuda::LocalBlockType::AttentionType::kKvCompressed );
        static_assert( !GemmaCuda::GlobalBlockType::AttentionType::kKvCompressed );

        SUCCEED();
    }

    // ====================================================================
    // J. Type identity
    // ====================================================================

    TEST_F( GemmaTransformerCudaTests, GetType_IsNetwork_GetModelType_IsGemma )
    {
        GemmaCuda net( "gemma", allLocalConfig(), Device::Cuda( 0 ) );

        // Structural kind is Network; the Gemma architecture identity is ModelType.
        EXPECT_EQ( net.getType(), ComponentType::Network );
        EXPECT_EQ( net.getModelType(), ModelType::Gemma );
    }
}
