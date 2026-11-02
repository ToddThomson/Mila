// Unit tests for GemmaConfig: the Step 0 head_dim decoupling (Gemma.md sections 2-3).
// The central contract: head_dim is independent of embedding_dim / num_heads, the
// derived Q-projection width may differ from the residual stream, and validate()
// accepts that mismatch (where LlamaConfig would bake in head_dim == embedding/heads).

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <cmath>

import Mila;

namespace Dnn::Components::Transformers::Tests
{
    using namespace Mila::Dnn;

    // ---- Constructor --------------------------------------------------------

    TEST( GemmaConfigTests, Constructor_ValidArgs_SetsEmbeddingDimAndNumLayers )
    {
        GemmaConfig cfg( 3840, 48 );

        EXPECT_EQ( cfg.getModelDim(), 3840 );
        EXPECT_EQ( cfg.getNumLayers(), 48 );
    }

    TEST( GemmaConfigTests, Constructor_ZeroEmbeddingDim_Throws )
    {
        EXPECT_THROW( GemmaConfig( 0, 48 ), std::invalid_argument );
    }

    TEST( GemmaConfigTests, Constructor_ZeroNumLayers_Throws )
    {
        EXPECT_THROW( GemmaConfig( 3840, 0 ), std::invalid_argument );
    }

    // ---- head_dim decoupling (the Step 0 contract) --------------------------

    TEST( GemmaConfigTests, HeadDim_Explicit_DecoupledFromResidualStream )
    {
        // Gemma 4 12B sliding geometry: 3840 / 16 = 240, but head_dim is 256.
        auto cfg = GemmaConfig( 3840, 48 )
            .withNumHeads( 16 )
            .withNumKVHeads( 8 )
            .withHeadDim( 256 );

        EXPECT_EQ( cfg.getHeadDim(), 256 );          // explicit, NOT 240
        EXPECT_NE( cfg.getHeadDim(), 3840 / 16 );    // decoupled from the residual derivation
    }

    TEST( GemmaConfigTests, HeadDim_ExplicitZero_FallsBackToResidualDerivation )
    {
        // head_dim defaults to 256 (Gemma 4 12B); an explicit 0 selects the
        // embedding_dim / num_heads derivation.
        auto cfg = GemmaConfig( 64, 2 ).withNumHeads( 4 ).withHeadDim( 0 );

        EXPECT_EQ( cfg.getHeadDim(), 16 );           // 64 / 4
    }

    // ---- Derived geometry ---------------------------------------------------

    TEST( GemmaConfigTests, SlidingGeometry_DerivedWidths )
    {
        auto cfg = GemmaConfig( 3840, 48 )
            .withNumHeads( 16 )
            .withNumKVHeads( 8 )
            .withHeadDim( 256 );

        EXPECT_EQ( cfg.getQProjectionWidth(), 4096 );    // 16 * 256, != residual 3840
        EXPECT_EQ( cfg.getKVProjectionWidth(), 2048 );   // 8 * 256
        EXPECT_EQ( cfg.getPackedQKVWidth(), 8192 );      // (16 + 2*8) * 256
        EXPECT_NE( cfg.getQProjectionWidth(), cfg.getModelDim() ); // o_proj is non-square
    }

    TEST( GemmaConfigTests, GlobalGeometry_DerivedWidths )
    {
        // Global-layer head_dim 512 with a single KV head (the standard packing;
        // the K=V global variant arrives in Step 1).
        auto cfg = GemmaConfig( 3840, 48 )
            .withNumHeads( 16 )
            .withNumKVHeads( 1 )
            .withHeadDim( 512 );

        EXPECT_EQ( cfg.getHeadDim(), 512 );
        EXPECT_EQ( cfg.getQProjectionWidth(), 8192 );    // 16 * 512
        EXPECT_EQ( cfg.getKVProjectionWidth(), 512 );    // 1 * 512
        EXPECT_EQ( cfg.getPackedQKVWidth(), 9216 );      // (16 + 2*1) * 512
    }

    // ---- validate -----------------------------------------------------------

    TEST( GemmaConfigTests, Validate_QWidthNotEqualResidual_DoesNotThrow )
    {
        // The critical assertion: validate() accepts num_heads * head_dim != embedding_dim,
        // where LlamaConfig bakes in head_dim == embedding_dim / num_heads.
        auto cfg = GemmaConfig( 3840, 48 )
            .withVocabularyLength( 262144 )
            .withNumHeads( 16 )
            .withNumKVHeads( 8 )
            .withHeadDim( 256 );

        EXPECT_NO_THROW( cfg.validate() );
    }

    TEST( GemmaConfigTests, Validate_OddHeadDim_Throws )
    {
        auto cfg = GemmaConfig( 3840, 48 )
            .withNumHeads( 16 )
            .withHeadDim( 255 );

        EXPECT_THROW( cfg.validate(), std::invalid_argument );
    }

    TEST( GemmaConfigTests, WithNumKVHeads_NotDivisorOfNumHeads_Throws )
    {
        EXPECT_THROW(
            GemmaConfig( 3840, 48 ).withNumHeads( 16 ).withNumKVHeads( 6 ),
            std::invalid_argument );
    }

    TEST( GemmaConfigTests, GetNumKVHeads_WhenSetToZero_FallsBackToNumHeads )
    {
        // The default num_kv_heads is 8 (Gemma 4 12B). Only an explicit 0 bypasses
        // the divisibility guard and falls back to num_heads (matches LlamaConfig).
        auto cfg = GemmaConfig( 3840, 48 ).withNumHeads( 16 ).withNumKVHeads( 0 );

        EXPECT_EQ( cfg.getNumKVHeads(), 16 );
    }

    // ---- Defaults -----------------------------------------------------------

    TEST( GemmaConfigTests, Defaults_MatchGemma4_12B )
    {
        GemmaConfig cfg( 3840, 48 );

        EXPECT_EQ( cfg.getVocabSize(), 262144 );
        EXPECT_EQ( cfg.getNumHeads(), 16 );
        EXPECT_EQ( cfg.getNumKVHeads(), 8 );
        EXPECT_EQ( cfg.getHeadDim(), 256 );           // explicit default, NOT the 240 derivation
        EXPECT_EQ( cfg.getHiddenDimension(), 15360 );
        EXPECT_EQ( cfg.getMaxSequenceLength(), 262144 );
        EXPECT_FLOAT_EQ( cfg.getRMSNormEpsilon(), 1e-6f );

        // Global-layer geometry defaults.
        EXPECT_EQ( cfg.getGlobalHeadDim(), 512 );
        EXPECT_EQ( cfg.getNumGlobalKVHeads(), 1 );
        EXPECT_TRUE( cfg.keyEqualsValue() );

        // Chassis defaults (Step 5a).
        EXPECT_EQ( cfg.getWindow(), 1024 );
        EXPECT_EQ( cfg.getSlidingWindowPattern(), 6 );
        EXPECT_EQ( cfg.getGlobalRotaryDim(), 128 );
        EXPECT_FLOAT_EQ( cfg.getRoPEThetaLocal(), 10000.0f );
        EXPECT_FLOAT_EQ( cfg.getRoPEThetaGlobal(), 1000000.0f );
        EXPECT_FLOAT_EQ( cfg.getFinalLogitSoftcapping(), 30.0f );
        EXPECT_FLOAT_EQ( cfg.getEmbeddingScale(), std::sqrt( 3840.0f ) );
    }

    // ---- Per-layer interleave (Step 5a) -------------------------------------

    TEST( GemmaConfigTests, IsGlobalLayer_FiveToOnePattern )
    {
        GemmaConfig cfg( 3840, 48 );   // sliding_window_pattern 6

        EXPECT_FALSE( cfg.isGlobalLayer( 0 ) );
        EXPECT_FALSE( cfg.isGlobalLayer( 4 ) );
        EXPECT_TRUE( cfg.isGlobalLayer( 5 ) );    // first global
        EXPECT_FALSE( cfg.isGlobalLayer( 6 ) );
        EXPECT_TRUE( cfg.isGlobalLayer( 11 ) );
        EXPECT_TRUE( cfg.isGlobalLayer( 47 ) );   // final layer is global
    }

    TEST( GemmaConfigTests, PerLayerGeometry_SlidingLayer )
    {
        GemmaConfig cfg( 3840, 48 );
        const dim_t layer = 0;   // sliding

        EXPECT_EQ( cfg.getHeadDimForLayer( layer ), 256 );
        EXPECT_EQ( cfg.getNumKVHeadsForLayer( layer ), 8 );
        EXPECT_FALSE( cfg.keyEqualsValueForLayer( layer ) );
        EXPECT_EQ( cfg.getWindowForLayer( layer ), 1024 );
        EXPECT_FLOAT_EQ( cfg.getRoPEThetaForLayer( layer ), 10000.0f );
        EXPECT_EQ( cfg.getRotaryDimForLayer( layer ), 0 );           // full rotation
        EXPECT_EQ( cfg.getQProjectionWidthForLayer( layer ), 4096 ); // 16 * 256
        EXPECT_EQ( cfg.getPackedQKVWidthForLayer( layer ), 8192 );   // (16 + 2*8) * 256
    }

    TEST( GemmaConfigTests, PerLayerGeometry_GlobalLayer )
    {
        GemmaConfig cfg( 3840, 48 );
        const dim_t layer = 5;   // global

        EXPECT_EQ( cfg.getHeadDimForLayer( layer ), 512 );
        EXPECT_EQ( cfg.getNumKVHeadsForLayer( layer ), 1 );
        EXPECT_TRUE( cfg.keyEqualsValueForLayer( layer ) );
        EXPECT_EQ( cfg.getWindowForLayer( layer ), 0 );              // global = unbounded
        EXPECT_FLOAT_EQ( cfg.getRoPEThetaForLayer( layer ), 1000000.0f );
        EXPECT_EQ( cfg.getRotaryDimForLayer( layer ), 128 );        // proportional partial-rotary
        EXPECT_EQ( cfg.getQProjectionWidthForLayer( layer ), 8192 ); // 16 * 512
        EXPECT_EQ( cfg.getPackedQKVWidthForLayer( layer ), 8704 );   // (16 + 1) * 512, K=V
    }

    // ---- Global-layer geometry (Step 1) -------------------------------------

    TEST( GemmaConfigTests, GlobalGeometry_DerivedWidths_KeyEqualsValue )
    {
        // Gemma 4 12B global layer: head_dim 512, single KV head, K=V.
        GemmaConfig cfg( 3840, 48 );   // defaults are the global geometry

        EXPECT_EQ( cfg.getGlobalQProjectionWidth(), 8192 );     // 16 * 512
        EXPECT_EQ( cfg.getGlobalKVProjectionWidth(), 512 );     // 1 * 512
        EXPECT_EQ( cfg.getGlobalPackedQKVWidth(), 8704 );       // (16 + 1*1) * 512, K=V drops the V section
    }

    TEST( GemmaConfigTests, GlobalGeometry_PackedWidth_WithoutKeyEqualsValue )
    {
        auto cfg = GemmaConfig( 3840, 48 ).withKeyEqualsValue( false );

        EXPECT_EQ( cfg.getGlobalPackedQKVWidth(), 9216 );       // (16 + 2*1) * 512, separate V section
    }

    TEST( GemmaConfigTests, GlobalGeometry_Unset_FallsBackToSlidingGeometry )
    {
        auto cfg = GemmaConfig( 3840, 48 )
            .withNumHeads( 16 )
            .withNumKVHeads( 8 )
            .withHeadDim( 256 )
            .withGlobalHeadDim( 0 )
            .withNumGlobalKVHeads( 0 );

        EXPECT_EQ( cfg.getGlobalHeadDim(), 256 );               // falls back to head_dim
        EXPECT_EQ( cfg.getNumGlobalKVHeads(), 8 );              // falls back to num_kv_heads
    }

    TEST( GemmaConfigTests, WithNumGlobalKVHeads_NotDivisorOfNumHeads_Throws )
    {
        EXPECT_THROW(
            GemmaConfig( 3840, 48 ).withNumHeads( 16 ).withNumGlobalKVHeads( 6 ),
            std::invalid_argument );
    }

    TEST( GemmaConfigTests, Validate_OddGlobalHeadDim_Throws )
    {
        auto cfg = GemmaConfig( 3840, 48 )
            .withNumHeads( 16 )
            .withGlobalHeadDim( 511 );

        EXPECT_THROW( cfg.validate(), std::invalid_argument );
    }

    // ---- toString -----------------------------------------------------------

    TEST( GemmaConfigTests, ToString_ContainsDecoupledGeometry )
    {
        auto cfg = GemmaConfig( 3840, 48 )
            .withNumHeads( 16 )
            .withNumKVHeads( 8 )
            .withHeadDim( 256 );

        std::string s = cfg.toString();

        EXPECT_NE( s.find( "Head Dim" ), std::string::npos );
        EXPECT_NE( s.find( "Q Projection Width" ), std::string::npos );
        EXPECT_NE( s.find( "Embedding Dim" ), std::string::npos );
    }

    // ---- Metadata round-trip ------------------------------------------------

    TEST( GemmaConfigTests, MetadataRoundTrip_PreservesDecoupledHeadDim )
    {
        auto original = GemmaConfig( 3840, 48 )
            .withVocabularyLength( 262144 )
            .withNumHeads( 16 )
            .withNumKVHeads( 8 )
            .withHeadDim( 256 )
            .withGlobalHeadDim( 512 )
            .withNumGlobalKVHeads( 1 )
            .withKeyEqualsValue( true )
            .withHiddenDimension( 15360 )
            .withMaxSequenceLength( 262144 )
            .withWindow( 1024 )
            .withSlidingWindowPattern( 6 )
            .withGlobalRotaryDim( 128 )
            .withRoPETheta( 10000.0f )
            .withGlobalRoPETheta( 1000000.0f )
            .withFinalLogitSoftcapping( 30.0f );

        auto meta = original.toMetadata();

        GemmaConfig restored( 1, 1 );
        restored.fromMetadata( meta );

        EXPECT_EQ( restored.getModelDim(), original.getModelDim() );
        EXPECT_EQ( restored.getNumHeads(), original.getNumHeads() );
        EXPECT_EQ( restored.getNumKVHeads(), original.getNumKVHeads() );
        EXPECT_EQ( restored.getHeadDim(), original.getHeadDim() );          // 256, not re-derived to 240
        EXPECT_EQ( restored.getQProjectionWidth(), original.getQProjectionWidth() );
        EXPECT_EQ( restored.getGlobalHeadDim(), original.getGlobalHeadDim() );
        EXPECT_EQ( restored.getNumGlobalKVHeads(), original.getNumGlobalKVHeads() );
        EXPECT_EQ( restored.keyEqualsValue(), original.keyEqualsValue() );
        EXPECT_EQ( restored.getGlobalPackedQKVWidth(), original.getGlobalPackedQKVWidth() );
        EXPECT_EQ( restored.getHiddenDimension(), original.getHiddenDimension() );
        EXPECT_EQ( restored.getMaxSequenceLength(), original.getMaxSequenceLength() );
        EXPECT_EQ( restored.getWindow(), original.getWindow() );
        EXPECT_EQ( restored.getSlidingWindowPattern(), original.getSlidingWindowPattern() );
        EXPECT_EQ( restored.getGlobalRotaryDim(), original.getGlobalRotaryDim() );
        EXPECT_FLOAT_EQ( restored.getRoPEThetaLocal(), original.getRoPEThetaLocal() );
        EXPECT_FLOAT_EQ( restored.getRoPEThetaGlobal(), original.getRoPEThetaGlobal() );
        EXPECT_FLOAT_EQ( restored.getFinalLogitSoftcapping(), original.getFinalLogitSoftcapping() );
    }
}
