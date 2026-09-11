/**
 * @file Qwen.Config.cpp
 * @brief QwenConfig: the published geometry, the derived widths, and validation.
 *
 * The derived widths are the part worth testing. Two of them differ by exactly the gate
 * half -- the projection emits getGatedQProjectionWidth() columns while attention consumes
 * getQProjectionWidth() of them -- and confusing the two is how `attn_output_gate` goes
 * missing, which is the defect Qwen3.8.md section 2 records as a 1.9% parameter undercount.
 *
 * CPU-only and device-free: a config allocates nothing.
 */

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

import Mila;

namespace Mila::Tests::Dnn::Components::Transformers::Qwen
{
    using namespace Mila::Dnn;

    namespace
    {
        // The published Qwen/Qwen3.8-27B values (Qwen3.8.md section 1).
        constexpr dim_t kModelDim = 5120;
        constexpr dim_t kLayers = 64;
        constexpr dim_t kHeads = 24;
        constexpr dim_t kKVHeads = 4;
        constexpr dim_t kHeadDim = 256;
        constexpr dim_t kHidden = 17408;
        constexpr dim_t kVocab = 248320;
    }

    // ====================================================================
    // Defaults are the real model, not a placeholder
    // ====================================================================

    TEST( QwenConfig, DefaultsAreThePublished27bGeometry )
    {
        QwenConfig config( kModelDim, kLayers );

        EXPECT_EQ( config.getModelDim(), kModelDim );
        EXPECT_EQ( config.getNumLayers(), kLayers );
        EXPECT_EQ( config.getVocabSize(), kVocab );
        EXPECT_EQ( config.getNumHeads(), kHeads );
        EXPECT_EQ( config.getNumKVHeads(), kKVHeads );
        EXPECT_EQ( config.getHeadDim(), kHeadDim );
        EXPECT_EQ( config.getHiddenDimension(), kHidden );
        EXPECT_EQ( config.getFullAttentionInterval(), 4 );
        EXPECT_TRUE( config.hasAttentionOutputGate() );
        EXPECT_FALSE( config.getTieWordEmbeddings() );

        // Generation reads one logit row, so the head costs one row until a scoring caller
        // asks for more.
        EXPECT_EQ( config.getLanguageModelHeadPositions(), 1 );

        EXPECT_NO_THROW( config.validate() );
    }

    TEST( QwenConfig, HeadDimIsDecoupledFromTheResidualStream )
    {
        QwenConfig config( kModelDim, kLayers );

        // 5120 / 24 = 213, not 256. A config that derived head_dim would be wrong here, and
        // the Q projection would come out 5112 wide instead of 6144.
        EXPECT_NE( config.getHeadDim(), config.getModelDim() / config.getNumHeads() );
        EXPECT_EQ( config.getQProjectionWidth(), kHeads * kHeadDim );
    }

    // ====================================================================
    // The gate half
    // ====================================================================

    TEST( QwenConfig, OutputGateDoublesTheProjectedQueryWidth )
    {
        QwenConfig config( kModelDim, kLayers );

        EXPECT_EQ( config.getQProjectionWidth(), 6144 );
        EXPECT_EQ( config.getGatedQProjectionWidth(), 12288 );
    }

    TEST( QwenConfig, PackedWidthsDifferByExactlyTheGateHalf )
    {
        QwenConfig config( kModelDim, kLayers );

        const dim_t kv = config.getKVProjectionWidth();

        EXPECT_EQ( kv, kKVHeads * kHeadDim );
        EXPECT_EQ( config.getPackedQKVWidth(), config.getGatedQProjectionWidth() + 2 * kv );
        EXPECT_EQ( config.getAttentionPackedQKVWidth(), config.getQProjectionWidth() + 2 * kv );
        EXPECT_EQ( config.getPackedQKVWidth() - config.getAttentionPackedQKVWidth(),
            config.getQProjectionWidth() );
    }

    TEST( QwenConfig, WithoutTheGateTheTwoPackedWidthsCoincide )
    {
        QwenConfig config = QwenConfig( kModelDim, kLayers ).withAttentionOutputGate( false );

        EXPECT_EQ( config.getGatedQProjectionWidth(), config.getQProjectionWidth() );
        EXPECT_EQ( config.getPackedQKVWidth(), config.getAttentionPackedQKVWidth() );
    }

    // ====================================================================
    // Partial rotary
    // ====================================================================

    TEST( QwenConfig, PartialRotaryResolvesToSixtyFourOfTwoFiftySix )
    {
        QwenConfig config( kModelDim, kLayers );

        EXPECT_FLOAT_EQ( config.getPartialRotaryFactor(), 0.25f );
        EXPECT_EQ( config.getRotaryDim(), 64 );
        EXPECT_LT( config.getRotaryDim(), config.getHeadDim() );
    }

    TEST( QwenConfig, RotaryWidthIsAlwaysEven )
    {
        // RoPE pairs dimensions, so an odd resolved width would rotate a lone element.
        QwenConfig config = QwenConfig( kModelDim, kLayers )
            .withHeadDim( 130 )
            .withPartialRotaryFactor( 0.5f );

        EXPECT_EQ( config.getRotaryDim() % 2, 0 );
    }

    // ====================================================================
    // The 3:1 interleave
    // ====================================================================

    TEST( QwenConfig, EveryFourthLayerIsFullAttentionAndTheLastOneIs )
    {
        QwenConfig config( kModelDim, kLayers );

        EXPECT_FALSE( config.isFullAttentionLayer( 0 ) );
        EXPECT_FALSE( config.isFullAttentionLayer( 1 ) );
        EXPECT_FALSE( config.isFullAttentionLayer( 2 ) );
        EXPECT_TRUE( config.isFullAttentionLayer( 3 ) );
        EXPECT_TRUE( config.isFullAttentionLayer( 7 ) );
        EXPECT_TRUE( config.isFullAttentionLayer( kLayers - 1 ) );
    }

    TEST( QwenConfig, SixteenOfSixtyFourLayersHoldAKvCache )
    {
        QwenConfig config( kModelDim, kLayers );

        // The property that makes a 27B model plausible on 12 GiB at all: a dense stack of
        // this width would cost 4x the KV (Qwen3.8.md section 3).
        EXPECT_EQ( config.getNumFullAttentionLayers(), 16 );
        EXPECT_EQ( config.getNumDeltaNetLayers(), 48 );
        EXPECT_EQ( config.getNumFullAttentionLayers() + config.getNumDeltaNetLayers(),
            config.getNumLayers() );
    }

    TEST( QwenConfig, IntervalOneMakesEveryLayerFullAttention )
    {
        QwenConfig config = QwenConfig( kModelDim, 8 ).withFullAttentionInterval( 1 );

        EXPECT_EQ( config.getNumFullAttentionLayers(), 8 );
        EXPECT_EQ( config.getNumDeltaNetLayers(), 0 );

        for ( dim_t i = 0; i < 8; ++i )
            EXPECT_TRUE( config.isFullAttentionLayer( i ) ) << "layer " << i;
    }

    // ====================================================================
    // DeltaNet geometry (carried now, consumed at Phase 3)
    // ====================================================================

    TEST( QwenConfig, CarriesTheDeltaNetGeometry )
    {
        QwenConfig config( kModelDim, kLayers );

        EXPECT_EQ( config.getLinearNumKeyHeads(), 16 );
        EXPECT_EQ( config.getLinearNumValueHeads(), 48 );
        EXPECT_EQ( config.getLinearHeadDim(), 128 );
        EXPECT_EQ( config.getLinearConvKernelDim(), 4 );
    }

    // ====================================================================
    // Validation
    // ====================================================================

    TEST( QwenConfig, RejectsNonDivisibleKvHeads )
    {
        EXPECT_THROW(
            QwenConfig( kModelDim, kLayers ).withNumHeads( 24 ).withNumKVHeads( 5 ),
            std::invalid_argument );
    }

    TEST( QwenConfig, RejectsOddHeadDim )
    {
        QwenConfig config = QwenConfig( kModelDim, kLayers ).withHeadDim( 255 );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST( QwenConfig, RejectsAnOutOfRangePartialRotaryFactor )
    {
        EXPECT_THROW( QwenConfig( kModelDim, kLayers ).withPartialRotaryFactor( 0.0f ),
            std::invalid_argument );
        EXPECT_THROW( QwenConfig( kModelDim, kLayers ).withPartialRotaryFactor( 1.5f ),
            std::invalid_argument );
    }

    TEST( QwenConfig, RejectsDeltaNetValueHeadsThatDoNotGroupOverKeyHeads )
    {
        QwenConfig config = QwenConfig( kModelDim, kLayers )
            .withLinearNumKeyHeads( 16 )
            .withLinearNumValueHeads( 40 );

        EXPECT_THROW( config.validate(), std::invalid_argument );
    }

    TEST( QwenConfig, RejectsANonPositiveLanguageModelHeadWidth )
    {
        EXPECT_THROW( QwenConfig( kModelDim, kLayers ).withLanguageModelHeadPositions( 0 ),
            std::invalid_argument );
    }

    // ====================================================================
    // Serialization round trip
    // ====================================================================

    TEST( QwenConfig, MetadataRoundTripPreservesTheGeometry )
    {
        QwenConfig source = QwenConfig( 1024, 12 )
            .withVocabularyLength( 4096 )
            .withNumHeads( 8 )
            .withNumKVHeads( 2 )
            .withHeadDim( 128 )
            .withAttentionOutputGate( true )
            .withHiddenDimension( 2048 )
            .withMaxSequenceLength( 8192 )
            .withRoPETheta( 1e7f )
            .withPartialRotaryFactor( 0.25f )
            .withFullAttentionInterval( 4 )
            .withLinearNumKeyHeads( 4 )
            .withLinearNumValueHeads( 12 )
            .withLinearHeadDim( 64 )
            .withLinearConvKernelDim( 4 );

        QwenConfig restored( 1, 1 );
        restored.fromMetadata( source.toMetadata() );

        EXPECT_EQ( restored.getModelDim(), source.getModelDim() );
        EXPECT_EQ( restored.getNumLayers(), source.getNumLayers() );
        EXPECT_EQ( restored.getVocabSize(), source.getVocabSize() );
        EXPECT_EQ( restored.getNumHeads(), source.getNumHeads() );
        EXPECT_EQ( restored.getNumKVHeads(), source.getNumKVHeads() );
        EXPECT_EQ( restored.getHeadDim(), source.getHeadDim() );
        EXPECT_EQ( restored.hasAttentionOutputGate(), source.hasAttentionOutputGate() );
        EXPECT_EQ( restored.getHiddenDimension(), source.getHiddenDimension() );
        EXPECT_EQ( restored.getMaxSequenceLength(), source.getMaxSequenceLength() );
        EXPECT_FLOAT_EQ( restored.getRoPETheta(), source.getRoPETheta() );
        EXPECT_FLOAT_EQ( restored.getPartialRotaryFactor(), source.getPartialRotaryFactor() );
        EXPECT_EQ( restored.getFullAttentionInterval(), source.getFullAttentionInterval() );
        EXPECT_EQ( restored.getLinearNumValueHeads(), source.getLinearNumValueHeads() );
        EXPECT_EQ( restored.getPackedQKVWidth(), source.getPackedQKVWidth() );

        EXPECT_NO_THROW( restored.validate() );
    }

    /**
     * @brief The head width describes the run, so a checkpoint does not carry it.
     *
     * Every other member here is published geometry and must survive the round trip. This one
     * is a buffer capacity the caller chooses, and restoring it from an artifact would let a
     * checkpoint written by a scoring run silently widen a generation build's head.
     */
    TEST( QwenConfig, MetadataDoesNotCarryTheLanguageModelHeadWidth )
    {
        QwenConfig source = QwenConfig( 1024, 12 ).withLanguageModelHeadPositions( 64 );

        QwenConfig restored( 1, 1 );
        restored.fromMetadata( source.toMetadata() );

        EXPECT_EQ( restored.getLanguageModelHeadPositions(), 1 );
    }
}
