// Unit tests for LlamaConfig: constructor, fluent setters, getters, validate, toString,
// and toMetadata/fromMetadata round-trip.

#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

import Mila;

namespace Dnn::Components::Transformers::Tests
{
    using namespace Mila::Dnn;

    // ---- Constructor --------------------------------------------------------

    TEST( LlamaConfigTests, Constructor_ValidArgs_SetsEmbeddingDimAndNumLayers )
    {
        LlamaConfig cfg( 64, 4 );

        EXPECT_EQ( cfg.getModelDim(), 64 );
        EXPECT_EQ( cfg.getNumLayers(), 4 );
    }

    TEST( LlamaConfigTests, Constructor_ZeroEmbeddingDim_Throws )
    {
        EXPECT_THROW( LlamaConfig( 0, 2 ), std::invalid_argument );
    }

    TEST( LlamaConfigTests, Constructor_ZeroNumLayers_Throws )
    {
        EXPECT_THROW( LlamaConfig( 64, 0 ), std::invalid_argument );
    }

    // ---- withVocabularyLength -----------------------------------------------

    TEST( LlamaConfigTests, WithVocabularyLength_SetsVocabSize )
    {
        LlamaConfig cfg( 64, 2 );
        cfg.withVocabularyLength( 512 );

        EXPECT_EQ( cfg.getVocabSize(), 512 );
    }

    TEST( LlamaConfigTests, WithVocabularyLength_ZeroValue_Throws )
    {
        LlamaConfig cfg( 64, 2 );

        EXPECT_THROW( cfg.withVocabularyLength( 0 ), std::invalid_argument );
    }

    // ---- withNumHeads -------------------------------------------------------

    TEST( LlamaConfigTests, WithNumHeads_SetsValue )
    {
        LlamaConfig cfg( 64, 2 );
        cfg.withNumHeads( 4 );

        EXPECT_EQ( cfg.getNumHeads(), 4 );
    }

    TEST( LlamaConfigTests, WithNumHeads_ZeroValue_Throws )
    {
        LlamaConfig cfg( 64, 2 );

        EXPECT_THROW( cfg.withNumHeads( 0 ), std::invalid_argument );
    }

    TEST( LlamaConfigTests, WithNumHeads_NotDivisorOfEmbeddingDim_Throws )
    {
        // embedding_dim=64 is not divisible by 6
        LlamaConfig cfg( 64, 2 );

        EXPECT_THROW( cfg.withNumHeads( 6 ), std::invalid_argument );
    }

    // ---- withNumKVHeads -----------------------------------------------------

    TEST( LlamaConfigTests, WithNumKVHeads_SetsValue )
    {
        LlamaConfig cfg( 64, 2 );
        cfg.withNumHeads( 4 ).withNumKVHeads( 2 );

        EXPECT_EQ( cfg.getNumKVHeads(), 2 );
    }

    TEST( LlamaConfigTests, WithNumKVHeads_NotDivisorOfNumHeads_Throws )
    {
        // 4 heads, 3 KV heads: 4 % 3 != 0
        EXPECT_THROW(
            LlamaConfig( 64, 2 ).withNumHeads( 4 ).withNumKVHeads( 3 ),
            std::invalid_argument );
    }

    TEST( LlamaConfigTests, GetNumKVHeads_WhenSetToZero_FallsBackToNumHeads )
    {
        // withNumKVHeads(0) bypasses the divisibility guard; getNumKVHeads() should
        // return num_heads_ as the fallback.
        LlamaConfig cfg( 64, 2 );
        cfg.withNumHeads( 4 ).withNumKVHeads( 0 );

        EXPECT_EQ( cfg.getNumKVHeads(), 4 );
    }

    // ---- withMaxSequenceLength ----------------------------------------------

    TEST( LlamaConfigTests, WithMaxSequenceLength_SetsValue )
    {
        LlamaConfig cfg( 64, 2 );
        cfg.withMaxSequenceLength( 128 );

        EXPECT_EQ( cfg.getMaxSequenceLength(), 128 );
    }

    TEST( LlamaConfigTests, WithMaxSequenceLength_ZeroValue_Throws )
    {
        LlamaConfig cfg( 64, 2 );

        EXPECT_THROW( cfg.withMaxSequenceLength( 0 ), std::invalid_argument );
    }

    // ---- withHiddenDimension ------------------------------------------------

    TEST( LlamaConfigTests, WithHiddenDimension_SetsValue )
    {
        LlamaConfig cfg( 64, 2 );
        cfg.withHiddenDimension( 256 );

        EXPECT_EQ( cfg.getHiddenDimension(), 256 );
    }

    // ---- withRoPETheta ------------------------------------------------------

    TEST( LlamaConfigTests, WithRoPETheta_SetsValue )
    {
        LlamaConfig cfg( 64, 2 );
        cfg.withRoPETheta( 10000.0f );

        EXPECT_FLOAT_EQ( cfg.getRoPETheta(), 10000.0f );
    }

    TEST( LlamaConfigTests, WithRoPETheta_ZeroValue_Throws )
    {
        LlamaConfig cfg( 64, 2 );

        EXPECT_THROW( cfg.withRoPETheta( 0.0f ), std::invalid_argument );
    }

    TEST( LlamaConfigTests, WithRoPETheta_NegativeValue_Throws )
    {
        LlamaConfig cfg( 64, 2 );

        EXPECT_THROW( cfg.withRoPETheta( -1.0f ), std::invalid_argument );
    }

    // ---- withRoPEFrequencyScaling -------------------------------------------

    TEST( LlamaConfigTests, WithRoPEFrequencyScaling_DefaultIsNone )
    {
        LlamaConfig cfg( 64, 2 );

        EXPECT_FALSE( cfg.getRoPEFrequencyScaling().isScaled() );
    }

    TEST( LlamaConfigTests, WithRoPEFrequencyScaling_SetsValue )
    {
        const RopeFrequencyScaling scaling{ 8.0f, 1.0f, 4.0f, 8192 };

        LlamaConfig cfg( 64, 2 );
        cfg.withRoPEFrequencyScaling( scaling );

        EXPECT_EQ( cfg.getRoPEFrequencyScaling(), scaling );
    }

    TEST( LlamaConfigTests, WithRoPEFrequencyScaling_ZeroFactor_Throws )
    {
        LlamaConfig cfg( 64, 2 );

        EXPECT_THROW( cfg.withRoPEFrequencyScaling( RopeFrequencyScaling{ 0.0f, 1.0f, 4.0f, 8192 } ), std::invalid_argument );
    }

    // The interpolation divides by high - low, so the band must not be empty.
    TEST( LlamaConfigTests, WithRoPEFrequencyScaling_EmptyBand_Throws )
    {
        LlamaConfig cfg( 64, 2 );

        EXPECT_THROW( cfg.withRoPEFrequencyScaling( RopeFrequencyScaling{ 8.0f, 4.0f, 4.0f, 8192 } ), std::invalid_argument );
    }

    // ---- withBias -----------------------------------------------------------

    TEST( LlamaConfigTests, WithBias_DefaultIsFalse )
    {
        LlamaConfig cfg( 64, 2 );

        EXPECT_FALSE( cfg.useBias() );
    }

    TEST( LlamaConfigTests, WithBias_SetTrue_ReturnsTrue )
    {
        LlamaConfig cfg( 64, 2 );
        cfg.withBias( true );

        EXPECT_TRUE( cfg.useBias() );
    }

    TEST( LlamaConfigTests, WithBias_SetFalse_ReturnsFalse )
    {
        LlamaConfig cfg( 64, 2 );
        cfg.withBias( true ).withBias( false );

        EXPECT_FALSE( cfg.useBias() );
    }

    // ---- withLogLikelihoodWindow --------------------------------------------

    TEST( LlamaConfigTests, WithLogLikelihoodWindow_DefaultIsOne )
    {
        LlamaConfig cfg( 64, 2 );

        EXPECT_EQ( cfg.getLogLikelihoodWindow(), 1 );
    }

    TEST( LlamaConfigTests, WithLogLikelihoodWindow_ZeroValue_Throws )
    {
        EXPECT_THROW( LlamaConfig( 64, 2 ).withLogLikelihoodWindow( 0 ), std::invalid_argument );
    }

    // A run capacity, not geometry of the checkpoint, so a restored config falls back to the default.
    TEST( LlamaConfigTests, WithLogLikelihoodWindow_AbsentFromMetadata )
    {
        const LlamaConfig source = LlamaConfig( 64, 2 ).withLogLikelihoodWindow( 64 );

        LlamaConfig restored( 64, 2 );
        restored.fromMetadata( source.toMetadata() );

        EXPECT_EQ( restored.getLogLikelihoodWindow(), 1 );
    }

    // ---- Getters / defaults -------------------------------------------------

    TEST( LlamaConfigTests, GetRMSNormEpsilon_ReturnsDefault )
    {
        LlamaConfig cfg( 64, 2 );

        EXPECT_FLOAT_EQ( cfg.getRMSNormEpsilon(), 1e-5f );
    }

    // ---- Fluent chain -------------------------------------------------------

    TEST( LlamaConfigTests, FluentChain_AllSetters_ProducesCorrectConfig )
    {
        auto cfg = LlamaConfig( 64, 2 )
            .withVocabularyLength( 128 )
            .withNumHeads( 4 )
            .withNumKVHeads( 2 )
            .withHiddenDimension( 256 )
            .withMaxSequenceLength( 32 )
            .withRoPETheta( 10000.0f )
            .withRoPEFrequencyScaling( RopeFrequencyScaling{ 8.0f, 1.0f, 4.0f, 16 } )
            .withBias( false );

        EXPECT_EQ( cfg.getVocabSize(), 128 );
        EXPECT_EQ( cfg.getModelDim(), 64 );
        EXPECT_EQ( cfg.getNumLayers(), 2 );
        EXPECT_EQ( cfg.getNumHeads(), 4 );
        EXPECT_EQ( cfg.getNumKVHeads(), 2 );
        EXPECT_EQ( cfg.getHiddenDimension(), 256 );
        EXPECT_EQ( cfg.getMaxSequenceLength(), 32 );
        EXPECT_FLOAT_EQ( cfg.getRoPETheta(), 10000.0f );
        EXPECT_EQ( cfg.getRoPEFrequencyScaling().original_context_length, 16 );
        EXPECT_FALSE( cfg.useBias() );
    }

    // ---- validate -----------------------------------------------------------

    TEST( LlamaConfigTests, Validate_FullyConfigured_DoesNotThrow )
    {
        auto cfg = LlamaConfig( 64, 2 )
            .withVocabularyLength( 128 )
            .withNumHeads( 4 )
            .withNumKVHeads( 2 )
            .withHiddenDimension( 256 )
            .withMaxSequenceLength( 32 )
            .withRoPETheta( 10000.0f )
            .withRoPEFrequencyScaling( RopeFrequencyScaling{ 8.0f, 1.0f, 4.0f, 16 } );

        EXPECT_NO_THROW( cfg.validate() );
    }

    TEST( LlamaConfigTests, Validate_DefaultConstructed_DoesNotThrow )
    {
        // Default field values satisfy all validate() constraints.
        LlamaConfig cfg( 64, 2 );

        EXPECT_NO_THROW( cfg.validate() );
    }

    // ---- toString -----------------------------------------------------------

    TEST( LlamaConfigTests, ToString_ContainsExpectedFields )
    {
        auto cfg = LlamaConfig( 64, 2 )
            .withVocabularyLength( 128 )
            .withNumHeads( 4 )
            .withNumKVHeads( 2 )
            .withMaxSequenceLength( 32 )
            .withRoPETheta( 10000.0f );

        std::string s = cfg.toString();

        EXPECT_NE( s.find( "Vocab Size" ), std::string::npos );
        EXPECT_NE( s.find( "Num Layers" ), std::string::npos );
        EXPECT_NE( s.find( "Embedding Dim" ), std::string::npos );
        EXPECT_NE( s.find( "Num Heads" ), std::string::npos );
        EXPECT_NE( s.find( "Num KV Heads" ), std::string::npos );
        EXPECT_NE( s.find( "RoPE" ), std::string::npos );
        EXPECT_NE( s.find( "RMSNorm" ), std::string::npos );
        EXPECT_NE( s.find( "Use Bias" ), std::string::npos );
    }

    // ---- Metadata round-trip ------------------------------------------------

    TEST( LlamaConfigTests, MetadataRoundTrip_PreservesAllFields )
    {
        auto original = LlamaConfig( 128, 4 )
            .withVocabularyLength( 512 )
            .withNumHeads( 8 )
            .withNumKVHeads( 4 )
            .withHiddenDimension( 512 )
            .withMaxSequenceLength( 64 )
            .withRoPETheta( 10000.0f )
            .withRoPEFrequencyScaling( RopeFrequencyScaling{ 8.0f, 1.0f, 4.0f, 8192 } )
            .withBias( true );

        auto meta = original.toMetadata();

        // Deserialize into a separately-constructed config.
        LlamaConfig restored( 1, 1 );
        restored.fromMetadata( meta );

        EXPECT_EQ( restored.getVocabSize(), original.getVocabSize() );
        EXPECT_EQ( restored.getNumLayers(), original.getNumLayers() );
        EXPECT_EQ( restored.getModelDim(), original.getModelDim() );
        EXPECT_EQ( restored.getNumHeads(), original.getNumHeads() );
        EXPECT_EQ( restored.getNumKVHeads(), original.getNumKVHeads() );
        EXPECT_EQ( restored.getHiddenDimension(), original.getHiddenDimension() );
        EXPECT_EQ( restored.getMaxSequenceLength(), original.getMaxSequenceLength() );
        EXPECT_FLOAT_EQ( restored.getRoPETheta(), original.getRoPETheta() );
        EXPECT_EQ( restored.getRoPEFrequencyScaling(), original.getRoPEFrequencyScaling() );
        EXPECT_EQ( restored.useBias(), original.useBias() );
    }
}