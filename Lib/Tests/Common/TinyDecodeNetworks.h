/**
 * @file TinyDecodeNetworks.h
 * @brief Small seeded Llama, Gemma and Qwen networks whose decode runs the kernels the published models run.
 *
 * Head sizes are ones the fused decode kernel serves (128, 256, 512), so the split count and the sliding ring are
 * exercised; the parameters are initialized from a fixed seed, so every build holds the same weights. Include after
 * `import Mila;`.
 */

#pragma once

#include <cstdint>
#include <memory>

namespace Mila::Tests::Common
{
    inline constexpr unsigned int kTinyDecodeSeed = 20260929;
    inline constexpr Mila::Dnn::dim_t kTinyPrefillRows = 64;
    inline constexpr std::int32_t kTinyVocabulary = 512;

    inline Mila::Dnn::LlamaConfig tinyLlamaConfig( Mila::Dnn::dim_t context_length )
    {
        return Mila::Dnn::LlamaConfig( 512, 2 )
            .withVocabularyLength( kTinyVocabulary )
            .withNumHeads( 4 )
            .withNumKVHeads( 2 )
            .withHiddenDimension( 1024 )
            .withMaxSequenceLength( context_length )
            .withRoPETheta( 500000.0f )
            .withBias( false );
    }

    /// Layer 1 is global (pattern 2): the MQA head-size-512 path beside the sliding head-size-256 one.
    inline Mila::Dnn::GemmaConfig tinyGemmaConfig( Mila::Dnn::dim_t context_length )
    {
        return Mila::Dnn::GemmaConfig( 256, 2 )
            .withVocabularyLength( kTinyVocabulary )
            .withNumHeads( 4 )
            .withNumKVHeads( 2 )
            .withHeadDim( 256 )
            .withGlobalHeadDim( 512 )
            .withNumGlobalKVHeads( 1 )
            .withKeyEqualsValue( true )
            .withHiddenDimension( 512 )
            .withMaxSequenceLength( context_length )
            .withRMSNormEpsilon( 1e-6f )
            .withWindow( 128 )
            .withSlidingWindowPattern( 2 )
            .withGlobalRotaryDim( 128 )
            .withRoPETheta( 10000.0f )
            .withGlobalRoPETheta( 1000000.0f )
            .withFinalLogitSoftcapping( 30.0f );
    }

    /// Layers 0-2 DeltaNet, layer 3 full attention.
    inline Mila::Dnn::QwenConfig tinyQwenConfig( Mila::Dnn::dim_t context_length )
    {
        return Mila::Dnn::QwenConfig( 256, 4 )
            .withVocabularyLength( kTinyVocabulary )
            .withNumHeads( 4 )
            .withNumKVHeads( 2 )
            .withHeadDim( 128 )
            .withAttentionOutputGate( true )
            .withHiddenDimension( 512 )
            .withMaxSequenceLength( context_length )
            .withRMSNormEpsilon( 1e-6f )
            .withRoPETheta( 1e7f )
            .withPartialRotaryFactor( 0.25f )
            .withFullAttentionInterval( 4 )
            .withLinearNumKeyHeads( 2 )
            .withLinearNumValueHeads( 6 )
            .withLinearHeadDim( 32 )
            .withLinearConvKernelDim( 4 );
    }

    /// Build `TNetwork` with its parameters initialized from kTinyDecodeSeed; the global seed is restored after.
    template<typename TNetwork, typename TConfig>
    std::unique_ptr<TNetwork> buildTinyNetwork( const TConfig& config, Mila::Dnn::dim_t context_length )
    {
        using namespace Mila::Dnn;

        auto& generator = Core::RandomGenerator::getInstance();
        const unsigned int previous = generator.getSeed();
        generator.setSeed( kTinyDecodeSeed );

        auto network = std::make_unique<TNetwork>( "tiny", config, Compute::Device::Cuda( 0 ) );
        network->build( BuildContext( shape_t{ 1, context_length }, RuntimeMode::Inference, true )
            .withPrefillSize( kTinyPrefillRows ) );
        network->synchronize();

        generator.setSeed( previous );

        return network;
    }
}
