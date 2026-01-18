/**
 * @file TransformerPresets.ixx
 * @brief Preset configurations for well-known transformer architectures.
 *
 * Provides factory functions that return pre-configured TransformerConfig
 * instances matching published model architectures (GPT-2, Llama, etc.).
 */

module;
#include <cstdint>

export module Dnn.Blocks.Transformer:Presets;

import :Config;
import Dnn.ActivationType;
import Dnn.TensorTypes;

namespace Mila::Dnn
{
    using Mila::Dnn::TransformerConfig;
    using Mila::Dnn::ActivationType;
    using Mila::Dnn::dim_t;

    /**
     * Usage Examples:
     *
     * // Use a preset directly
     * auto config = Mila::Dnn::GPT2_Small();
     * auto transformer = Transformer( config );
     *
     * // Customize a preset
     * auto config = Mila::Dnn::Llama3_2_1B()
     *     .withResidualScale( 0.5f );  // Custom residual scaling
     *
     * // Mix and match for research
     * auto hybrid = Mila::Dnn::Llama3_8B()
     *     .withBias( true )  // Add bias (normally false for Llama)
     *     .withActivation( ActivationType::Gelu );  // Try GELU instead of SwiGLU
     *
     * // Advanced: Component-level control
     * // (Your HAL layer handles norm type, attention type, positional encoding)
     * auto config = Mila::Dnn::Presets::Llama3_8B();
     * auto transformer = Transformer::Builder()
     *     .withConfig( config )
     *     .withNormType( NormType::RMSNorm )
     *     .withAttentionType( AttentionType::GroupedQuery, 8 )  // 8 KV heads
     *     .withEncoding( PositionalType::RoPE, 500000.0f )
     *     .build();
     */

    // ========================================================================
    // GPT-2 Preset Family
    // ========================================================================

    /**
     * @brief GPT-2 Small (117M parameters)
     *
     * Architecture:
     * - 768 embedding dim, 12 layers, 12 heads
     * - Standard multi-head attention
     * - LayerNorm + GELU activation
     * - 3072 hidden dim (4x embedding)
     * - Bias enabled
     */
    export TransformerConfig GPT2_Small()
    {
        return TransformerConfig( 768, 12 )
            .withHiddenDimension( 3072 )
            .withBias( true )
            .withActivation( ActivationType::Gelu );
    }

    /**
     * @brief GPT-2 Medium (345M parameters)
     */
    export TransformerConfig GPT2_Medium()
    {
        return TransformerConfig( 1024, 16 )
            .withHiddenDimension( 4096 )
            .withBias( true )
            .withActivation( ActivationType::Gelu );
    }

    /**
     * @brief GPT-2 Large (774M parameters)
     */
    export TransformerConfig GPT2_Large()
    {
        return TransformerConfig( 1280, 20 )
            .withHiddenDimension( 5120 )
            .withBias( true )
            .withActivation( ActivationType::Gelu );
    }

    /**
     * @brief GPT-2 XL (1.5B parameters)
     */
    export TransformerConfig GPT2_XL()
    {
        return TransformerConfig( 1600, 25 )
            .withHiddenDimension( 6400 )
            .withBias( true )
            .withActivation( ActivationType::Gelu );
    }

    // ========================================================================
    // Llama 3 Preset Family
    // ========================================================================

    /**
     * @brief Llama 3 8B
     *
     * Architecture:
     * - 4096 embedding dim, 32 layers, 32 heads
     * - Grouped Query Attention (8 KV heads)
     * - RMSNorm + SwiGLU activation
     * - 14336 hidden dim (~3.5x for SwiGLU)
     * - No bias
     * - RoPE positional encoding (theta=500000)
     */
    export TransformerConfig Llama3_8B()
    {
        return TransformerConfig( 4096, 32 )
            .withHiddenDimension( 14336 )
            .withBias( false )
            .withActivation( ActivationType::Swiglu )
            .withNormType( NormType::RMSNorm )
            .withAttentionType( AttentionType::GroupedQuery )
            .withKVHeads( 8 )
            .withEncoding( EncodingType::RoPE )
            .withRoPETheta( 500000.0f )
            .withMaxSequenceLength( 8192 );
    }

    /**
     * @brief Llama 3 70B
     *
     * Architecture:
     * - 8192 embedding dim, 80 layers, 64 heads
     * - GQA with 8 KV heads
     * - RMSNorm + SwiGLU
     * - 28672 hidden dim
     */
    export TransformerConfig Llama3_70B()
    {
        return TransformerConfig( 8192, 64 )
            .withHiddenDimension( 28672 )
            .withBias( false )
            .withActivation( ActivationType::Swiglu )
            .withNormType( NormType::RMSNorm )
            .withAttentionType( AttentionType::GroupedQuery )
            .withKVHeads( 8 )
            .withEncoding( EncodingType::RoPE )
            .withRoPETheta( 500000.0f )
            .withMaxSequenceLength( 8192 );
    }

    // ========================================================================
    // Llama 3.1 Preset Family (128K context variants)
    // ========================================================================

    /**
     * @brief Llama 3.1 8B (128K context)
     *
     * Same architecture as Llama 3 8B but trained for extended context.
     * Uses RoPE scaling factor of 8.0 for long context support.
     */
    export TransformerConfig Llama3_1_8B()
    {
        return TransformerConfig( 4096, 32 )
            .withHiddenDimension( 14336 )
            .withBias( false )
            .withActivation( ActivationType::Swiglu );
        // Note: RoPE scaling handled by positional encoding component
    }

    /**
     * @brief Llama 3.1 70B (128K context)
     */
    export TransformerConfig Llama3_1_70B()
    {
        return TransformerConfig( 8192, 64 )
            .withHiddenDimension( 28672 )
            .withBias( false )
            .withActivation( ActivationType::Swiglu );
    }

    /**
     * @brief Llama 3.1 405B (128K context)
     *
     * Architecture:
     * - 16384 embedding dim, 126 layers, 128 heads
     * - GQA with 8 KV heads (16:1 ratio!)
     * - 53248 hidden dim
     */
    export TransformerConfig Llama3_1_405B()
    {
        return TransformerConfig( 16384, 128 )
            .withHiddenDimension( 53248 )
            .withBias( false )
            .withActivation( ActivationType::Swiglu );
    }

    // ========================================================================
    // Llama 3.2 Preset Family (Efficient small models)
    // ========================================================================

    /**
     * @brief Llama 3.2 1B
     *
     * Architecture:
     * - 2048 embedding dim, 16 layers, 32 heads
     * - GQA with 8 KV heads
     * - 8192 hidden dim (4x)
     * - RoPE scaling factor 32.0 for 128K context
     *
     * Excellent for fine-tuning on limited hardware (fits in 12GB).
     */
    export TransformerConfig Llama3_2_1B()
    {
        return TransformerConfig( 2048, 32 )
            .withHiddenDimension( 8192 )
            .withBias( false )
            .withActivation( ActivationType::Swiglu );
    }

    /**
     * @brief Llama 3.2 3B
     *
     * Architecture:
     * - 3072 embedding dim, 28 layers, 24 heads
     * - GQA with 8 KV heads
     * - 8192 hidden dim
     */
    export TransformerConfig Llama3_2_3B()
    {
        return TransformerConfig( 3072, 24 )
            .withHiddenDimension( 8192 )
            .withBias( false )
            .withActivation( ActivationType::Swiglu );
    }

    // ========================================================================
    // Additional Architecture Presets
    // ========================================================================

    /**
     * @brief Mistral 7B v0.1
     *
     * Architecture similar to Llama but with sliding window attention.
     * - 4096 embedding dim, 32 layers, 32 heads
     * - GQA with 8 KV heads
     * - SwiGLU activation
     * - Sliding window of 4096 tokens
     */
    export TransformerConfig Mistral_7B()
    {
        return TransformerConfig( 4096, 32 )
            .withHiddenDimension( 14336 )
            .withBias( false )
            .withActivation( ActivationType::Swiglu );
        // Note: Sliding window attention handled by attention component
    }

    /**
     * @brief Phi-3 Mini (3.8B parameters)
     *
     * Architecture:
     * - 3072 embedding dim, 32 layers, 32 heads
     * - Standard MHA (no GQA)
     * - SwiGLU activation
     * - 8192 hidden dim
     */
    export TransformerConfig Phi3_Mini()
    {
        return TransformerConfig( 3072, 32 )
            .withHiddenDimension( 8192 )
            .withBias( false )
            .withActivation( ActivationType::Swiglu );
    }

} // namespace Mila::Dnn::Presets

