// LlamaNetwork-Presets.ixx

export module Dnn.Components.LlamaTransformer:Presets;

import :Config;

namespace Mila::Dnn
{
    /**
     * Usage Examples:
     *
     * // Use a preset directly
     * auto config = Mila::Dnn::Networks::Llama3_2_1B();
     * auto network = LlamaNetwork(config, 128256, 131072, "llama3_2_1b");
     *
     * // Customize a preset
     * auto config = Mila::Dnn::Networks::Llama3_8B()
     *     .withRoPETheta(1000000.0f);  // Extend context with higher theta
     *
     * // Mix and match for research
     * auto custom = Mila::Dnn::Networks::Llama3_8B()
     *     .withNumKVHeads(32)  // Convert GQA to MHA
     *     .withResidualScale(0.5f);  // Add residual scaling
     */

     // ========================================================================
     // Llama 3.2 Preset Family
     // ========================================================================

     /**
      * @brief Llama 3.2 1B
      *
      * Architecture:
      * - 2048 embedding dim, 16 layers
      * - 32 heads
      * - Grouped Query Attention (8 KV heads)
      * - RMSNorm + SwiGLU activation
      * - 8192 hidden dim (~4x for SwiGLU)
      * - No bias
      * - RoPE positional encoding (theta=500000)
      * - 128k context window
      */
    export LlamaConfig Llama3_2_1B()
    {
        return LlamaConfig( 2048, 16 )
            .withNumHeads( 32 )
            .withNumKVHeads( 8 )
            .withHiddenDimension( 8192 )
            .withRoPETheta( 500000.0f )
            .withRoPEScalingFactor( 32.0f );
    }

    /**
     * @brief Llama 3.2 3B
     *
     * Architecture:
     * - 3072 embedding dim, 28 layers
     * - 24 heads
     * - Grouped Query Attention (8 KV heads)
     * - 8192 hidden dim
     * - 128k context window
     */
    export LlamaConfig Llama3_2_3B()
    {
        return LlamaConfig( 3072, 28 )
            .withNumHeads( 24 )
            .withNumKVHeads( 8 )
            .withHiddenDimension( 8192 )
            .withRoPETheta( 500000.0f )
            .withRoPEScalingFactor( 32.0f );
    }

    // ========================================================================
    // Llama 3.1 Preset Family
    // ========================================================================

    /**
     * @brief Llama 3.1 8B
     *
     * Architecture:
     * - 4096 embedding dim, 32 layers
     * - 32 heads
     * - Grouped Query Attention (8 KV heads)
     * - 14336 hidden dim (~3.5x for SwiGLU)
     * - No bias
     * - RoPE positional encoding (theta=500000)
     * - 128k context window
     */
    export LlamaConfig Llama3_1_8B()
    {
        return LlamaConfig( 4096, 32 )
            .withNumHeads( 32 )
            .withNumKVHeads( 8 )
            .withHiddenDimension( 14336 )
            .withRoPETheta( 500000.0f )
            .withRoPEScalingFactor( 8.0f );
    }

    /**
     * @brief Llama 3.1 70B
     *
     * Architecture:
     * - 8192 embedding dim, 80 layers, 64 heads
     * - Grouped Query Attention (8 KV heads)
     * - 28672 hidden dim
     * - 128k context window
     */
    export LlamaConfig Llama3_1_70B()
    {
        return LlamaConfig( 8192, 80 )
            .withNumHeads( 64 )
            .withNumKVHeads( 8 )
            .withHiddenDimension( 28672 )
            .withRoPETheta( 500000.0f )
            .withRoPEScalingFactor( 8.0f );
    }

    /**
     * @brief Llama 3.1 405B
     *
     * Architecture:
     * - 16384 embedding dim, 126 layers, 128 heads
     * - Grouped Query Attention (8 KV heads)
     * - 53248 hidden dim
     * - 128k context window
     */
    export LlamaConfig Llama3_1_405B()
    {
        return LlamaConfig( 16384, 126 )
            .withNumHeads( 128 )
            .withNumKVHeads( 8 )
            .withHiddenDimension( 53248 )
            .withRoPETheta( 500000.0f )
            .withRoPEScalingFactor( 8.0f );
    }

    // ========================================================================
    // Llama 3 (Original) Preset Family
    // ========================================================================

    /**
     * @brief Llama 3 8B (Original release)
     *
     * Architecture:
     * - Same as 3.1 8B but with 8k context window (no scaling)
     */
    export LlamaConfig Llama3_8B()
    {
        return LlamaConfig( 4096, 32 )
            .withNumHeads( 32 )
            .withNumKVHeads( 8 )
            .withHiddenDimension( 14336 )
            .withRoPETheta( 500000.0f );
        // No RoPE scaling - 8k native context
    }

    /**
     * @brief Llama 3 70B (Original release)
     *
     * Architecture:
     * - Same as 3.1 70B but with 8k context window (no scaling)
     */
    export LlamaConfig Llama3_70B()
    {
        return LlamaConfig( 8192, 80 )
            .withNumHeads( 64 )
            .withNumKVHeads( 8 )
            .withHiddenDimension( 28672 )
            .withRoPETheta( 500000.0f );
        // No RoPE scaling - 8k native context
    }

    // ========================================================================
    // Llama 2 Preset Family
    // ========================================================================

    /**
     * @brief Llama 2 7B
     *
     * Architecture:
     * - 4096 embedding dim, 32 layers, 32 heads
     * - Grouped Query Attention (32 KV heads - effectively MHA)
     * - 11008 hidden dim
     * - RoPE theta=10000 (original)
     * - 4k context window
     */
    export LlamaConfig Llama2_7B()
    {
        return LlamaConfig( 4096, 32 )
            .withNumHeads( 32 )
            .withNumKVHeads( 32 )  // MHA for Llama 2
            .withHiddenDimension( 11008 )
            .withRoPETheta( 10000.0f );
    }

    /**
     * @brief Llama 2 13B
     *
     * Architecture:
     * - 5120 embedding dim, 40 layers, 40 heads
     * - MHA (40 KV heads)
     * - 13824 hidden dim
     */
    export LlamaConfig Llama2_13B()
    {
        return LlamaConfig( 5120, 40 )
            .withNumHeads( 40 )
            .withNumKVHeads( 40 )  // MHA for Llama 2
            .withHiddenDimension( 13824 )
            .withRoPETheta( 10000.0f );
    }

    /**
     * @brief Llama 2 70B
     *
     * Architecture:
     * - 8192 embedding dim, 80 layers, 64 heads
     * - Grouped Query Attention (8 KV heads)
     * - 28672 hidden dim
     */
    export LlamaConfig Llama2_70B()
    {
        return LlamaConfig( 8192, 80 )
            .withNumHeads( 64 )
            .withNumKVHeads( 8 )  // GQA for Llama 2 70B
            .withHiddenDimension( 28672 )
            .withRoPETheta( 10000.0f );
    }
}