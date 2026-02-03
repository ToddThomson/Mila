export module Dnn.Networks.Gpt:Presets;

import :Config;

namespace Mila::Dnn::Networks
{
    /**
     * Usage Examples:
     *
     * // Use a preset directly
     * auto config = Mila::Dnn::Networks::GPT2_Small();
     * auto network = GptNetwork(config, 50257, 1024, "gpt2_small");
     *
     * // Customize a preset
     * auto config = Mila::Dnn::Networks::GPT2_Small()
     *     .withDropout(0.2f);  // Custom dropout
     *
     * // Mix and match for research
     * auto custom = Mila::Dnn::Networks::GPT2_Medium()
     *     .withBias(false)  // Remove bias
     *     .withResidualScale(0.5f);  // Add residual scaling
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
      * - Learned positional embeddings
      */
    export GptConfig GPT2_Small()
    {
        return GptConfig( 768, 12 )
            .withNumHeads( 12 )
            .withHiddenSize( 3072 )
            .withBias( true );
    }

    /**
     * @brief GPT-2 Medium (345M parameters)
     *
     * Architecture:
     * - 1024 embedding dim, 24 layers, 16 heads
     * - 4096 hidden dim (4x embedding)
     */
    export GptConfig GPT2_Medium()
    {
        return GptConfig( 1024, 24 )
            .withNumHeads( 16 )
            .withHiddenSize( 4096 )
            .withBias( true );
    }

    /**
     * @brief GPT-2 Large (774M parameters)
     *
     * Architecture:
     * - 1280 embedding dim, 36 layers, 20 heads
     * - 5120 hidden dim (4x embedding)
     */
    export GptConfig GPT2_Large()
    {
        return GptConfig( 1280, 36 )
            .withNumHeads( 20 )
            .withHiddenSize( 5120 )
            .withBias( true );
    }

    /**
     * @brief GPT-2 XL (1.5B parameters)
     *
     * Architecture:
     * - 1600 embedding dim, 48 layers, 25 heads
     * - 6400 hidden dim (4x embedding)
     */
    export GptConfig GPT2_XL()
    {
        return GptConfig( 1600, 48 )
            .withNumHeads( 25 )
            .withHiddenSize( 6400 )
            .withBias( true );
    }
}