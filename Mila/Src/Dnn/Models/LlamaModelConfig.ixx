/**
 * @file LlamaModelConfig.ixx
 * @brief Deployment configuration for Llama language models.
 *
 * LlamaModelConfig is the concrete configuration type passed to
 * LlamaModel::fromPretrained(). It inherits all universal language model
 * deployment concerns from LanguageModelConfig<LlamaModelConfig>:
 *
 *   - context_length        — maximum sequence length
 *   - weight_quantization   — Linear weight storage strategy
 *   - kv_cache_compression  — GroupedQueryAttention cache strategy
 *
 * All Llama architectural parameters (num_layers, num_heads, hidden_dim,
 * rope_theta, vocab_size, etc.) are read from checkpoint metadata at load
 * time and are not deployment concerns. LlamaModelConfig carries no
 * architecture-specific fields beyond what the base provides.
 *
 * ## Usage
 *
 * @code
 * // Standard BF16 inference
 * auto config = LlamaModelConfig( context_length );
 *
 * // FP8 weights + FP8 KV cache
 * auto config = LlamaModelConfig( context_length )
 *     .withFP8Quantization();
 *
 * // FP8 weights only — no KV compression
 * auto config = LlamaModelConfig( context_length )
 *     .withWeightQuantization( WeightQuantization::FP8 );
 * @endcode
 */

 module;
 #include <stdexcept>
 #include <string>
 #include <type_traits>
 #include <concepts>
 
export module Dnn.Models.LlamaModelConfig;

import Dnn.LanguageModelConfig;
import Dnn.TensorTypes;

namespace Mila::Dnn
{
    /**
     * @brief Deployment configuration for Llama language models.
     *
     * Inherits all fluent setters and accessors from
     * LanguageModelConfig<LlamaModelConfig>. Chains work across base
     * and derived methods without casting.
     */
    export struct LlamaModelConfig : LanguageModelConfig<LlamaModelConfig>
    {
        // =====================================================================
        // Construction
        // =====================================================================

        /**
         * @brief Default constructor.
         *
         * context_length defaults to zero. Call withContextLength() before
         * passing to fromPretrained(), or use the explicit constructor.
         */
        LlamaModelConfig() = default;

        /**
         * @brief Construct with a required context length.
         *
         * @param context_length  Maximum sequence length in tokens. Must be > 0.
         * @throws std::invalid_argument if context_length is zero.
         */
        explicit LlamaModelConfig( dim_t context_length )
            : LanguageModelConfig<LlamaModelConfig>( context_length )
        {
        }

        // =====================================================================
        // Diagnostics
        // =====================================================================

        /**
         * @brief Produce a human-readable summary of the Llama model configuration.
         */
        std::string toString() const
        {
            std::string result = "LlamaModelConfig:\n";
            result += baseToString();
            return result;
        }
    };

}
