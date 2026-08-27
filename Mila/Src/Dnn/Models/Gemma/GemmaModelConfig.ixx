/**
 * @file GemmaModelConfig.ixx
 * @brief Deployment configuration for Gemma language models.
 *
 * GemmaModelConfig is the concrete configuration type passed to
 * GemmaModel::fromPretrained(). It inherits all universal language model
 * deployment concerns from LanguageModelConfig<GemmaModelConfig>:
 *
 *   - context_length        -- maximum sequence length
 *   - weight_quantization   -- Linear weight storage strategy
 *   - kv_cache_compression  -- GroupedQueryAttention cache strategy
 *
 * Every Gemma architectural parameter (num_layers, head_dim, the global-layer
 * geometry, the sliding window, dual RoPE bases, logit softcap, etc.) is read
 * from checkpoint metadata at load time and is not a deployment concern, so
 * GemmaModelConfig carries no architecture-specific fields beyond the base.
 */

module;
#include <string>

export module Dnn.Models.GemmaModelConfig;

// Re-exported, not merely imported: this config's own setters take WeightQuantization
// and KvCacheCompression, so a consumer that cannot name them cannot call them.
export import Dnn.LanguageModelConfig;
import Dnn.TensorTypes;

namespace Mila::Dnn
{
    /**
     * @brief Deployment configuration for Gemma language models.
     *
     * Inherits all fluent setters and accessors from
     * LanguageModelConfig<GemmaModelConfig>.
     */
    export struct GemmaModelConfig : LanguageModelConfig<GemmaModelConfig>
    {
        GemmaModelConfig() = default;

        /**
         * @brief Construct with a required context length.
         *
         * @param context_length  Maximum sequence length in tokens. Must be > 0.
         * @throws std::invalid_argument if context_length is zero.
         */
        explicit GemmaModelConfig( dim_t context_length )
            : LanguageModelConfig<GemmaModelConfig>( context_length )
        {
        }

        std::string toString() const
        {
            std::string result = "GemmaModelConfig:\n";
            result += baseToString();

            return result;
        }
    };
}
