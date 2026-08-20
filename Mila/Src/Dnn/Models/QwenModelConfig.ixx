/**
 * @file QwenModelConfig.ixx
 * @brief Deployment configuration for Qwen 3.8 language models.
 *
 * QwenModelConfig is the concrete configuration type passed to
 * QwenModel::fromPretrained(). It inherits every universal deployment concern from
 * LanguageModelConfig<QwenModelConfig>:
 *
 *   - context_length        -- maximum sequence length
 *   - weight_quantization   -- Linear weight storage strategy
 *   - kv_cache_compression  -- GroupedQueryAttention cache strategy
 *
 * Every architectural parameter (the 3:1 mixer interleave, the DeltaNet head geometry,
 * the attention output gate, the partial rotary width) is read from checkpoint metadata
 * at load time and is not a deployment concern, so this carries no architecture fields
 * beyond the base.
 */

module;
#include <string>

export module Dnn.Models.QwenModelConfig;

// Re-exported, not merely imported: this config's own setters take WeightQuantization
// and KvCacheCompression, so a consumer that cannot name them cannot call them.
export import Dnn.LanguageModelConfig;
import Dnn.TensorTypes;

namespace Mila::Dnn
{
    /**
     * @brief Deployment configuration for Qwen 3.8 language models.
     *
     * Inherits all fluent setters and accessors from
     * LanguageModelConfig<QwenModelConfig>.
     */
    export struct QwenModelConfig : LanguageModelConfig<QwenModelConfig>
    {
        QwenModelConfig() = default;

        /**
         * @brief Construct with a required context length.
         *
         * @param context_length  Maximum sequence length in tokens. Must be > 0.
         * @throws std::invalid_argument if context_length is zero.
         */
        explicit QwenModelConfig( dim_t context_length )
            : LanguageModelConfig<QwenModelConfig>( context_length )
        {
        }

        std::string toString() const
        {
            std::string result = "QwenModelConfig:\n";
            result += baseToString();

            return result;
        }
    };
}
