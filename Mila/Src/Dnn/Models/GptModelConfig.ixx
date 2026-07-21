/**
 * @file GptModelConfig.ixx
 * @brief Deployment configuration for Gpt2 language models.
 *
 * GptModelConfig is the concrete configuration type passed to
 * GptModel::fromPretrained(). It inherits all universal language model
 * deployment concerns from LanguageModelConfig<GemmaModelConfig>:
 *
 *   - context_length        -- maximum sequence length
 *   - weight_quantization   -- Linear weight storage strategy
 *   - kv_cache_compression  -- GroupedQueryAttention cache strategy
 *
 */

module;
#include <string>

export module Dnn.Models.GptModelConfig;

import Dnn.LanguageModelConfig;
import Dnn.TensorTypes;

namespace Mila::Dnn
{
    /**
     * @brief Deployment configuration for Gpt language models.
     *
     * Inherits all fluent setters and accessors from
     * LanguageModelConfig<GptModelConfig>.
     */
    export struct GptModelConfig : LanguageModelConfig<GptModelConfig>
    {
        GptModelConfig() = default;

        /**
         * @brief Construct with a required context length.
         *
         * @param context_length  Maximum sequence length in tokens. Must be > 0.
         * @throws std::invalid_argument if context_length is zero.
         */
        explicit GptModelConfig( dim_t context_length )
            : LanguageModelConfig<GptModelConfig>( context_length )
        {
        }

        std::string toString() const
        {
            std::string result = "GptModelConfig:\n";
            result += baseToString();

            return result;
        }
    };
}
