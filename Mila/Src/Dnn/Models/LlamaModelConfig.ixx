/**
 * @file LlamaModelConfig.ixx
 * @brief Deployment configuration for LlamaModel.
 *
 * LlamaModelConfig is the concrete ModelConfig for LlamaModel. It carries
 * all model-wide deployment concerns via ModelConfig (context_length, strict,
 * precision_policy, quantization) with no additional Llama-specific fields
 * at this stage.
 *
 * All Llama architectural parameters (embedding_dim, num_heads, num_kv_heads,
 * hidden_dim, vocab_size, max_seq_length, rope_theta, use_bias) are sourced
 * from checkpoint metadata at load time and flow into LlamaConfig — they are
 * not deployment decisions and do not belong here.
 *
 * ## Intended usage
 *
 *   auto model_config = LlamaModelConfig( 4096 )
 *       .withQuantization( QuantizationConfig::fp8() )
 *       .withPrecisionPolicy( ComputePrecision::Policy::Performance )
 *       .withStrict( true );
 *
 *   auto model = LlamaModel<DeviceType::Cuda, TensorDataType::BF16>
 *       ::fromPretrained( path, model_config, device_id );
 *
 * ## Relationship to LlamaConfig
 *
 * LlamaConfig     — architectural config for LlamaTransformer (dims, heads, layers).
 *                   Populated from checkpoint metadata. Not a deployment concern.
 * LlamaModelConfig — deployment config for LlamaModel (precision, quantization,
 *                   context_length). Set by the caller at load time.
 */

module;
#include <string>
#include <cstdint>

export module Dnn.Models.LlamaModelConfig;

import Dnn.ModelConfig;
import Dnn.QuantizationConfig;
import Dnn.TensorTypes;

namespace Mila::Dnn
{
    /**
     * @brief Deployment configuration for LlamaModel.
     *
     * Concrete instantiation of ModelConfig for the Llama model family
     * (Llama 3.1, 3.2, and future variants). Provides type safety at the
     * fromPretrained() call site — a Qwen3ModelConfig cannot be passed
     * to LlamaModel::fromPretrained() and vice versa.
     *
     * Llama-specific deployment concerns will be added here as needed.
     * Architectural parameters always come from checkpoint metadata.
     */
    export class LlamaModelConfig : public ModelConfig
    {
    public:

        // ====================================================================
        // Construction
        // ====================================================================

        /**
         * @brief Construct with required context length.
         *
         * context_length is the only required deployment parameter —
         * everything else has a sensible default.
         *
         * @param context_length  Maximum sequence length in tokens. Must be > 0.
         */
        explicit LlamaModelConfig( dim_t context_length )
            : ModelConfig( context_length )
        {
        }

        /**
         * @brief Default constructor.
         *
         * context_length defaults to zero. Caller must invoke
         * withContextLength() before passing to fromPretrained().
         * Provided for cases where context_length is not known at
         * construction time.
         */
        LlamaModelConfig() = default;

        ~LlamaModelConfig() override = default;

        // ====================================================================
        // Diagnostics
        // ====================================================================

        /**
         * @brief Produce a human-readable summary of the Llama model config.
         *
         * Prepends a Llama-specific header and delegates base field
         * formatting to ModelConfig::baseToString().
         */
        std::string toString() const override
        {
            std::string result;
            result += "LlamaModelConfig\n";
            result += baseToString();
            return result;
        }
    };
}