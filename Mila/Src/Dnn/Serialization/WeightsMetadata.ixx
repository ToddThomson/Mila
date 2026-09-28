/**
 * @file WeightsMetadata.ixx
 * @brief The model configuration a weights file carries, and the JSON a writer stores it as.
 */

module;
#include <cstdint>
#include <string>

export module Serialization.WeightsMetadata;

import nlohmann.json;

namespace Mila::Dnn::Serialization
{
    /**
     * @brief The model configuration a weights file carries.
     */
    export struct WeightsMetadata
    {
        std::string architecture;
        std::string model_name;
        uint32_t vocab_size;
        uint32_t max_seq_length;
        uint32_t embedding_dim;
        uint32_t num_layers;
        uint32_t num_heads;
        uint32_t num_kv_heads;
        uint32_t head_dim;          // explicit per-head width (Gemma decouples it from embedding_dim/num_heads); 0 = derive
        uint32_t hidden_dim;
        bool use_bias;
        bool tie_word_embeddings = false;

        std::string activation;
        std::string norm_type;
        std::string attention_type;
        std::string positional_encoding;

        float rope_theta;
        float norm_epsilon;

        // Rotary frequency scaling (RopeFrequencyScaling). `rope_scaling` names the rule: "none", or "llama3" with the
        // four values below. Empty means a file written before the field existed, which says nothing either way.
        std::string rope_scaling;
        float rope_scaling_factor = 0.0f;
        float rope_low_frequency_factor = 0.0f;
        float rope_high_frequency_factor = 0.0f;
        uint32_t rope_original_context_length = 0;

        // Gemma-specific geometry (0 / false for other architectures). The global
        // (full-attention) layers diverge from the sliding layers; the chassis fields
        // drive the 5:1 interleave, dual RoPE, and logit softcap. See GemmaConfig.
        uint32_t global_head_dim;
        uint32_t num_global_kv_heads;
        bool     key_equals_value;
        uint32_t window;
        uint32_t sliding_window_pattern;
        uint32_t global_rotary_dim;
        float    rope_theta_local;
        float    rope_theta_global;
        float    final_logit_softcapping;

        // Routed feed-forward geometry, zero for a dense model (Gemma 4 26B-A4B: 128 experts, top 8,
        // expert width 704). hidden_dim stays the width of the always-on dense branch.
        uint32_t num_experts = 0;
        uint32_t top_k_experts = 0;
        uint32_t expert_hidden_dim = 0;

        // Qwen 3.8 geometry (0 / false for other architectures). This stack interleaves two
        // different MIXERS rather than two geometries of one mixer, so the Gated DeltaNet
        // fields are its own rather than variants of the attention ones above. See QwenConfig.
        bool     attention_output_gate = false;
        uint32_t full_attention_interval = 0;
        float    partial_rotary_factor = 0.0f;
        uint32_t linear_num_key_heads = 0;
        uint32_t linear_num_value_heads = 0;
        uint32_t linear_head_dim = 0;
        uint32_t linear_conv_kernel_dim = 0;
    };

    /**
     * @brief Serialize WeightsMetadata to the JSON the reader parses back.
     *
     * The inverse of WeightsReader's parser: every field it extracts is emitted, so a written model carries the same
     * architecture description a converted one does.
     *
     * Keys are quoted on both sides by the parser, so no key can match inside a longer one
     * ("rope_theta" does not match within "rope_theta_local"). Do not introduce a key that
     * is a prefix of another up to its closing quote.
     */
    export inline std::string toMetadataJSON( const WeightsMetadata& metadata )
    {
        nlohmann::json json;

        json[ "architecture" ] = metadata.architecture;
        json[ "model_name" ] = metadata.model_name;
        json[ "vocab_size" ] = metadata.vocab_size;
        json[ "max_seq_length" ] = metadata.max_seq_length;
        json[ "embedding_dim" ] = metadata.embedding_dim;
        json[ "num_layers" ] = metadata.num_layers;
        json[ "num_heads" ] = metadata.num_heads;
        json[ "num_kv_heads" ] = metadata.num_kv_heads;
        json[ "head_dim" ] = metadata.head_dim;
        json[ "hidden_dim" ] = metadata.hidden_dim;
        json[ "use_bias" ] = metadata.use_bias;
        json[ "tie_word_embeddings" ] = metadata.tie_word_embeddings;

        json[ "activation" ] = metadata.activation;
        json[ "norm_type" ] = metadata.norm_type;
        json[ "attention_type" ] = metadata.attention_type;
        json[ "positional_encoding" ] = metadata.positional_encoding;

        json[ "rope_theta" ] = metadata.rope_theta;
        json[ "norm_epsilon" ] = metadata.norm_epsilon;

        // Omitted rather than written empty, so a rewritten old file still reads as one.
        if ( !metadata.rope_scaling.empty() )
        {
            json[ "rope_scaling" ] = metadata.rope_scaling;
            json[ "rope_scaling_factor" ] = metadata.rope_scaling_factor;
            json[ "rope_low_frequency_factor" ] = metadata.rope_low_frequency_factor;
            json[ "rope_high_frequency_factor" ] = metadata.rope_high_frequency_factor;
            json[ "rope_original_context_length" ] = metadata.rope_original_context_length;
        }

        json[ "global_head_dim" ] = metadata.global_head_dim;
        json[ "num_global_kv_heads" ] = metadata.num_global_kv_heads;
        json[ "key_equals_value" ] = metadata.key_equals_value;
        json[ "window" ] = metadata.window;
        json[ "sliding_window_pattern" ] = metadata.sliding_window_pattern;
        json[ "global_rotary_dim" ] = metadata.global_rotary_dim;
        json[ "rope_theta_local" ] = metadata.rope_theta_local;
        json[ "rope_theta_global" ] = metadata.rope_theta_global;
        json[ "final_logit_softcapping" ] = metadata.final_logit_softcapping;

        json[ "num_experts" ] = metadata.num_experts;
        json[ "top_k_experts" ] = metadata.top_k_experts;
        json[ "expert_hidden_dim" ] = metadata.expert_hidden_dim;

        json[ "attention_output_gate" ] = metadata.attention_output_gate;
        json[ "full_attention_interval" ] = metadata.full_attention_interval;
        json[ "partial_rotary_factor" ] = metadata.partial_rotary_factor;
        json[ "linear_num_key_heads" ] = metadata.linear_num_key_heads;
        json[ "linear_num_value_heads" ] = metadata.linear_num_value_heads;
        json[ "linear_head_dim" ] = metadata.linear_head_dim;
        json[ "linear_conv_kernel_dim" ] = metadata.linear_conv_kernel_dim;

        return json.dump();
    }
}
