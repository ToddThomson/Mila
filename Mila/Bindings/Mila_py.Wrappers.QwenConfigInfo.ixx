/**
 * @file Mila_py.Wrappers.QwenConfigInfo.ixx
 * @brief Qwen 3.8's shape, as a session reports it.
 */

export module Mila.Bindings:QwenConfigInfo;

import std;

namespace Mila::Bindings
{
    /**
     * @brief Qwen 3.8's shape, including the axes no other family in this binding has.
     *
     * The DeltaNet fields describe the recurrent mixer that occupies three of every four
     * layers, and full_attention_interval is what says how often the attention layer falls.
     * They are reported rather than derived, because nothing outside the checkpoint knows them.
     */
    export struct QwenConfigInfo
    {
        std::int64_t vocab_size;
        std::int64_t max_sequence_length;
        std::int64_t model_dim;
        std::int64_t num_layers;
        std::int64_t num_heads;
        std::int64_t num_kv_heads;
        std::int64_t head_dim;
        std::int64_t hidden_dim;
        double rope_theta;
        double partial_rotary_factor;
        bool attention_output_gate;
        std::int64_t full_attention_interval;
        std::int64_t linear_num_key_heads;
        std::int64_t linear_num_value_heads;
        std::int64_t linear_head_dim;
        std::int64_t linear_conv_kernel_dim;
    };
}
