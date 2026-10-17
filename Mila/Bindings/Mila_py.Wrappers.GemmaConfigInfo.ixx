/**
 * @file Mila_py.Wrappers.GemmaConfigInfo.ixx
 * @brief Gemma 4's shape, as a session reports it.
 */

export module Mila.Bindings:GemmaConfigInfo;

import std;

namespace Mila::Bindings
{
    export struct GemmaConfigInfo
    {
        std::int64_t vocab_size;
        std::int64_t max_sequence_length;
        std::int64_t model_dim;
        std::int64_t num_layers;
        std::int64_t num_heads;
        std::int64_t num_kv_heads;
        std::int64_t head_dim;
        std::int64_t global_head_dim;
        std::int64_t hidden_dim;
        std::int64_t window;
        double rope_theta_local;
        double rope_theta_global;
        double final_logit_softcapping;
    };
}
