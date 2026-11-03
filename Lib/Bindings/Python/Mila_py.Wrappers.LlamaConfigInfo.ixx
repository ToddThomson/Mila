/**
 * @file Mila_py.Wrappers.LlamaConfigInfo.ixx
 * @brief Llama 3.x's shape, as a session reports it.
 */

export module Mila.Bindings:LlamaConfigInfo;

import std;

namespace Mila::Bindings
{
    export struct LlamaConfigInfo
    {
        std::int64_t vocab_size;
        std::int64_t max_sequence_length;
        std::int64_t model_dim;
        std::int64_t num_layers;
        std::int64_t num_heads;
        std::int64_t num_kv_heads;
        std::int64_t hidden_dim;
        double rope_theta;
    };
}
