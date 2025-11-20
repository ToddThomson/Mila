/**
 * @file VisualizerContext.ixx
 * @brief Context container holding snapshot tensor references for the visualization pipeline.
 */

module;
#include <memory>
#include <vector>

export module Visualization.Context;

import Dnn.ITensor;
import Dnn.TensorTypes;

namespace Mila::Dnn::Visualization
{
    // Lightweight context that holds references to tensors (snapshotted views).
    // Consumers should provide ownership (shared_ptr) to concrete tensor instances
    // that implement the `Mila::Dnn::ITensor` interface.
    export struct VisualizerContext
    {
        // Per-token / per-position tensors
        std::shared_ptr<Mila::Dnn::ITensor> ln_out;   // (tokens x hidden)
        std::shared_ptr<Mila::Dnn::ITensor> q_proj;   // (tokens x head_dim_total)
        std::shared_ptr<Mila::Dnn::ITensor> k_proj;
        std::shared_ptr<Mila::Dnn::ITensor> v_proj;

        // Attention heads: one ITensor per head (each typically tokens x tokens)
        std::vector<std::shared_ptr<Mila::Dnn::ITensor>> attn_heads;

        // MLP activations
        std::shared_ptr<Mila::Dnn::ITensor> mlp_l1;   // (tokens x updim)
        std::shared_ptr<Mila::Dnn::ITensor> mlp_act;  // (tokens x updim)
        std::shared_ptr<Mila::Dnn::ITensor> mlp_l2;   // (tokens x hidden)

        // Per-token metadata (simple host-side values)
        std::vector<float> residual_norms; // length = tokens

        // metadata
        int seq_len{0};
        int hidden_dim{0};
        int n_heads{0};
        int head_dim{0};
    };
}