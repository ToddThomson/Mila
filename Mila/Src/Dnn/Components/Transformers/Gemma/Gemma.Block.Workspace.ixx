/**
 * @file Gemma.Block.Workspace.ixx
 * @brief Transformer-owned shared activation slots for GemmaBlock (pooling).
 *
 * One slot set serves every layer; the transformer owns it and accounts for it.
 */

module;
#include <memory>

export module Dnn.Components.GemmaBlock:Workspace;

import Dnn.Tensor;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Transformer-owned shared activation workspace for GemmaBlock (pooling).
     *
     * One slot per block-graph position, shared by every layer: the inference path
     * is strictly sequential, so exactly one block is live at a time and 47/48 of
     * per-layer retained activations are never read again. Slots are sized
     * [B, chunk, max(local, global) width]; components view prefixes (the GQA
     * workspace max-geometry convention). The single stream slot is alias-safe:
     * a block's input is last read at res_1 (mid-block) and only overwritten by
     * its own res_2 at block end.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    struct GemmaBlockWorkspace
    {
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;

        // Block-owned split scratch (written by split, read by the QK/V norms,
        // RoPE, and -- on global K=V layers -- v_norm reading the raw k projection).
        std::shared_ptr<TensorType> q;
        std::shared_ptr<TensorType> k;
        std::shared_ptr<TensorType> v;

        // Component output slots, one per graph position (prefill order).
        std::shared_ptr<TensorType> normed;      // input_norm out       [B, chunk, model_dim]
        std::shared_ptr<TensorType> qkv;         // qkv_proj out         [B, chunk, max packed QKV width]
        std::shared_ptr<TensorType> q_normed;    // q_norm out           [B, chunk, NH * max head_dim]
        std::shared_ptr<TensorType> k_normed;    // k_norm out           [B, chunk, max KV width]
        std::shared_ptr<TensorType> v_normed;    // v_norm out           [B, chunk, max KV width]
        std::shared_ptr<TensorType> attn;        // gqa prefill out      [B, chunk, NH * max head_dim]
        std::shared_ptr<TensorType> o;           // o_proj out           [B, chunk, model_dim]
        std::shared_ptr<TensorType> o_normed;    // post_attn_norm out   [B, chunk, model_dim]
        std::shared_ptr<TensorType> res1;        // res_1 out            [B, chunk, model_dim]
        std::shared_ptr<TensorType> ffn_in;      // pre_ffn_norm out     [B, chunk, model_dim]
        std::shared_ptr<TensorType> gate_up;     // fc_gate_up out       [B, chunk, 2 * hidden_dim]
        std::shared_ptr<TensorType> ffn_act;     // geglu out            [B, chunk, hidden_dim]
        std::shared_ptr<TensorType> ffn_down;    // fc_down out          [B, chunk, model_dim]
        std::shared_ptr<TensorType> ffn_normed;  // post_ffn_norm out    [B, chunk, model_dim]
        std::shared_ptr<TensorType> stream;      // res_2 out            [B, chunk, model_dim]
    };
}
