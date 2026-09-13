/**
 * @file Gemma.Block.Workspace.ixx
 * @brief Transformer-owned shared activation slots for GemmaBlock (pooling), and their factory.
 *
 * One slot set serves every layer; the transformer owns it and accounts for it.
 */

module;
#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>

export module Dnn.Components.GemmaBlock:Workspace;

import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.Components.GemmaConfig;
import Compute.DeviceId;
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

        // Routed feed-forward only (kMixtureOfExperts); null on a dense model.
        std::shared_ptr<TensorType> ffn_dense_normed;  // post_ffn_norm_1 out  [B, chunk, model_dim]
        std::shared_ptr<TensorType> ffn_expert_in;     // pre_ffn_norm_2 out   [B, chunk, model_dim]
        std::shared_ptr<TensorType> ffn_expert_normed; // post_ffn_norm_2 out  [B, chunk, model_dim]
        std::shared_ptr<TensorType> ffn_sum;           // ffn_sum out          [B, chunk, model_dim]

        std::size_t deviceStorageBytes() const
        {
            std::size_t total = 0;

            for ( const auto* t : { q.get(), k.get(), v.get(), normed.get(), qkv.get(), q_normed.get(),
                                    k_normed.get(), v_normed.get(), attn.get(), o.get(), o_normed.get(),
                                    res1.get(), ffn_in.get(), gate_up.get(), ffn_act.get(), ffn_down.get(),
                                    ffn_normed.get(), stream.get(), ffn_dense_normed.get(), ffn_expert_in.get(),
                                    ffn_expert_normed.get(), ffn_sum.get() } )
            {
                if ( t )
                    total += t->getStorageSize();
            }

            return total;
        }
    };

    /**
     * @brief Max-geometry slot widths, shared by the allocation and the transformer's chunk row-cost model.
     */
    export struct GemmaBlockWorkspaceWidths
    {
        dim_t model_dim{ 0 };
        dim_t hidden_dim{ 0 };
        dim_t q_width{ 0 };
        dim_t kv_width{ 0 };
        dim_t qkv_width{ 0 };

        // Stream-wide slots the routed feed-forward adds; zero on a dense model.
        dim_t routed_stream_slots{ 0 };

        // q + q_normed + attn; k + v + k_normed + v_normed; the eight
        // model_dim-wide stream-side slots; qkv; gate_up (2h) + ffn_act (h).
        dim_t totalRowElements() const
        {
            return 3 * q_width + 4 * kv_width + ( 8 + routed_stream_slots ) * model_dim + qkv_width + 3 * hidden_dim;
        }
    };

    export GemmaBlockWorkspaceWidths gemmaBlockWorkspaceWidths( const GemmaConfig& config )
    {
        const dim_t NH = config.getNumHeads();

        GemmaBlockWorkspaceWidths widths;
        widths.model_dim = config.getModelDim();
        widths.hidden_dim = config.getHiddenDimension();
        widths.q_width = NH * std::max( config.getHeadDim(), config.getGlobalHeadDim() );
        widths.kv_width = std::max(
            config.getNumKVHeads() * config.getHeadDim(),
            config.getNumGlobalKVHeads() * config.getGlobalHeadDim() );

        // Packed QKV width per layer kind: global K=V layers drop the V section.
        const dim_t packed_local = ( NH + 2 * config.getNumKVHeads() ) * config.getHeadDim();
        const dim_t packed_global =
            ( NH + ( config.keyEqualsValue() ? 1 : 2 ) * config.getNumGlobalKVHeads() ) * config.getGlobalHeadDim();
        widths.qkv_width = std::max( packed_local, packed_global );
        widths.routed_stream_slots = config.hasMixtureOfExperts() ? 4 : 0;

        return widths;
    }

    /**
     * @brief Allocate the shared block workspace at the widths gemmaBlockWorkspaceWidths() reports.
     *
     * Lives beside the struct because the transformer is not the only caller: a layer-streamed parity
     * harness holds one block at a time and must measure the geometry the model builds, not a copy of it.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    GemmaBlockWorkspace<TDeviceType, TPrecision> makeGemmaBlockWorkspace(
        const GemmaConfig& config, DeviceId device, dim_t B, dim_t prefill_chunk, const std::string& name_prefix )
    {
        using TensorType = typename GemmaBlockWorkspace<TDeviceType, TPrecision>::TensorType;

        const GemmaBlockWorkspaceWidths widths = gemmaBlockWorkspaceWidths( config );

        auto slot = [&]( dim_t width, const char* name )
        {
            return std::make_shared<TensorType>( device, shape_t{ B, prefill_chunk, width }, name_prefix + name );
        };

        GemmaBlockWorkspace<TDeviceType, TPrecision> workspace;

        workspace.q = slot( widths.q_width, "q" );
        workspace.k = slot( widths.kv_width, "k" );
        workspace.v = slot( widths.kv_width, "v" );
        workspace.normed = slot( widths.model_dim, "normed" );
        workspace.qkv = slot( widths.qkv_width, "qkv" );
        workspace.q_normed = slot( widths.q_width, "q_normed" );
        workspace.k_normed = slot( widths.kv_width, "k_normed" );
        workspace.v_normed = slot( widths.kv_width, "v_normed" );
        workspace.attn = slot( widths.q_width, "attn" );
        workspace.o = slot( widths.model_dim, "o" );
        workspace.o_normed = slot( widths.model_dim, "o_normed" );
        workspace.res1 = slot( widths.model_dim, "res1" );
        workspace.ffn_in = slot( widths.model_dim, "ffn_in" );
        workspace.gate_up = slot( 2 * widths.hidden_dim, "gate_up" );
        workspace.ffn_act = slot( widths.hidden_dim, "ffn_act" );
        workspace.ffn_down = slot( widths.model_dim, "ffn_down" );
        workspace.ffn_normed = slot( widths.model_dim, "ffn_normed" );
        workspace.stream = slot( widths.model_dim, "stream" );

        if ( widths.routed_stream_slots > 0 )
        {
            workspace.ffn_dense_normed = slot( widths.model_dim, "ffn_dense_normed" );
            workspace.ffn_expert_in = slot( widths.model_dim, "ffn_expert_in" );
            workspace.ffn_expert_normed = slot( widths.model_dim, "ffn_expert_normed" );
            workspace.ffn_sum = slot( widths.model_dim, "ffn_sum" );
        }

        return workspace;
    }
}
