/**
 * @file Qwen.DeltaNetBlock.Workspace.ixx
 * @brief Transformer-owned shared activation slots for QwenDeltaNetBlock (pooling), and their widths and factory.
 *
 * A module of its own: QwenTransformer and the layer-streamed harnesses name the workspace apart from the block.
 */

module;
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

export module Dnn.Components.QwenDeltaNetBlockWorkspace;

import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.Components.QwenConfig;
import Compute.DeviceAllocation;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    /**
     * @brief Transformer-owned shared activation workspace for QwenDeltaNetBlock (pooling).
     *
     * The SECOND workspace struct QwenAttentionBlockWorkspace's comment anticipates. Not a
     * wider version of it: a DeltaNet layer's slots are shaped by the mixer's own geometry
     * -- fused [query|key], the value stream, two per-head gating scalars -- and share no
     * width with the attention block's beyond the residual stream itself.
     *
     * One slot set serves every DeltaNet layer, because inference is strictly sequential and
     * exactly one block is live at a time. Slots are sized [B, chunk, width] and components
     * view prefixes of them. Aliasing is the same argument the attention workspace makes and
     * it has three live points here: `normed` is read by all five input projections, `z`
     * survives from its projection to the output gate, and `res1` is read again at res_2 --
     * so each of those is its own slot, and `stream` (the block's own input, and its output)
     * is only overwritten at block end.
     *
     * Without this, every DeltaNet layer self-allocated its outputs: 138.2 MiB per layer on
     * the 27B at chunk 512, ~6.5 GiB over 48 layers, which is what capped the model at 512
     * context and made WDDM page its weights (Qwen3.8.md section 8, Phase 5).
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    struct QwenDeltaNetBlockWorkspace
    {
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;

        // Block-owned split scratch: the q and k halves of the convolved [query|key] stream.
        std::shared_ptr<TensorType> q;
        std::shared_ptr<TensorType> k;

        // Component output slots, one per graph position (prefill order).
        std::shared_ptr<TensorType> normed;       // input_norm out     [B, chunk, model_dim]
        std::shared_ptr<TensorType> qk;           // fc_in_proj_qk out  [B, chunk, qk_width]
        std::shared_ptr<TensorType> v;            // fc_in_proj_v out   [B, chunk, value_width]
        std::shared_ptr<TensorType> z;            // fc_in_proj_z out   [B, chunk, value_width]
        std::shared_ptr<TensorType> a;            // fc_in_proj_a out   [B, chunk, gating_width]
        std::shared_ptr<TensorType> b;            // fc_in_proj_b out   [B, chunk, gating_width]
        std::shared_ptr<TensorType> conv_qk;      // conv_qk out        [B, chunk, qk_width]
        std::shared_ptr<TensorType> conv_v;       // conv_v out         [B, chunk, value_width]
        std::shared_ptr<TensorType> act_qk;       // conv_act_qk out    [B, chunk, qk_width]
        std::shared_ptr<TensorType> act_v;        // conv_act_v out     [B, chunk, value_width]
        std::shared_ptr<TensorType> core;         // delta_rule out     [B, chunk, value_width]
        std::shared_ptr<TensorType> core_normed;  // norm_gate out      [B, chunk, value_width]
        std::shared_ptr<TensorType> gated;        // output_gate out    [B, chunk, value_width]
        std::shared_ptr<TensorType> mixed;        // fc_out_proj out    [B, chunk, model_dim]
        std::shared_ptr<TensorType> res1;         // res_1 out          [B, chunk, model_dim]
        std::shared_ptr<TensorType> ffn_in;       // post_attn_norm out [B, chunk, model_dim]
        std::shared_ptr<TensorType> gate_up;      // fc_gate_up out     [B, chunk, 2 * hidden]
        std::shared_ptr<TensorType> ffn_act;      // swiglu out         [B, chunk, hidden_dim]
        std::shared_ptr<TensorType> ffn_down;     // fc_down out        [B, chunk, model_dim]
        std::shared_ptr<TensorType> stream;       // res_2 out          [B, chunk, model_dim]

        /// Total device bytes held, for memory accounting.
        std::size_t deviceStorageBytes() const
        {
            std::size_t total = 0;

            for ( const auto* t : { q.get(), k.get(), normed.get(), qk.get(), v.get(), z.get(),
                                    a.get(), b.get(), conv_qk.get(), conv_v.get(), act_qk.get(),
                                    act_v.get(), core.get(), core_normed.get(), gated.get(),
                                    mixed.get(), res1.get(), ffn_in.get(), gate_up.get(),
                                    ffn_act.get(), ffn_down.get(), stream.get() } )
            {
                if ( t )
                    total += occupiedTensorBytes( *t );
            }

            return total;
        }
    };

    /**
     * @brief The width of each workspace slot, one entry per [B, chunk, width] allocation.
     *
     * Shared by the transformer's footprint and read against the allocation below, so the two
     * cannot drift -- the same reason the attention block's widths live in one place.
     */
    export inline std::vector<dim_t> qwenDeltaNetWorkspaceSlotWidths( const QwenConfig& config )
    {
        const dim_t model_dim = config.getModelDim();
        const dim_t hidden_dim = config.getHiddenDimension();
        const dim_t qk_width = config.getDeltaNetQueryKeyWidth();
        const dim_t key_width = config.getDeltaNetKeyWidth();
        const dim_t value_width = config.getDeltaNetValueWidth();
        const dim_t gating_width = config.getDeltaNetGatingWidth();

        // q, k, normed, qk, v, z, a, b, conv_qk, conv_v, act_qk, act_v, core, core_normed, gated,
        // mixed, res1, ffn_in, gate_up, ffn_act, ffn_down, stream -- the order the factory allocates.
        return { key_width, key_width, model_dim, qk_width, value_width, value_width, gating_width, gating_width,
                 qk_width, value_width, qk_width, value_width, value_width, value_width, value_width, model_dim,
                 model_dim, model_dim, 2 * hidden_dim, hidden_dim, model_dim, model_dim };
    }

    /**
     * @brief Allocate the shared DeltaNet-block workspace.
     *
     * Lives beside the struct rather than inside QwenTransformer for the reason its
     * attention counterpart does: a layer-streamed harness drives one block at a time and
     * needs the identical slot geometry, and a second copy of these widths would agree
     * until the first time one of them changed.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    QwenDeltaNetBlockWorkspace<TDeviceType, TPrecision> makeQwenDeltaNetBlockWorkspace(
        const QwenConfig& config, DeviceId device, dim_t B, dim_t prefill_chunk,
        const std::string& name_prefix )
    {
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;

        const dim_t model_dim = config.getModelDim();
        const dim_t hidden_dim = config.getHiddenDimension();
        const dim_t qk_width = config.getDeltaNetQueryKeyWidth();
        const dim_t key_width = config.getDeltaNetKeyWidth();
        const dim_t value_width = config.getDeltaNetValueWidth();
        const dim_t gating_width = config.getDeltaNetGatingWidth();

        auto slot = [&]( dim_t width, const char* name )
        {
            return std::make_shared<TensorType>(
                device, shape_t{ B, prefill_chunk, width }, name_prefix + name );
        };

        QwenDeltaNetBlockWorkspace<TDeviceType, TPrecision> workspace;

        workspace.q = slot( key_width, "q" );
        workspace.k = slot( key_width, "k" );
        workspace.normed = slot( model_dim, "normed" );
        workspace.qk = slot( qk_width, "qk" );
        workspace.v = slot( value_width, "v" );
        workspace.z = slot( value_width, "z" );
        workspace.a = slot( gating_width, "a" );
        workspace.b = slot( gating_width, "b" );
        workspace.conv_qk = slot( qk_width, "conv_qk" );
        workspace.conv_v = slot( value_width, "conv_v" );
        workspace.act_qk = slot( qk_width, "act_qk" );
        workspace.act_v = slot( value_width, "act_v" );
        workspace.core = slot( value_width, "core" );
        workspace.core_normed = slot( value_width, "core_normed" );
        workspace.gated = slot( value_width, "gated" );
        workspace.mixed = slot( model_dim, "mixed" );
        workspace.res1 = slot( model_dim, "res1" );
        workspace.ffn_in = slot( model_dim, "ffn_in" );
        workspace.gate_up = slot( 2 * hidden_dim, "gate_up" );
        workspace.ffn_act = slot( hidden_dim, "ffn_act" );
        workspace.ffn_down = slot( model_dim, "ffn_down" );
        workspace.stream = slot( model_dim, "stream" );

        return workspace;
    }
}
