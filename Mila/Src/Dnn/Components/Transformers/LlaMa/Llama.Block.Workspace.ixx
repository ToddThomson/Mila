/**
 * @file Llama.Block.Workspace.ixx
 * @brief Transformer-owned shared activation slots for LlamaBlock (pooling), and their factory.
 *
 * One slot set serves every layer; the transformer owns it and accounts for it.
 */

module;
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

export module Dnn.Components.LlamaTransformer:BlockWorkspace;
import :Config;

import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.DeviceAllocation;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Transformer-owned shared activation workspace for LlamaBlock (pooling).
     *
     * One slot per block-graph position, shared by every layer: the inference path is strictly sequential, so
     * exactly one block is live at a time. Slots are sized [B, chunk, width]; components view prefixes. The single
     * stream slot is alias-safe: a block's input is last read at res_1 (mid-block) and only overwritten by its own
     * res_2 at block end.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    struct LlamaBlockWorkspace
    {
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;

        // Block-owned split scratch: written by split, rotated in place by RoPE, read by attention.
        std::shared_ptr<TensorType> q;
        std::shared_ptr<TensorType> k;
        std::shared_ptr<TensorType> v;

        // Component output slots, one per graph position (prefill order).
        std::shared_ptr<TensorType> normed;    // rmsn_1 out       [B, chunk, model_dim]
        std::shared_ptr<TensorType> qkv;       // fc_qkv_proj out  [B, chunk, packed QKV width]
        std::shared_ptr<TensorType> attn;      // gqa prefill out  [B, chunk, model_dim]
        std::shared_ptr<TensorType> o;         // fc_out_proj out  [B, chunk, model_dim]
        std::shared_ptr<TensorType> res1;      // res_1 out        [B, chunk, model_dim]
        std::shared_ptr<TensorType> ffn_in;    // rmsn_2 out       [B, chunk, model_dim]
        std::shared_ptr<TensorType> gate_up;   // fc_gate_up out   [B, chunk, 2 * hidden_dim]
        std::shared_ptr<TensorType> ffn_act;   // sglu out         [B, chunk, hidden_dim]
        std::shared_ptr<TensorType> ffn_down;  // fc_down out      [B, chunk, model_dim]
        std::shared_ptr<TensorType> stream;    // res_2 out        [B, chunk, model_dim]

        std::size_t deviceStorageBytes() const
        {
            std::size_t total = 0;

            for ( const auto* t : { q.get(), k.get(), v.get(), normed.get(), qkv.get(), attn.get(), o.get(),
                                    res1.get(), ffn_in.get(), gate_up.get(), ffn_act.get(), ffn_down.get(),
                                    stream.get() } )
            {
                if ( t )
                    total += occupiedTensorBytes( *t );
            }

            return total;
        }
    };

    /**
     * @brief The width of each slot makeLlamaBlockWorkspace() allocates, in allocation order, each a separate
     *        allocation of [B, chunk, width].
     *
     * The one statement of the slot geometry: the factory allocates from it and the transformer's footprint prices
     * from it, so the two cannot disagree.
     */
    export std::vector<dim_t> llamaBlockWorkspaceSlotWidths( const LlamaConfig& config )
    {
        const dim_t model_dim = config.getModelDim();
        const dim_t head_dim = model_dim / config.getNumHeads();
        const dim_t kv_width = config.getNumKVHeads() * head_dim;
        const dim_t hidden_dim = config.getHiddenDimension() > 0 ? config.getHiddenDimension() : model_dim * 4;

        return { model_dim, kv_width, kv_width, model_dim, model_dim + 2 * kv_width, model_dim, model_dim, model_dim,
                 model_dim, 2 * hidden_dim, hidden_dim, model_dim, model_dim };
    }

    /**
     * @brief Allocate the shared block workspace at the widths llamaBlockWorkspaceSlotWidths() reports.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    LlamaBlockWorkspace<TDeviceType, TPrecision> makeLlamaBlockWorkspace(
        const LlamaConfig& config, DeviceId device, dim_t B, dim_t prefill_chunk, const std::string& name_prefix )
    {
        using TensorType = typename LlamaBlockWorkspace<TDeviceType, TPrecision>::TensorType;

        const std::vector<dim_t> widths = llamaBlockWorkspaceSlotWidths( config );
        std::size_t next = 0;

        auto slot = [&]( const char* name )
        {
            return std::make_shared<TensorType>( device, shape_t{ B, prefill_chunk, widths.at( next++ ) }, name_prefix + name );
        };

        LlamaBlockWorkspace<TDeviceType, TPrecision> workspace;

        workspace.q = slot( "q" );
        workspace.k = slot( "k" );
        workspace.v = slot( "v" );
        workspace.normed = slot( "normed" );
        workspace.qkv = slot( "qkv" );
        workspace.attn = slot( "attn" );
        workspace.o = slot( "o" );
        workspace.res1 = slot( "res1" );
        workspace.ffn_in = slot( "ffn_in" );
        workspace.gate_up = slot( "gate_up" );
        workspace.ffn_act = slot( "ffn_act" );
        workspace.ffn_down = slot( "ffn_down" );
        workspace.stream = slot( "stream" );

        if ( next != widths.size() )
        {
            throw std::logic_error( "makeLlamaBlockWorkspace: the slots allocated and the widths priced disagree" );
        }

        return workspace;
    }
}
