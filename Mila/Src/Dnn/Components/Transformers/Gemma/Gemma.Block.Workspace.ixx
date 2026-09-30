/**
 * @file Gemma.Block.Workspace.ixx
 * @brief Transformer-owned shared activation slots for GemmaBlock (pooling), and their factory.
 *
 * One slot set serves every layer; the transformer owns it and accounts for it.
 */

module;
#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <string>

export module Dnn.Components.GemmaBlock:Workspace;

import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.Component;
import Dnn.Components.GemmaConfig;
import Compute.DeviceAllocation;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief The geometry a slot is as wide as, taken at the wider of the two layer kinds.
     */
    enum class GemmaSlotWidth
    {
        Model,
        Hidden,
        GateUp,
        Query,
        KeyValue,
        PackedQkv
    };

    dim_t gemmaSlotWidth( GemmaSlotWidth width, const GemmaConfig& config )
    {
        const dim_t NH = config.getNumHeads();

        switch ( width )
        {
            case GemmaSlotWidth::Hidden:
                return config.getHiddenDimension();

            case GemmaSlotWidth::GateUp:
                return 2 * config.getHiddenDimension();

            case GemmaSlotWidth::Query:
                return NH * std::max( config.getHeadDim(), config.getGlobalHeadDim() );

            case GemmaSlotWidth::KeyValue:
                return std::max(
                    config.getNumKVHeads() * config.getHeadDim(),
                    config.getNumGlobalKVHeads() * config.getGlobalHeadDim() );

            case GemmaSlotWidth::PackedQkv:
            {
                // Global K=V layers drop the V section.
                const dim_t packed_local = ( NH + 2 * config.getNumKVHeads() ) * config.getHeadDim();
                const dim_t packed_global =
                    ( NH + ( config.keyEqualsValue() ? 1 : 2 ) * config.getNumGlobalKVHeads() ) * config.getGlobalHeadDim();

                return std::max( packed_local, packed_global );
            }

            case GemmaSlotWidth::Model:
            default:
                return config.getModelDim();
        }
    }

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
     *
     * slots() is the one list of slots. The allocation, the bytes a built workspace holds and the bytes one
     * would take are all read from it, so a slot cannot be allocated without being priced.
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

        // Routed feed-forward only; null on a dense model.
        std::shared_ptr<TensorType> ffn_dense_normed;  // post_ffn_norm_1 out  [B, chunk, model_dim]
        std::shared_ptr<TensorType> ffn_expert_in;     // pre_ffn_norm_2 out   [B, chunk, model_dim]
        std::shared_ptr<TensorType> ffn_expert_normed; // post_ffn_norm_2 out  [B, chunk, model_dim]
        std::shared_ptr<TensorType> ffn_sum;           // ffn_sum out          [B, chunk, model_dim]

        struct Slot
        {
            std::shared_ptr<TensorType> GemmaBlockWorkspace::* member;
            GemmaSlotWidth width;
            const char* name;
            bool routed_only;
        };

        // Allocation order.
        static constexpr std::array<Slot, 22> slots()
        {
            return { {
                { &GemmaBlockWorkspace::q, GemmaSlotWidth::Query, "q", false },
                { &GemmaBlockWorkspace::k, GemmaSlotWidth::KeyValue, "k", false },
                { &GemmaBlockWorkspace::v, GemmaSlotWidth::KeyValue, "v", false },
                { &GemmaBlockWorkspace::normed, GemmaSlotWidth::Model, "normed", false },
                { &GemmaBlockWorkspace::qkv, GemmaSlotWidth::PackedQkv, "qkv", false },
                { &GemmaBlockWorkspace::q_normed, GemmaSlotWidth::Query, "q_normed", false },
                { &GemmaBlockWorkspace::k_normed, GemmaSlotWidth::KeyValue, "k_normed", false },
                { &GemmaBlockWorkspace::v_normed, GemmaSlotWidth::KeyValue, "v_normed", false },
                { &GemmaBlockWorkspace::attn, GemmaSlotWidth::Query, "attn", false },
                { &GemmaBlockWorkspace::o, GemmaSlotWidth::Model, "o", false },
                { &GemmaBlockWorkspace::o_normed, GemmaSlotWidth::Model, "o_normed", false },
                { &GemmaBlockWorkspace::res1, GemmaSlotWidth::Model, "res1", false },
                { &GemmaBlockWorkspace::ffn_in, GemmaSlotWidth::Model, "ffn_in", false },
                { &GemmaBlockWorkspace::gate_up, GemmaSlotWidth::GateUp, "gate_up", false },
                { &GemmaBlockWorkspace::ffn_act, GemmaSlotWidth::Hidden, "ffn_act", false },
                { &GemmaBlockWorkspace::ffn_down, GemmaSlotWidth::Model, "ffn_down", false },
                { &GemmaBlockWorkspace::ffn_normed, GemmaSlotWidth::Model, "ffn_normed", false },
                { &GemmaBlockWorkspace::stream, GemmaSlotWidth::Model, "stream", false },
                { &GemmaBlockWorkspace::ffn_dense_normed, GemmaSlotWidth::Model, "ffn_dense_normed", true },
                { &GemmaBlockWorkspace::ffn_expert_in, GemmaSlotWidth::Model, "ffn_expert_in", true },
                { &GemmaBlockWorkspace::ffn_expert_normed, GemmaSlotWidth::Model, "ffn_expert_normed", true },
                { &GemmaBlockWorkspace::ffn_sum, GemmaSlotWidth::Model, "ffn_sum", true },
            } };
        }

        static bool allocates( const Slot& slot, const GemmaConfig& config ) noexcept
        {
            return !slot.routed_only || config.hasMixtureOfExperts();
        }

        /**
         * @brief Bytes the workspace for this config would occupy, allocating nothing.
         */
        static std::size_t requiredBytes( const GemmaConfig& config, dim_t B, dim_t prefill_chunk, std::size_t granularity )
        {
            std::size_t bytes = 0;

            for ( const Slot& slot : slots() )
            {
                if ( allocates( slot, config ) )
                {
                    const dim_t elements = B * prefill_chunk * gemmaSlotWidth( slot.width, config );

                    bytes += occupiedDeviceBytes( storageBytes<TPrecision>( elements ), granularity );
                }
            }

            return bytes;
        }

        std::size_t deviceStorageBytes() const
        {
            std::size_t total = 0;

            for ( const Slot& slot : slots() )
            {
                if ( const auto& tensor = this->*slot.member )
                    total += occupiedTensorBytes( *tensor );
            }

            return total;
        }
    };

    /**
     * @brief Allocate the shared block workspace, one tensor per slot the config needs.
     *
     * Lives beside the struct because the transformer is not the only caller: a layer-streamed parity
     * harness holds one block at a time and must measure the geometry the model builds, not a copy of it.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    GemmaBlockWorkspace<TDeviceType, TPrecision> makeGemmaBlockWorkspace(
        const GemmaConfig& config, DeviceId device, dim_t B, dim_t prefill_chunk, const std::string& name_prefix )
    {
        using Workspace = GemmaBlockWorkspace<TDeviceType, TPrecision>;
        using TensorType = typename Workspace::TensorType;

        Workspace workspace;

        for ( const auto& slot : Workspace::slots() )
        {
            if ( Workspace::allocates( slot, config ) )
            {
                workspace.*slot.member = std::make_shared<TensorType>(
                    device, shape_t{ B, prefill_chunk, gemmaSlotWidth( slot.width, config ) }, name_prefix + slot.name );
            }
        }

        return workspace;
    }
}
