/**
 * @file Gemma.Block.Workspace.ixx
 * @brief Transformer-owned shared activation slots for GemmaBlock and its feed-forward sublayer (pooling).
 *
 * One slot set serves every layer; the transformer owns it and accounts for it. A module of its own rather than
 * a partition of the block's, because the feed-forward sublayers take their slots from it too.
 */

module;
#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

export module Dnn.Components.GemmaBlockWorkspace;

import Dnn.Tensor;
import Dnn.ITensor;
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

    // The element type of a slot's tensor, which prices it. Only a Tensor has one.
    template<typename TTensor>
    struct SlotDataType;

    template<TensorDataType TDataType, typename TMemoryResource>
    struct SlotDataType<Tensor<TDataType, TMemoryResource>>
    {
        static constexpr TensorDataType value = TDataType;
    };

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
        PackedQkv,
        Experts,
        TopKExperts,
        ExpertGated
    };

    dim_t gemmaSlotWidth( GemmaSlotWidth width, const GemmaConfig& config )
    {
        const dim_t NH = config.getNumHeads();

        switch ( width )
        {
            case GemmaSlotWidth::Experts:
                return config.getNumExperts();

            case GemmaSlotWidth::TopKExperts:
                return config.getTopKExperts();

            // Each selected expert's gated activations, FP32.
            case GemmaSlotWidth::ExpertGated:
                return config.getTopKExperts() * config.getExpertHiddenDimension();

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
        using IndexTensorType = Tensor<TensorDataType::INT32, MR>;
        using ScratchTensorType = Tensor<TensorDataType::FP32, MR>;

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
        std::shared_ptr<TensorType> ffn_in;      // ffn.pre_norm out     [B, chunk, model_dim]
        std::shared_ptr<TensorType> gate_up;     // ffn.mlp.fc_gate_up   [B, chunk, 2 * hidden_dim]
        std::shared_ptr<TensorType> ffn_act;     // ffn.mlp.gate out     [B, chunk, hidden_dim]
        std::shared_ptr<TensorType> ffn_down;    // ffn.mlp.fc_down out  [B, chunk, model_dim]
        std::shared_ptr<TensorType> ffn_normed;  // ffn.post_norm out    [B, chunk, model_dim]
        std::shared_ptr<TensorType> stream;      // res_2 out            [B, chunk, model_dim]

        // Routed feed-forward only; null on a dense model.
        std::shared_ptr<TensorType> ffn_dense_normed;  // ffn.dense_post_norm out    [B, chunk, model_dim]
        std::shared_ptr<TensorType> ffn_router_normed; // ffn.router.norm out        [B, chunk, model_dim]
        std::shared_ptr<TensorType> ffn_router_logits; // ffn.router.proj out        [B, chunk, experts]
        std::shared_ptr<TensorType> ffn_routing_weights;      // ffn.router weights  [B, chunk, top_k]
        std::shared_ptr<IndexTensorType> ffn_routing_indices; // ffn.router indices  [B, chunk, top_k]
        std::shared_ptr<TensorType> ffn_expert_in;     // ffn.experts_pre_norm out   [B, chunk, model_dim]
        std::shared_ptr<ScratchTensorType> ffn_expert_gated;  // ffn.experts gated   [B, chunk, top_k * expert_hidden]
        std::shared_ptr<TensorType> ffn_experts;       // ffn.experts out            [B, chunk, model_dim]
        std::shared_ptr<TensorType> ffn_expert_normed; // ffn.experts_post_norm out  [B, chunk, model_dim]
        std::shared_ptr<TensorType> ffn_sum;           // ffn.sum out                [B, chunk, model_dim]

        /**
         * @brief One slot: its geometry, and the member it fills, bound by slot<>().
         *
         * The member is reached through functions rather than held as a pointer to member, because the slots
         * are not all one tensor type and a member pointer's type names the tensor's.
         */
        struct Slot
        {
            GemmaSlotWidth width;
            const char* name;
            bool routed_only;
            void ( *allocate )( GemmaBlockWorkspace&, DeviceId, const shape_t&, const std::string& );
            const ITensor* ( *tensor )( const GemmaBlockWorkspace& );
            std::size_t ( *occupiedBytes )( dim_t elements, std::size_t granularity );
        };

        // Allocation order.
        static constexpr std::array<Slot, 28> slots()
        {
            return { {
                slot<&GemmaBlockWorkspace::q>( GemmaSlotWidth::Query, "q" ),
                slot<&GemmaBlockWorkspace::k>( GemmaSlotWidth::KeyValue, "k" ),
                slot<&GemmaBlockWorkspace::v>( GemmaSlotWidth::KeyValue, "v" ),
                slot<&GemmaBlockWorkspace::normed>( GemmaSlotWidth::Model, "normed" ),
                slot<&GemmaBlockWorkspace::qkv>( GemmaSlotWidth::PackedQkv, "qkv" ),
                slot<&GemmaBlockWorkspace::q_normed>( GemmaSlotWidth::Query, "q_normed" ),
                slot<&GemmaBlockWorkspace::k_normed>( GemmaSlotWidth::KeyValue, "k_normed" ),
                slot<&GemmaBlockWorkspace::v_normed>( GemmaSlotWidth::KeyValue, "v_normed" ),
                slot<&GemmaBlockWorkspace::attn>( GemmaSlotWidth::Query, "attn" ),
                slot<&GemmaBlockWorkspace::o>( GemmaSlotWidth::Model, "o" ),
                slot<&GemmaBlockWorkspace::o_normed>( GemmaSlotWidth::Model, "o_normed" ),
                slot<&GemmaBlockWorkspace::res1>( GemmaSlotWidth::Model, "res1" ),
                slot<&GemmaBlockWorkspace::ffn_in>( GemmaSlotWidth::Model, "ffn_in" ),
                slot<&GemmaBlockWorkspace::gate_up>( GemmaSlotWidth::GateUp, "gate_up" ),
                slot<&GemmaBlockWorkspace::ffn_act>( GemmaSlotWidth::Hidden, "ffn_act" ),
                slot<&GemmaBlockWorkspace::ffn_down>( GemmaSlotWidth::Model, "ffn_down" ),
                slot<&GemmaBlockWorkspace::ffn_normed>( GemmaSlotWidth::Model, "ffn_normed" ),
                slot<&GemmaBlockWorkspace::stream>( GemmaSlotWidth::Model, "stream" ),
                slot<&GemmaBlockWorkspace::ffn_dense_normed>( GemmaSlotWidth::Model, "ffn_dense_normed", true ),
                slot<&GemmaBlockWorkspace::ffn_router_normed>( GemmaSlotWidth::Model, "ffn_router_normed", true ),
                slot<&GemmaBlockWorkspace::ffn_router_logits>( GemmaSlotWidth::Experts, "ffn_router_logits", true ),
                slot<&GemmaBlockWorkspace::ffn_routing_weights>( GemmaSlotWidth::TopKExperts, "ffn_routing_weights", true ),
                slot<&GemmaBlockWorkspace::ffn_routing_indices>( GemmaSlotWidth::TopKExperts, "ffn_routing_indices", true ),
                slot<&GemmaBlockWorkspace::ffn_expert_in>( GemmaSlotWidth::Model, "ffn_expert_in", true ),
                slot<&GemmaBlockWorkspace::ffn_expert_gated>( GemmaSlotWidth::ExpertGated, "ffn_expert_gated", true ),
                slot<&GemmaBlockWorkspace::ffn_experts>( GemmaSlotWidth::Model, "ffn_experts", true ),
                slot<&GemmaBlockWorkspace::ffn_expert_normed>( GemmaSlotWidth::Model, "ffn_expert_normed", true ),
                slot<&GemmaBlockWorkspace::ffn_sum>( GemmaSlotWidth::Model, "ffn_sum", true ),
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
                    bytes += slot.occupiedBytes( B * prefill_chunk * gemmaSlotWidth( slot.width, config ), granularity );
                }
            }

            return bytes;
        }

        std::size_t deviceStorageBytes() const
        {
            std::size_t total = 0;

            for ( const Slot& slot : slots() )
            {
                if ( const ITensor* tensor = slot.tensor( *this ) )
                    total += occupiedTensorBytes( *tensor );
            }

            return total;
        }

    private:

        template<auto TMember>
        static constexpr Slot slot( GemmaSlotWidth width, const char* name, bool routed_only = false )
        {
            using SlotTensorType =
                typename std::remove_cvref_t<decltype( std::declval<GemmaBlockWorkspace&>().*TMember )>::element_type;

            return {
                width, name, routed_only,
                []( GemmaBlockWorkspace& workspace, DeviceId device, const shape_t& shape, const std::string& tensor_name )
                {
                    workspace.*TMember = std::make_shared<SlotTensorType>( device, shape, tensor_name );
                },
                []( const GemmaBlockWorkspace& workspace ) -> const ITensor*
                {
                    return ( workspace.*TMember ).get();
                },
                []( dim_t elements, std::size_t granularity )
                {
                    return occupiedDeviceBytes( storageBytes<SlotDataType<SlotTensorType>::value>( elements ), granularity );
                } };
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

        Workspace workspace;

        for ( const auto& slot : Workspace::slots() )
        {
            if ( Workspace::allocates( slot, config ) )
            {
                slot.allocate(
                    workspace, device, shape_t{ B, prefill_chunk, gemmaSlotWidth( slot.width, config ) }, name_prefix + slot.name );
            }
        }

        return workspace;
    }
}
