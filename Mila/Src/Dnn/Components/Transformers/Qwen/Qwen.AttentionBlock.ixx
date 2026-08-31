/**
 * @file Qwen.AttentionBlock.ixx
 * @brief Qwen 3.8 full-attention decoder block (inference: prefill + decode).
 *
 * One of the two block kinds in the 3:1 interleave. The other, QwenDeltaNetBlock, is a
 * separate class rather than a flag on this one: Gemma's two kinds are ONE mixer at two
 * geometries, so a `bool kGlobal` fits it, while Qwen's two kinds are different mixers and
 * share no projection shape (Specifications/Qwen3.8.md section 8, Phase 2 revision).
 *
 * Standard pre-norm decoder, with one delta that has no precedent in the tree:
 *
 *  - `attn_output_gate`. The query projection is DOUBLE width and emits [query | gate];
 *    swish(gate) scales the attention output elementwise before o_proj (section 2). The
 *    gate half never reaches attention -- it bypasses RoPE and the KV cache entirely and
 *    rejoins after the attention output is computed.
 *
 * And four that are ordinary once stated: head_dim 256 is decoupled from the 5120 residual
 * stream (so o_proj is non-square), rotary width is 64 of 256 (partial rotary), query and key
 * carry a per-head RMSNorm before RoPE (q_norm/k_norm, [head_dim]), and the FFN is SwiGLU
 * over a fused gate+up projection.
 *
 * PER-ROLE PRECISION. The block's third parameter is a PLAN, not a policy: its four Linears
 * carry four independently-chosen weight formats, which is the whole reason this model fits
 * a 12 GiB card (section 4 -- no uniform 4-bit path exists). A bare policy still works and
 * means "uniform", because PrecisionPlanFor lifts it.
 *
 *   using QwenAttentionLayer = QwenAttentionBlock<DeviceType::Cuda, TensorDataType::BF16,
 *       QwenPrecisionPlan>;
 *
 * A plan that omits a role this block builds fails the requires-clause below, naming the
 * block -- rather than silently falling back to some default precision.
 *
 * Inference-only, like the Gemma blocks: implements ITransformerBlock's prefill/decode, not
 * Component forward/backward.
 *
 * WEIGHT LAYOUT CONTRACT. `fc_qkv_proj` is fused as [query | gate | key | value], with query
 * and gate as contiguous halves rather than interleaved per head. The converter (section 8
 * item 8) owns producing that ordering; the block's two splits assume it.
 *
 * The checkpoint does NOT store it that way, which is what makes this a contract rather than
 * a description. `Qwen/Qwen3.8-27B` carries three separate tensors -- q_proj [12288, 5120],
 * k_proj and v_proj [1024, 5120] -- and the query projection interleaves the two halves PER
 * HEAD, [q_h0 | gate_h0 | q_h1 | gate_h1 | ...], because the reference views it as
 * [..., heads, 2 * head_dim] and chunks on the last axis. So the converter concatenates the
 * three and de-interleaves the query half; nothing here changes.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <format>
#include <sstream>
#include <stdexcept>
#include <optional>
#include <type_traits>

export module Dnn.Components.QwenAttentionBlock;

export import Dnn.Components.QwenConfig;
export import Dnn.Components.ITransformerBlock;

import Dnn.ITensor;
import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorOps;
import Dnn.ActivationType;
import Dnn.Component;
import Dnn.ComponentType;
import Dnn.CompositeComponent;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.ExecutionContextFactory;
import Compute.GqaState;
import Compute.CpuMemoryResource;
import Compute.Observation;
import Dnn.Components.RmsNorm;
import Dnn.Components.Rope;
import Dnn.Components.Gqa;
import Dnn.Components.Residual;
import Dnn.Components.Linear;
import Dnn.Components.Swiglu;
import Dnn.Components.AttentionOutputGate;
import Serialization.ModelArchive;
import Serialization.Mode;
import Dnn.Quantization.Weight.Policies;
import Dnn.Quantization.Weight.PrecisionPlan;
import Dnn.Quantization.KvCache.Policy;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;
    using namespace Mila::Dnn::Quant::Weight;
    using namespace Mila::Dnn::Quant::KvCache;

    /**
     * @brief Transformer-owned shared activation workspace for QwenAttentionBlock (pooling).
     *
     * One slot set serves every layer: the inference path is strictly sequential, so exactly
     * one block is live at a time. Slots are sized [B, chunk, width]; components view
     * prefixes. The single stream slot is alias-safe -- a block's input is last read at
     * res_1, mid-block, and only overwritten by its own res_2 at block end.
     *
     * The DeltaNet block will need a DIFFERENT workspace (chunk-parallel delta-rule buffers,
     * convolution staging), which is exactly the mismatch section 7 records against
     * ITransformerBlock's concrete GqaState. Keeping this struct per-block-kind rather than
     * per-family is what leaves room for that.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    struct QwenAttentionBlockWorkspace
    {
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;

        // Block-owned split scratch. `gate` is the second half of the query projection and
        // has no counterpart in any other family's block.
        std::shared_ptr<TensorType> q;
        std::shared_ptr<TensorType> gate;
        std::shared_ptr<TensorType> k;
        std::shared_ptr<TensorType> v;

        // Component output slots, one per graph position (prefill order).
        std::shared_ptr<TensorType> normed;      // input_norm out    [B, chunk, model_dim]
        std::shared_ptr<TensorType> qkv;         // fc_qkv_proj out   [B, chunk, packed gated QKV]
        std::shared_ptr<TensorType> query_gate;  // query|gate split  [B, chunk, 2 * q_width]
        std::shared_ptr<TensorType> q_normed;    // q_norm out        [B, chunk, q_width]
        std::shared_ptr<TensorType> k_normed;    // k_norm out        [B, chunk, kv_width]
        std::shared_ptr<TensorType> attn;        // gqa out           [B, chunk, q_width]
        std::shared_ptr<TensorType> gated;       // output gate out   [B, chunk, q_width]
        std::shared_ptr<TensorType> o;           // fc_o_proj out     [B, chunk, model_dim]
        std::shared_ptr<TensorType> res1;        // res_1 out         [B, chunk, model_dim]
        std::shared_ptr<TensorType> ffn_in;      // post_attn_norm    [B, chunk, model_dim]
        std::shared_ptr<TensorType> gate_up;     // fc_gate_up out    [B, chunk, 2 * hidden_dim]
        std::shared_ptr<TensorType> ffn_act;     // swiglu out        [B, chunk, hidden_dim]
        std::shared_ptr<TensorType> ffn_down;    // fc_down out       [B, chunk, model_dim]
        std::shared_ptr<TensorType> stream;      // res_2 out         [B, chunk, model_dim]

        /// Total device bytes held, for memory accounting.
        std::size_t deviceStorageBytes() const
        {
            std::size_t total = 0;

            for ( const auto* t : { q.get(), gate.get(), k.get(), v.get(), normed.get(),
                                    qkv.get(), query_gate.get(), q_normed.get(), k_normed.get(),
                                    attn.get(), gated.get(), o.get(), res1.get(), ffn_in.get(),
                                    gate_up.get(), ffn_act.get(), ffn_down.get(), stream.get() } )
            {
                if ( t )
                    total += t->getStorageSize();
            }

            return total;
        }
    };

    /**
     * @brief Allocate the shared attention-block workspace.
     *
     * Lives here, beside the struct, rather than inside QwenTransformer, because the network
     * is not the only thing that drives these blocks: a layer-streamed parity harness holds
     * one block at a time and needs the identical slot geometry. A second copy of these
     * eighteen widths would agree until the first time one of them changed, and then measure
     * a different model than the one it is checking.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    QwenAttentionBlockWorkspace<TDeviceType, TPrecision> makeQwenAttentionBlockWorkspace(
        const QwenConfig& config, DeviceId device, dim_t B, dim_t prefill_chunk,
        const std::string& name_prefix )
    {
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;

        const dim_t model_dim = config.getModelDim();
        const dim_t hidden_dim = config.getHiddenDimension();
        const dim_t q_width = config.getQProjectionWidth();
        const dim_t kv_width = config.getKVProjectionWidth();
        const dim_t qkv_width = config.getPackedQKVWidth();

        auto slot = [&]( dim_t width, const char* name )
        {
            return std::make_shared<TensorType>(
                device, shape_t{ B, prefill_chunk, width }, name_prefix + name );
        };

        QwenAttentionBlockWorkspace<TDeviceType, TPrecision> workspace;

        workspace.q = slot( q_width, "q" );
        workspace.gate = slot( q_width, "gate" );
        workspace.k = slot( kv_width, "k" );
        workspace.v = slot( kv_width, "v" );
        workspace.normed = slot( model_dim, "normed" );
        workspace.qkv = slot( qkv_width, "qkv" );
        workspace.query_gate = slot( 2 * q_width, "query_gate" );
        workspace.q_normed = slot( q_width, "q_normed" );
        workspace.k_normed = slot( kv_width, "k_normed" );
        workspace.attn = slot( q_width, "attn" );
        workspace.gated = slot( q_width, "gated" );
        workspace.o = slot( model_dim, "o" );
        workspace.res1 = slot( model_dim, "res1" );
        workspace.ffn_in = slot( model_dim, "ffn_in" );
        workspace.gate_up = slot( 2 * hidden_dim, "gate_up" );
        workspace.ffn_act = slot( hidden_dim, "ffn_act" );
        workspace.ffn_down = slot( model_dim, "ffn_down" );
        workspace.stream = slot( model_dim, "stream" );

        return workspace;
    }

    /**
     * @brief The GQA transient the attention blocks share, owned as one unit.
     *
     * Owned together rather than as seven loose tensors so a caller cannot allocate half of
     * it: `GqaState` is a struct of raw pointers, and a null one is a crash at the kernel
     * rather than an error at the call.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    struct QwenGqaWorkspace
    {
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;

        std::unique_ptr<TensorType> q_permute;
        std::unique_ptr<TensorType> preatt;
        std::unique_ptr<TensorType> att;
        std::unique_ptr<TensorType> v_out;
        std::unique_ptr<TensorType> preatt_decode;
        std::unique_ptr<TensorType> att_decode;
        std::unique_ptr<TensorType> v_out_decode;

        GqaState state() const
        {
            GqaState gqa_state;
            gqa_state.q_permute = q_permute.get();
            gqa_state.preatt = preatt.get();
            gqa_state.att = att.get();
            gqa_state.v_out = v_out.get();
            gqa_state.preatt_decode = preatt_decode.get();
            gqa_state.att_decode = att_decode.get();
            gqa_state.v_out_decode = v_out_decode.get();

            return gqa_state;
        }

        std::size_t deviceStorageBytes() const
        {
            std::size_t total = 0;

            for ( const auto* t : { q_permute.get(), preatt.get(), att.get(), v_out.get(),
                                    preatt_decode.get(), att_decode.get(), v_out_decode.get() } )
            {
                if ( t )
                    total += t->getStorageSize();
            }

            return total;
        }
    };

    /**
     * @brief Allocate the shared GQA transient.
     *
     * `score_width` is the caller's flash decision made concrete: the flash path reads no
     * score buffer, so it passes 1, while the cuBLASLt path needs the full context and would
     * overflow a narrow buffer. The parameter exists so the caller cannot allocate for one
     * path and then run the other -- the mismatch the transformer's own comment warns about.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    QwenGqaWorkspace<TDeviceType, TPrecision> makeQwenGqaWorkspace(
        const QwenConfig& config, DeviceId device, dim_t B, dim_t T_ctx,
        dim_t prefill_chunk, dim_t score_width, const std::string& name_prefix )
    {
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;

        const dim_t NH = config.getNumHeads();
        const dim_t HD = config.getHeadDim();

        QwenGqaWorkspace<TDeviceType, TPrecision> workspace;

        workspace.q_permute = std::make_unique<TensorType>(
            device, shape_t{ B, NH, prefill_chunk, HD }, name_prefix + "q_perm" );
        workspace.preatt = std::make_unique<TensorType>(
            device, shape_t{ B, NH, prefill_chunk, score_width }, name_prefix + "preatt" );
        workspace.att = std::make_unique<TensorType>(
            device, shape_t{ B, NH, prefill_chunk, score_width }, name_prefix + "att" );
        workspace.v_out = std::make_unique<TensorType>(
            device, shape_t{ B, NH, prefill_chunk, HD }, name_prefix + "v_out" );

        workspace.preatt_decode = std::make_unique<TensorType>(
            device, shape_t{ B, NH, 1, T_ctx }, name_prefix + "preatt_dec" );
        workspace.att_decode = std::make_unique<TensorType>(
            device, shape_t{ B, NH, 1, T_ctx }, name_prefix + "att_dec" );
        workspace.v_out_decode = std::make_unique<TensorType>(
            device, shape_t{ B, NH, 1, HD }, name_prefix + "v_out_dec" );

        return workspace;
    }

    /**
     * @brief One Qwen 3.8 full-attention decoder block.
     *
     * @tparam TWeightPlan A precision plan (or a bare policy, which lifts to a uniform one).
     * @tparam TKvPolicy   KV-cache compression policy. These layers attend the FULL context,
     *                     so a bounded ring is not applicable to them.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision,
        typename TWeightPlan = NoWeightQuant, KvCachePolicy TKvPolicy = NoKvCompression>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
              && DecoderPrecisionPlan<PrecisionPlanFor<TWeightPlan>>
    class QwenAttentionBlock : public CompositeComponent<TDeviceType, TPrecision>, public ITransformerBlock<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using CompositeComponentBase = CompositeComponent<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;

        /// The resolved plan. Named so a caller can ask a block what it decided.
        using PrecisionPlan = PrecisionPlanFor<TWeightPlan>;

        using RmsNormType = RmsNorm<TDeviceType, TPrecision>;
        using RopeType = Rope<TDeviceType, TPrecision>;
        using AttentionType = GroupedQueryAttention<TDeviceType, TPrecision, TKvPolicy>;
        using ResidualType = Residual<TDeviceType, TPrecision>;
        using SwigluType = Swiglu<TDeviceType, TPrecision, ActivationType::Silu>;

        // SIGMOID, not swish. The published config says `output_gate_type: "swish"`, but that
        // key appears nowhere in the reference implementation, which applies a plain
        // `sigmoid(gate)` to the attention output. The config field is dead; the code is the
        // contract. sigmoid(x) and silu(x) = x * sigmoid(x) differ everywhere but the origin.
        using OutputGateType = AttentionOutputGate<TDeviceType, TPrecision, ActivationType::Sigmoid>;

        // The four per-role Linear types. This block of five lines IS the mechanism the
        // whole precision-plan design exists for: four projections, four independently
        // chosen storage formats, each named by the role it governs.
        using QkvProjectionType = Linear<TDeviceType, TPrecision, typename PrecisionPlan::QkvProjection>;
        using OutputProjectionType = Linear<TDeviceType, TPrecision, typename PrecisionPlan::OutputProjection>;
        using FeedForwardGateUpType = Linear<TDeviceType, TPrecision, typename PrecisionPlan::FeedForwardGateUp>;
        using FeedForwardDownType = Linear<TDeviceType, TPrecision, typename PrecisionPlan::FeedForwardDown>;

        explicit QwenAttentionBlock( const std::string& name, const QwenConfig& config,
            std::optional<DeviceId> device_id = std::nullopt )
            : CompositeComponentBase( name ), config_( config )
        {
            config_.validate();

            if ( !config_.hasAttentionOutputGate() )
            {
                // Not a defect in the config -- a different architecture. Without the gate
                // half the block is plain GQA, which LlamaBlock already is, and silently
                // accepting it here would produce a Qwen-named block that is not one.
                throw std::invalid_argument( std::format(
                    "QwenAttentionBlock '{}': requires attn_output_gate; the ungated geometry "
                    "is plain grouped-query attention", name ) );
            }

            createGraph();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                {
                    throw std::invalid_argument( "QwenAttentionBlock: device type mismatch" );
                }

                owned_exec_context_ = createExecutionContext( device_id.value() );
                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~QwenAttentionBlock() override = default;

        // ====================================================================
        // Geometry
        // ====================================================================

        dim_t headDim() const noexcept { return config_.getHeadDim(); }
        dim_t numKVHeads() const noexcept { return config_.getNumKVHeads(); }
        dim_t qProjWidth() const noexcept { return config_.getQProjectionWidth(); }
        dim_t kvProjWidth() const noexcept { return config_.getKVProjectionWidth(); }
        dim_t packedQKVWidth() const noexcept { return config_.getPackedQKVWidth(); }

        // ====================================================================
        // ITransformerBlock -- inference path
        // ====================================================================

        TensorType& prefill( const TensorType& input, dim_t position_offset ) override
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "QwenAttentionBlock::prefill: must be built before prefill()." );

            const int64_t B = input.shape()[ 0 ];
            const int64_t T = input.shape()[ 1 ];

            auto& output = run( input, B, T, position_offset, /*is_decode*/ false );

            this->publish( ComputePass::Prefill, "output", output );

            return output;
        }

        TensorType& decode( const TensorType& input, dim_t position ) override
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "QwenAttentionBlock::decode: must be built before decode()." );

            const int64_t B = input.shape()[ 0 ];

            auto& output = run( input, B, 1, position, /*is_decode*/ true );

            this->publish( ComputePass::Decode, "output", output );

            return output;
        }

        void setState( const GqaState& state ) override
        {
            if ( attn_ )
                attn_->setState( state );
        }

        /**
         * @brief Route this block's GQA op through the fused FlashAttention prefill kernel.
         *
         * BF16 only. These layers are unbounded, so this is the unbounded kernel; the
         * transformer couples the toggle to the shared score-buffer width.
         */
        void setUseFlashPrefill( bool enabled )
        {
            if ( attn_ )
                attn_->setUseFlashPrefill( enabled );
        }

        /// Route this block's GQA op through the fused decode-attention kernel. BF16 only.
        void setUseFlashDecode( bool enabled )
        {
            if ( attn_ )
                attn_->setUseFlashDecode( enabled );
        }

        bool supportsKvCache() const noexcept override
        {
            return attn_ && attn_->supportsKvCache();
        }

        void resetKvCache() override
        {
            if ( attn_ )
                attn_->resetKvCache();
        }

        bool rewindKvCache( dim_t position ) override
        {
            return attn_ && attn_->rewindKvCache( position );
        }

        /**
         * @brief Install the transformer-owned shared activation workspace (pooling).
         *
         * Must be called before build(); onBuilding then routes each slot into the matching
         * child via installSharedOutput and keeps the split scratch for the block's own
         * views. Self-allocation remains the default for standalone blocks (tests).
         */
        void installSharedWorkspace( const QwenAttentionBlockWorkspace<TDeviceType, TPrecision>& workspace )
        {
            if ( this->isBuilt() )
                throw std::logic_error(
                    "QwenAttentionBlock::installSharedWorkspace: must be called before build()" );

            workspace_ = workspace;
            q_ = workspace_.q;
            gate_ = workspace_.gate;
            k_ = workspace_.k;
            v_ = workspace_.v;
            query_gate_ = workspace_.query_gate;
            workspace_installed_ = true;
        }

        // ====================================================================
        // Component interface
        // ====================================================================

        const ComponentType getType() const override
        {
            return ComponentType::Transformer;
        }

        std::vector<ObservableStage> getObservableStages() const override
        {
            return { { "output", ComputePassMask{ ComputePass::Prefill, ComputePass::Decode } } };
        }

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;

            for ( const auto& child : this->getComponents() )
                stats += child->getMemoryStats();

            // Installed shared workspace tensors are owned and counted once by the
            // transformer (the no-double-count rule).
            if ( !workspace_installed_ )
            {
                for ( auto* t : { q_.get(), gate_.get(), k_.get(), v_.get(), query_gate_.get() } )
                {
                    if ( t )
                        stats.device_state_bytes += t->getStorageSize();
                }
            }

            return stats;
        }

        /**
         * @brief What onBuilding() would allocate for this context, without allocating.
         *
         * Children are named rather than walked: each is built with its OWN context, and a
         * generic recursion over getComponents() would size every one of them against the
         * block's shape. See Specifications/MemoryFootprint.md section 4.4.
         */
        MemoryStats getRequiredMemory( const BuildContext& context ) const override
        {
            validateBuildContext( context );

            const BlockBuildContexts contexts = resolveBlockBuildContexts( context );
            const std::string n = this->getName();

            // workspace_installed_ is still false here: the transformer calls
            // installSharedWorkspace between constructing this block and building it, and
            // states its intent through the context instead. Without this every pooled slot
            // is counted twice.
            const bool pooled = workspace_installed_ || context.hasInstalledOutput();

            auto required = [&]( const auto& component, const BuildContext& child_context )
            {
                return component->getRequiredMemory( child_context.withInstalledOutput( pooled ) );
            };

            MemoryStats stats;

            stats += required( this->template getComponentAs<RmsNormType>( n + ".input_norm" ), contexts.stream );
            stats += required( this->template getComponentAs<QkvProjectionType>( n + ".fc_qkv_proj" ), contexts.stream );
            stats += required( this->template getComponentAs<RmsNormType>( n + ".q_norm" ), contexts.qknorm );
            stats += required( this->template getComponentAs<RmsNormType>( n + ".k_norm" ), contexts.kknorm );
            stats += required( this->template getComponentAs<RopeType>( n + ".rope" ), contexts.qproj );
            stats += required( this->template getComponentAs<AttentionType>( n + ".gqa" ), contexts.qkv );
            stats += required( this->template getComponentAs<OutputGateType>( n + ".output_gate" ), contexts.qproj );
            stats += required( this->template getComponentAs<OutputProjectionType>( n + ".fc_o_proj" ), contexts.qproj );
            stats += required( this->template getComponentAs<ResidualType>( n + ".res_1" ), contexts.stream );
            stats += required( this->template getComponentAs<RmsNormType>( n + ".post_attn_norm" ), contexts.stream );
            stats += required( this->template getComponentAs<FeedForwardGateUpType>( n + ".fc_gate_up" ), contexts.stream );
            stats += required( this->template getComponentAs<SwigluType>( n + ".swiglu" ), contexts.gate_up );
            stats += required( this->template getComponentAs<FeedForwardDownType>( n + ".fc_down" ), contexts.hidden );
            stats += required( this->template getComponentAs<ResidualType>( n + ".res_2" ), contexts.stream );

            // Split scratch. An installed workspace is owned and counted by the transformer.
            if ( !pooled )
            {
                stats.device_state_bytes += storageBytes<TPrecision>( 2 * contexts.splitQElements() );  // query_gate
                stats.device_state_bytes += storageBytes<TPrecision>( contexts.splitQElements() );      // q
                stats.device_state_bytes += storageBytes<TPrecision>( contexts.splitQElements() );      // gate
                stats.device_state_bytes += storageBytes<TPrecision>( contexts.splitKvElements() );     // k
                stats.device_state_bytes += storageBytes<TPrecision>( contexts.splitKvElements() );     // v
            }

            return stats;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "QwenAttentionBlock: " << this->getName() << std::endl;
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << std::endl;
            oss << "Heads: " << config_.getNumHeads() << " q / " << numKVHeads() << " kv, head_dim "
                << headDim() << std::endl;
            oss << "Query projection: " << qProjWidth() << " (projected "
                << config_.getGatedQProjectionWidth() << ", gated)" << std::endl;

            return oss.str();
        }

    protected:

        /**
         * @brief The per-child build contexts and split-scratch geometry this block implies.
         *
         * A block does NOT cascade one context to its children: the fused gate/up projection
         * is double width, and the attention layer is built at the FULL context length so it
         * sizes the KV cache while everything else is built at the prefill chunk. Derived
         * once here so onBuilding() and getRequiredMemory() cannot disagree about any of it.
         */
        struct BlockBuildContexts
        {
            BuildContext stream;
            BuildContext qproj;
            BuildContext gate_up;
            BuildContext hidden;
            BuildContext qkv;
            BuildContext qknorm;
            BuildContext kknorm;

            dim_t batch{ 0 };
            dim_t chunk{ 0 };
            dim_t num_heads{ 0 };
            dim_t num_kv_heads{ 0 };
            dim_t head_dim{ 0 };

            dim_t splitQElements() const noexcept { return batch * chunk * num_heads * head_dim; }
            dim_t splitKvElements() const noexcept { return batch * chunk * num_kv_heads * head_dim; }
        };

        BlockBuildContexts resolveBlockBuildContexts( const BuildContext& context ) const
        {
            const auto& input_shape = context.inputShape();

            BlockBuildContexts contexts;
            contexts.batch = input_shape[ 0 ];
            contexts.num_heads = config_.getNumHeads();
            contexts.num_kv_heads = numKVHeads();
            contexts.head_dim = headDim();
            contexts.chunk = context.getPrefillSize();

            const dim_t model_dim = config_.getModelDim();
            const dim_t hidden_dim = config_.getHiddenDimension();
            const dim_t B = contexts.batch;
            const dim_t chunk = contexts.chunk;

            contexts.stream = context.withShape( shape_t{ B, chunk, model_dim } );
            contexts.qproj = context.withShape( shape_t{ B, chunk, qProjWidth() } );
            contexts.gate_up = context.withShape( shape_t{ B, chunk, 2 * hidden_dim } );
            contexts.hidden = context.withShape( shape_t{ B, chunk, hidden_dim } );

            // QK-norm normalizes over head_dim, so its rows are per-head: [B, chunk * heads, HD].
            contexts.qknorm = context.withShape(
                shape_t{ B, chunk * contexts.num_heads, contexts.head_dim } );
            contexts.kknorm = context.withShape(
                shape_t{ B, chunk * contexts.num_kv_heads, contexts.head_dim } );

            // The GQA op reads only B/T here (its geometry comes from GqaConfig) and
            // validates the trailing dim as its own UNGATED packing -- the gate half is not
            // part of what attention sees.
            contexts.qkv = context.withShape(
                shape_t{ B, input_shape[ 1 ], config_.getAttentionPackedQKVWidth() } );

            return contexts;
        }

        void onBuilding( const BuildContext& context ) override
        {
            validateBuildContext( context );

            const BlockBuildContexts contexts = resolveBlockBuildContexts( context );
            const std::string n = this->getName();

            // With an installed workspace, route each graph position's slot into its
            // component before build so the component skips output self-allocation.
            auto install = [&]( auto& component, const std::shared_ptr<TensorType>& slot )
            {
                if ( workspace_installed_ )
                    component->installSharedOutput( slot );
            };

            input_norm_ = this->template getComponentAs<RmsNormType>( n + ".input_norm" );
            install( input_norm_, workspace_.normed );
            input_norm_->build( contexts.stream );

            qkv_proj_ = this->template getComponentAs<QkvProjectionType>( n + ".fc_qkv_proj" );
            install( qkv_proj_, workspace_.qkv );
            qkv_proj_->build( contexts.stream );

            q_norm_ = this->template getComponentAs<RmsNormType>( n + ".q_norm" );
            install( q_norm_, workspace_.q_normed );
            q_norm_->build( contexts.qknorm );

            k_norm_ = this->template getComponentAs<RmsNormType>( n + ".k_norm" );
            install( k_norm_, workspace_.k_normed );
            k_norm_->build( contexts.kknorm );

            rope_ = this->template getComponentAs<RopeType>( n + ".rope" );
            rope_->build( contexts.qproj );

            attn_ = this->template getComponentAs<AttentionType>( n + ".gqa" );
            install( attn_, workspace_.attn );
            attn_->build( contexts.qkv );

            output_gate_ = this->template getComponentAs<OutputGateType>( n + ".output_gate" );
            install( output_gate_, workspace_.gated );
            output_gate_->build( contexts.qproj );

            o_proj_ = this->template getComponentAs<OutputProjectionType>( n + ".fc_o_proj" );
            install( o_proj_, workspace_.o );
            o_proj_->build( contexts.qproj );

            res1_ = this->template getComponentAs<ResidualType>( n + ".res_1" );
            install( res1_, workspace_.res1 );
            res1_->build( contexts.stream );

            post_attn_norm_ = this->template getComponentAs<RmsNormType>( n + ".post_attn_norm" );
            install( post_attn_norm_, workspace_.ffn_in );
            post_attn_norm_->build( contexts.stream );

            fc_gate_up_ = this->template getComponentAs<FeedForwardGateUpType>( n + ".fc_gate_up" );
            install( fc_gate_up_, workspace_.gate_up );
            fc_gate_up_->build( contexts.stream );

            swiglu_ = this->template getComponentAs<SwigluType>( n + ".swiglu" );
            install( swiglu_, workspace_.ffn_act );
            swiglu_->build( contexts.gate_up );

            fc_down_ = this->template getComponentAs<FeedForwardDownType>( n + ".fc_down" );
            install( fc_down_, workspace_.ffn_down );
            fc_down_->build( contexts.hidden );

            res2_ = this->template getComponentAs<ResidualType>( n + ".res_2" );
            install( res2_, workspace_.stream );
            res2_->build( contexts.stream );

            // Split scratch (sized at the prefill chunk; decode uses T=1 sub-views).
            if ( workspace_installed_ )
            {
                const dim_t q_needed = contexts.splitQElements();
                const dim_t kv_needed = contexts.splitKvElements();

                const bool covered =
                    query_gate_ && query_gate_->size() >= 2 * q_needed
                    && q_ && q_->size() >= q_needed
                    && gate_ && gate_->size() >= q_needed
                    && k_ && k_->size() >= kv_needed
                    && v_ && v_->size() >= kv_needed;

                if ( !covered )
                    throw std::invalid_argument( std::format(
                        "QwenAttentionBlock '{}': installed shared scratch is smaller than the "
                        "block geometry requires", n ) );
            }
            else
            {
                auto device = this->getExecutionContext()->getDeviceId();
                const shape_t q_shape{ contexts.batch, contexts.chunk, qProjWidth() };
                const shape_t kv_shape{ contexts.batch, contexts.chunk, kvProjWidth() };

                query_gate_ = std::make_shared<TensorType>(
                    device, shape_t{ contexts.batch, contexts.chunk, 2 * qProjWidth() }, n + ".query_gate" );
                q_ = std::make_shared<TensorType>( device, q_shape, n + ".q" );
                gate_ = std::make_shared<TensorType>( device, q_shape, n + ".gate" );
                k_ = std::make_shared<TensorType>( device, kv_shape, n + ".k" );
                v_ = std::make_shared<TensorType>( device, kv_shape, n + ".v" );
            }
        }

        void onTrainingModeChanging( TrainingMode training_mode ) override
        {
            // Inference-only block: forward training mode to children for lifecycle
            // consistency, but the block does not implement training forward/backward.
            for ( const auto& child : this->getComponents() )
                child->setTrainingMode( training_mode );
        }

    private:
        QwenConfig config_;
        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };

        std::shared_ptr<RmsNormType> input_norm_{ nullptr };
        std::shared_ptr<QkvProjectionType> qkv_proj_{ nullptr };
        std::shared_ptr<RmsNormType> q_norm_{ nullptr };
        std::shared_ptr<RmsNormType> k_norm_{ nullptr };
        std::shared_ptr<RopeType> rope_{ nullptr };
        std::shared_ptr<AttentionType> attn_{ nullptr };
        std::shared_ptr<OutputGateType> output_gate_{ nullptr };
        std::shared_ptr<OutputProjectionType> o_proj_{ nullptr };
        std::shared_ptr<ResidualType> res1_{ nullptr };
        std::shared_ptr<RmsNormType> post_attn_norm_{ nullptr };
        std::shared_ptr<FeedForwardGateUpType> fc_gate_up_{ nullptr };
        std::shared_ptr<SwigluType> swiglu_{ nullptr };
        std::shared_ptr<FeedForwardDownType> fc_down_{ nullptr };
        std::shared_ptr<ResidualType> res2_{ nullptr };

        // Split scratch: self-allocated at build, or the transformer workspace's set
        // (installSharedWorkspace) which all layers view prefixes of.
        std::shared_ptr<TensorType> query_gate_{ nullptr };
        std::shared_ptr<TensorType> q_{ nullptr };
        std::shared_ptr<TensorType> gate_{ nullptr };
        std::shared_ptr<TensorType> k_{ nullptr };
        std::shared_ptr<TensorType> v_{ nullptr };

        QwenAttentionBlockWorkspace<TDeviceType, TPrecision> workspace_{};
        bool workspace_installed_{ false };

        /**
         * @brief The block's forward path. Prefill and decode differ only in T and in which
         * positional entry point the RoPE and attention components take.
         *
         * Written once rather than twice because -- unlike Gemma, whose K=V global layers
         * derive V from the raw key projection -- there is no structural difference between
         * the two paths here. Two copies of a sequence this long is where a fix lands in one
         * of them; the pair of `is_decode` branches is cheaper than that risk.
         */
        TensorType& run( const TensorType& input, int64_t B, int64_t T, dim_t position, bool is_decode )
        {
            const dim_t NH = config_.getNumHeads();
            const dim_t NKV = numKVHeads();
            const dim_t HD = headDim();

            // --- Attention sub-block ----------------------------------------
            auto& normed = input_norm_->forward( input );
            auto& qkv = qkv_proj_->forward( normed );

            // Two splits, because the projection is [query | gate | key | value] and the
            // primitive takes three outputs. The halves of query_gate cannot be reached as
            // views instead: a row of query_gate is [query row | gate row], so the query
            // rows are strided by 2 * q_width and a flat view would interleave them.
            auto query_gate = query_gate_->view( shape_t{ B, T, 2 * NH * HD }, 0 );
            auto k = k_->view( shape_t{ B, T, NKV * HD }, 0 );
            auto v = v_->view( shape_t{ B, T, NKV * HD }, 0 );

            split( qkv, query_gate, k, v, this->getExecutionContext() );

            auto q = q_->view( shape_t{ B, T, NH * HD }, 0 );
            auto gate = gate_->view( shape_t{ B, T, NH * HD }, 0 );

            split( query_gate, q, gate, this->getExecutionContext() );

            // QK-norm: per-head RMSNorm over head_dim, before RoPE. Viewing [B,T,heads*HD] as
            // [B, T*heads, HD] makes each head's HD vector one normalization group. The gate
            // is NOT normalized -- it never reaches attention.
            auto q_perhead = q.view( shape_t{ B, T * NH, HD }, 0 );
            auto k_perhead = k.view( shape_t{ B, T * NKV, HD }, 0 );
            auto& q_normed = q_norm_->forward( q_perhead );
            auto& k_normed = k_norm_->forward( k_perhead );

            auto q_roped = q_normed.view( shape_t{ B, T, NH * HD }, 0 );
            auto k_roped = k_normed.view( shape_t{ B, T, NKV * HD }, 0 );

            // Partial rotary: 64 of head_dim 256, from RopeConfig::withRotaryDim. Runs in
            // place on the norm outputs, so the raw q_/k_ split buffers stay untouched. The
            // gate half is deliberately absent here -- it carries no position.
            if ( is_decode )
                rope_->decode( q_roped, k_roped, position );
            else
                rope_->prefill( q_roped, k_roped, position );

            auto& attn = is_decode
                ? attn_->decode( q_roped, k_roped, v, position )
                : attn_->prefill( q_roped, k_roped, v, position );

            // The Qwen delta: swish(gate) scales the attention output elementwise before
            // the output projection.
            auto& gated = output_gate_->forward( gate, attn );

            auto& o = o_proj_->forward( gated );

            // res_1 reads the caller's input directly: every op above writes only its own
            // component-owned output, never the previous block's output buffer.
            auto& res1 = res1_->forward( input, o );

            // --- Feed-forward sub-block (SwiGLU) ----------------------------
            auto& ffn_in = post_attn_norm_->forward( res1 );
            auto& gate_up = fc_gate_up_->forward( ffn_in );
            auto& ffn_act = swiglu_->forward( gate_up );
            auto& ffn = fc_down_->forward( ffn_act );

            return res2_->forward( res1, ffn );
        }

        void createGraph()
        {
            const std::string n = this->getName();
            const dim_t model_dim = config_.getModelDim();
            const dim_t hidden_dim = config_.getHiddenDimension();
            const float eps = config_.getRMSNormEpsilon();

            // UNIT OFFSET. Qwen 3.8's RMSNorm scales by (1 + weight) with weights stored
            // zero-centered, the same convention Gemma uses and the opposite of Llama's raw
            // (weight) -- so the default 0.0 offset would silently compute x_norm * w against
            // a checkpoint whose weights sit near zero. Every RmsNorm in this block takes it.
            // The DeltaNet block's gated norm does NOT: that one is genuinely raw.
            auto rms = [&]( const shape_t& shape )
            {
                return RmsNormConfig( shape ).withEpsilon( eps ).withBias( false ).withUnitOffset( 1.0f );
            };

            this->addComponent( std::make_shared<RmsNormType>( n + ".input_norm", rms( shape_t{ model_dim } ) ) );
            this->addComponent( std::make_shared<RmsNormType>( n + ".post_attn_norm", rms( shape_t{ model_dim } ) ) );

            // Fused QKV projection, DOUBLE-width query section: [query | gate | key | value].
            this->addComponent( std::make_shared<QkvProjectionType>(
                n + ".fc_qkv_proj", LinearConfig( model_dim, packedQKVWidth() ).withBias( false ) ) );

            // QK-norm over head_dim, one group per head. Applied before RoPE, and only to
            // query and key -- the gate half bypasses both.
            this->addComponent( std::make_shared<RmsNormType>(
                n + ".q_norm", rms( shape_t{ config_.getHeadDim() } ) ) );
            this->addComponent( std::make_shared<RmsNormType>(
                n + ".k_norm", rms( shape_t{ config_.getHeadDim() } ) ) );

            // RoPE over the query width (not the gated width) -- the gate carries no position.
            auto rope_config = RopeConfig( qProjWidth(), config_.getNumHeads(), numKVHeads(),
                    config_.getMaxSequenceLength() )
                .withBase( config_.getRoPETheta() )
                .withRotaryDim( config_.getRotaryDim() )
                // Qwen confines the rotation to the leading rotary_dim channels, pairs INSIDE
                // that slice, and compresses the frequency spectrum into rotary_dim rather
                // than head_dim. Gemma -- and so the shared default -- does neither. BOTH
                // halves matter: selecting this layout while the cache still spread the
                // spectrum across head_dim made every number worse, because the pairing then
                // disagreed with the frequencies. See Specifications/Qwen3.8.md.
                .withRotaryLayout( RotaryLayout::RotaryPrefix );
            this->addComponent( std::make_shared<RopeType>( n + ".rope", rope_config ) );

            // Full attention: no window. Attention scale stays at the GqaConfig default,
            // which is the 1/sqrt(head_dim) the reference uses.
            auto gqa_config = GqaConfig( qProjWidth(), config_.getNumHeads(), numKVHeads() )
                .withWindow( dim_t{ 0 } );
            this->addComponent( std::make_shared<AttentionType>( n + ".gqa", gqa_config ) );

            this->addComponent( std::make_shared<OutputGateType>(
                n + ".output_gate", AttentionOutputGateConfig() ) );

            // Non-square output projection: (num_heads * head_dim) -> model_dim.
            this->addComponent( std::make_shared<OutputProjectionType>(
                n + ".fc_o_proj", LinearConfig( qProjWidth(), model_dim ).withBias( false ) ) );

            this->addComponent( std::make_shared<ResidualType>( n + ".res_1", ResidualConfig{} ) );

            // SwiGLU FFN: fused gate+up -> Swiglu<Silu> -> down.
            this->addComponent( std::make_shared<FeedForwardGateUpType>(
                n + ".fc_gate_up", LinearConfig( model_dim, 2 * hidden_dim ).withBias( false ) ) );
            this->addComponent( std::make_shared<SwigluType>( n + ".swiglu", SwigluConfig() ) );
            this->addComponent( std::make_shared<FeedForwardDownType>(
                n + ".fc_down", LinearConfig( hidden_dim, model_dim ).withBias( false ) ) );

            this->addComponent( std::make_shared<ResidualType>( n + ".res_2", ResidualConfig{} ) );
        }

        void validateBuildContext( const BuildContext& context ) const
        {
            const auto& s = context.inputShape();

            if ( s.size() != 3 )
                throw std::invalid_argument( std::format(
                    "QwenAttentionBlock: input must be rank 3 [B, T, model_dim], got rank {}", s.size() ) );

            if ( s.back() != config_.getModelDim() )
                throw std::invalid_argument( std::format(
                    "QwenAttentionBlock: model_dim mismatch -- expected {}, got {}",
                    config_.getModelDim(), s.back() ) );
        }
    };
}
