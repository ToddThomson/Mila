/**
 * @file Gemma.Block.ixx
 * @brief Gemma 4 decoder block (inference: prefill + decode), one type per attention kind.
 *
 * Modeled on LlamaBlock, with the Gemma 4 deltas:
 *  - Sandwich norm (4 RMSNorms): input -> attn -> post_attn -> +res -> pre_ffn -> ffn -> post_ffn -> +res.
 *  - QK-norm: per-head RMSNorm over head_dim on Q and K, applied BEFORE RoPE.
 *  - GeGLU FFN (Swiglu<..., Gelu>), decoupled head_dim, non-square o_proj.
 *  - Per-layer geometry via the compile-time kGlobal flag: global layers use
 *    global_head_dim / a single shared KV head / K=V (no separate v_proj) / full attention /
 *    proportional partial-rotary; sliding layers use head_dim / num_kv_heads / window / full rotation.
 *  - V is per-head normalized (v_norm, no learnable scale, no RoPE) on every layer: sliding
 *    layers normalize the separate V projection; global K=V layers derive V from the RAW key
 *    projection -- V = v_norm(k_proj), distinct from K = RoPE(k_norm(k_proj)).
 *  - Attention scale 1.0 (QK-norm controls magnitude; GqaConfig::withAttentionScale).
 *
 * Inference-only (Gemma is an inference target): implements IDecoderLayer's prefill/decode;
 * no training forward/backward. Gemma RMSNorm is x_norm * (1 + weight) (HF Gemma3RMSNorm): the
 * converter writes the weights RAW (zero-centered) and the +1 is applied at the kernel via
 * RmsNormConfig::withUnitOffset(1.0) on every norm -- so the stored weights stay identical to the
 * source checkpoint and the shared RmsNorm kernel stays Llama-safe (offset 0 = raw).
 *
 * HF reference forward order (Gemma4TextDecoderLayer):
 *   res0 = x
 *   a = self_attn( input_layernorm(x) )         [qkv_proj, q_norm/k_norm, RoPE, GQA, o_proj]
 *   x = res0 + post_attention_layernorm(a)
 *   res1 = x
 *   f = mlp( pre_feedforward_layernorm(x) )      [GeGLU]
 *   x = res1 + post_feedforward_layernorm(f)
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

export module Dnn.Components.GemmaBlock;

export import Dnn.Components.GemmaConfig;
export import Dnn.Components.IDecoderLayer;

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
#ifdef MILA_HAS_CUDA
import Compute.CudaPinnedMemoryResource;
#endif
import Dnn.Components.RmsNorm;
import Dnn.Components.Rope;
import Dnn.Components.Gqa;
import Dnn.Components.Residual;
import Dnn.Components.Linear;
import Dnn.Components.Swiglu;
import Serialization.ModelArchive;
import Serialization.Mode;
import Serialization.Tensor;
import Dnn.Quantization.Weight.Policies;
import Dnn.Quantization.KvCache.Policy;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;
    using namespace Mila::Dnn::Quant::Weight;
    using namespace Mila::Dnn::Quant::KvCache;

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

    /**
     * @brief One Gemma 4 decoder block; kGlobal selects the global (full-attention) geometry.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision, bool kGlobal,
        WeightQuantPolicy TWeightQuant = NoWeightQuant, KvCachePolicy TKvPolicy = NoKvCompression>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class GemmaBlock
        : public CompositeComponent<TDeviceType, TPrecision>,
          public IDecoderLayer<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using CompositeComponentBase = CompositeComponent<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using RmsNormType = RmsNorm<TDeviceType, TPrecision>;
        using RopeType = Rope<TDeviceType, TPrecision>;
        using AttentionType = GroupedQueryAttention<TDeviceType, TPrecision, TKvPolicy>;
        using ResidualType = Residual<TDeviceType, TPrecision>;
        using LinearType = Linear<TDeviceType, TPrecision, TWeightQuant>;
        using GeGLUType = Swiglu<TDeviceType, TPrecision, ActivationType::Gelu>;

        explicit GemmaBlock( const std::string& name, const GemmaConfig& config, std::optional<DeviceId> device_id = std::nullopt )
            : CompositeComponentBase( name ), config_( config )
        {
            config_.validate();

            createGraph();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                {
                    throw std::invalid_argument( "GemmaBlock: device type mismatch" );
                }

                owned_exec_context_ = createExecutionContext( device_id.value() );
                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~GemmaBlock() override = default;

        // ====================================================================
        // Compile-time per-layer geometry (kGlobal selects local vs global)
        // ====================================================================

        static constexpr bool isGlobal() noexcept { return kGlobal; }

        dim_t headDim() const noexcept { return kGlobal ? config_.getGlobalHeadDim() : config_.getHeadDim(); }
        dim_t numKVHeads() const noexcept { return kGlobal ? config_.getNumGlobalKVHeads() : config_.getNumKVHeads(); }
        bool keyEqualsValue() const noexcept { return kGlobal ? config_.keyEqualsValue() : false; }
        dim_t window() const noexcept { return kGlobal ? dim_t{ 0 } : config_.getWindow(); }
        float ropeTheta() const noexcept { return kGlobal ? config_.getRoPEThetaGlobal() : config_.getRoPEThetaLocal(); }
        dim_t rotaryDim() const noexcept { return kGlobal ? config_.getGlobalRotaryDim() : dim_t{ 0 }; }
        dim_t qProjWidth() const noexcept { return config_.getNumHeads() * headDim(); }
        dim_t kvProjWidth() const noexcept { return numKVHeads() * headDim(); }
        dim_t packedQKVWidth() const noexcept
        {
            const dim_t kv_sections = keyEqualsValue() ? 1 : 2;
            return (config_.getNumHeads() + kv_sections * numKVHeads()) * headDim();
        }

        // ====================================================================
        // IDecoderLayer -- inference path
        // ====================================================================

        TensorType& prefill( const TensorType& input, dim_t position_offset ) override
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "GemmaBlock::prefill: must be built before prefill()." );

            const int64_t B = input.shape()[ 0 ];
            const int64_t T = input.shape()[ 1 ];
            const dim_t NH = config_.getNumHeads();
            const dim_t NKV = numKVHeads();
            const dim_t HD = headDim();

            // --- Attention sub-block ----------------------------------------
            auto& normed = input_norm_->forward( input );
            auto& qkv = qkv_proj_->forward( normed );

            // Split fused QKV into per-projection contiguous buffers.
            auto q = q_->view( shape_t{ B, T, NH * HD }, 0 );
            auto k = k_->view( shape_t{ B, T, NKV * HD }, 0 );

            if constexpr ( kGlobal )
            {
                // K=V: the fused projection is [Q | K] (2-way split). V is derived from
                // this same K projection via v_norm at the attention step (see below).
                split( qkv, q, k, this->getExecutionContext() );
            }
            else
            {
                auto v = v_->view( shape_t{ B, T, NKV * HD }, 0 );
                split( qkv, q, k, v, this->getExecutionContext() );
            }

            // QK-norm: per-head RMSNorm over head_dim. View [B,T,heads*HD] as
            // [B, T*heads, HD] so each head's HD vector is one normalization group.
            auto q_perhead = q.view( shape_t{ B, T * NH, HD }, 0 );
            auto k_perhead = k.view( shape_t{ B, T * NKV, HD }, 0 );
            auto& q_normed = q_norm_->forward( q_perhead );
            auto& k_normed = k_norm_->forward( k_perhead );

            auto q_roped = q_normed.view( shape_t{ B, T, NH * HD }, 0 );
            auto k_roped = k_normed.view( shape_t{ B, T, NKV * HD }, 0 );

            rope_->prefill( q_roped, k_roped, position_offset );

            // GQA: V is per-head normalized (v_norm, no learnable scale, no RoPE) before
            // attention -- distinct from K = RoPE(k_norm(k_proj)). Global K=V layers have no
            // separate v_proj, so V is derived from the RAW key projection.
            TensorType* attn_ptr = nullptr;
            if constexpr ( kGlobal )
            {
                // The raw k_proj still lives in the k_ buffer: k_norm wrote to its own output
                // and RoPE ran in-place on THAT, so k_ is untouched. V = v_norm(k_proj).
                auto k_raw_perhead = k.view( shape_t{ B, T * NKV, HD }, 0 );
                auto& v_normed = v_norm_->forward( k_raw_perhead );
                auto v_view = v_normed.view( shape_t{ B, T, NKV * HD }, 0 );
                attn_ptr = &attn_->prefill( q_roped, k_roped, v_view, position_offset );
            }
            else
            {
                auto v_perhead = v_->view( shape_t{ B, T * NKV, HD }, 0 );
                auto& v_normed = v_norm_->forward( v_perhead );
                auto v_view = v_normed.view( shape_t{ B, T, NKV * HD }, 0 );
                attn_ptr = &attn_->prefill( q_roped, k_roped, v_view, position_offset );
            }
            auto& attn = *attn_ptr;

            auto& o = o_proj_->forward( attn );
            auto& o_normed = post_attn_norm_->forward( o );

            // res_1 reads the caller's input directly (exactly as decode() does): every
            // op above writes only its own component-owned output, never the previous
            // block's output buffer, so the former defensive res0 copy was redundant --
            // one launch + one full stream read/write per layer per chunk. HF-greedy
            // parity pins this.
            auto& res1 = res1_->forward( input, o_normed );

            // --- Feed-forward sub-block (GeGLU) -----------------------------
            auto& ffn_in = pre_ffn_norm_->forward( res1 );
            auto& gate_up = fc_gate_up_->forward( ffn_in );
            auto& ffn_act = geglu_->forward( gate_up );
            auto& ffn = fc_down_->forward( ffn_act );
            auto& ffn_normed = post_ffn_norm_->forward( ffn );
            auto& res2 = res2_->forward( res1, ffn_normed );

            // Gemma 4 Unified: per-layer learned output scale, hidden_states *= layer_scalar.
            // Default 1.0 (identity) until loaded; without it the residual stream blows up ~18x.
            scale( res2, layer_scalar_, res2, this->getExecutionContext() );

            return res2;
        }

        TensorType& decode( const TensorType& input, dim_t position ) override
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "GemmaBlock::decode: must be built before decode()." );

            const int64_t B = input.shape()[ 0 ];
            const dim_t model_dim = config_.getModelDim();
            const dim_t NH = config_.getNumHeads();
            const dim_t NKV = numKVHeads();
            const dim_t HD = headDim();

            auto& normed = input_norm_->forward( input );
            auto& qkv = qkv_proj_->forward( normed );

            auto q = q_->view( shape_t{ B, 1, NH * HD }, 0 );
            auto k = k_->view( shape_t{ B, 1, NKV * HD }, 0 );

            if constexpr ( kGlobal )
            {
                split( qkv, q, k, this->getExecutionContext() );
            }
            else
            {
                auto v = v_->view( shape_t{ B, 1, NKV * HD }, 0 );
                split( qkv, q, k, v, this->getExecutionContext() );
            }

            auto q_perhead = q.view( shape_t{ B, NH, HD }, 0 );
            auto k_perhead = k.view( shape_t{ B, NKV, HD }, 0 );
            auto& q_normed = q_norm_->forward( q_perhead );
            auto& k_normed = k_norm_->forward( k_perhead );

            auto q_roped = q_normed.view( shape_t{ B, 1, NH * HD }, 0 );
            auto k_roped = k_normed.view( shape_t{ B, 1, NKV * HD }, 0 );

            rope_->decode( q_roped, k_roped, position );

            // Per-head V normalization (v_norm, no learnable scale, no RoPE); global K=V
            // layers derive V from the raw k_proj still held in the k_ buffer (see prefill).
            TensorType* attn_ptr = nullptr;
            if constexpr ( kGlobal )
            {
                auto k_raw_perhead = k.view( shape_t{ B, NKV, HD }, 0 );
                auto& v_normed = v_norm_->forward( k_raw_perhead );
                auto v_view = v_normed.view( shape_t{ B, 1, NKV * HD }, 0 );
                attn_ptr = &attn_->decode( q_roped, k_roped, v_view, position );
            }
            else
            {
                auto v_perhead = v_->view( shape_t{ B, NKV, HD }, 0 );
                auto& v_normed = v_norm_->forward( v_perhead );
                auto v_view = v_normed.view( shape_t{ B, 1, NKV * HD }, 0 );
                attn_ptr = &attn_->decode( q_roped, k_roped, v_view, position );
            }
            auto& attn = *attn_ptr;

            auto& o = o_proj_->forward( attn );
            auto& o_normed = post_attn_norm_->forward( o );
            auto& res1 = res1_->forward( input, o_normed );

            auto& ffn_in = pre_ffn_norm_->forward( res1 );
            auto& gate_up = fc_gate_up_->forward( ffn_in );
            auto& ffn_act = geglu_->forward( gate_up );
            auto& ffn = fc_down_->forward( ffn_act );
            auto& ffn_normed = post_ffn_norm_->forward( ffn );
            auto& res2 = res2_->forward( res1, ffn_normed );

            // Gemma 4 Unified per-layer learned output scale (see prefill).
            scale( res2, layer_scalar_, res2, this->getExecutionContext() );

            return res2;
        }

        void setState( const GqaState& state ) override
        {
            if ( attn_ )
                attn_->setState( state );
        }

        // Route this block's GQA op through the fused FlashAttention prefill kernel --
        // the unbounded kernel on global layers, the ring variant on local (sliding)
        // layers. BF16 only; the transformer couples it to the shared preatt/att
        // workspace width (GqaFlashAttention.md 5.6).
        void setUseFlashPrefill( bool enabled )
        {
            if ( attn_ )
                attn_->setUseFlashPrefill( enabled );
        }

        // Route this block's GQA op through the fused decode-attention kernel.
        // BF16 only; unlike prefill there is no context threshold and no workspace
        // coupling -- the fused decode is band-limited and unconditionally cheaper.
        void setUseFlashDecode( bool enabled )
        {
            if ( attn_ )
                attn_->setUseFlashDecode( enabled );
        }

        bool supportsKVCache() const noexcept override
        {
            return attn_ && attn_->supportsKVCache();
        }

        void resetKVCache() override
        {
            if ( attn_ )
                attn_->resetKVCache();
        }

        bool rewindKvCache( dim_t position ) override
        {
            return attn_ && attn_->rewindKvCache( position );
        }

        /**
         * @brief Install the transformer-owned shared activation workspace (pooling).
         *
         * One workspace serves every layer: the inference path is strictly
         * sequential, so exactly one block is live at a time. Must be called
         * before build(); onBuilding then routes each slot into the matching
         * child component via installSharedOutput (and keeps the q/k/v split
         * scratch for the block's own views), skipping all per-layer output
         * self-allocation. Self-allocation remains the default for standalone
         * blocks (tests).
         */
        void installSharedWorkspace( const GemmaBlockWorkspace<TDeviceType, TPrecision>& workspace )
        {
            if ( this->isBuilt() )
                throw std::logic_error(
                    "GemmaBlock::installSharedWorkspace: must be called before build()" );

            workspace_ = workspace;
            q_ = workspace_.q;
            k_ = workspace_.k;
            v_ = workspace_.v;
            workspace_installed_ = true;
        }

        // ====================================================================
        // Component interface
        // ====================================================================

        const ComponentType getType() const override
        {
            return ComponentType::Transformer;
        }

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;

            for ( const auto& child : this->getComponents() )
                stats += child->getMemoryStats();

            // Installed shared workspace tensors are owned and counted once by the
            // transformer (the D7 no-double-count rule, same shape as the tied-weight
            // correction); the child components apply the same rule to their slots.
            if ( !workspace_installed_ )
            {
                for ( auto* t : { q_.get(), k_.get(), v_.get() } )
                {
                    if ( t )
                        stats.device_state_bytes += t->getStorageSize();
                }
            }

            return stats;
        }

        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            for ( const auto& child : this->getComponents() )
                child->save_( archive, mode );
        }

        /**
         * @brief Load the block's own parameters. Sub-component weights are routed by the
         * reader directly to the sub-components; the only parameter owned by the block itself
         * is the Gemma 4 Unified per-layer output scale `layer_scalar` (a [1] FP32 value).
         */
        void loadParameter( const std::string& name, const ITensorBlob& blob ) override
        {
            if ( name == "layer_scalar" )
            {
                Tensor<TensorDataType::FP32, MR> buf(
                    this->getDeviceId(), shape_t{ 1 }, this->getName() + ".layer_scalar" );
                this->template loadParameterFromBlob<TensorDataType::FP32, MR>(
                    "layer_scalar", blob, buf, shape_t{ 1 } );

                // Read the single value to a host float for the per-forward output scale.
                Tensor<TensorDataType::FP32, HostStagingMR> host(
                    TDeviceType == DeviceType::Cuda ? this->getDeviceId() : Device::Cpu(), shape_t{ 1 } );
                copy( buf, host );
                this->getExecutionContext()->synchronize();
                layer_scalar_ = host.data()[ 0 ];
            }
            else
            {
                CompositeComponentBase::loadParameter( name, blob );
            }
        }

    protected:

        void onBuilding( const BuildContext& context ) override
        {
            validateBuildContext( context );

            const auto& input_shape = context.inputShape();
            const int64_t B = input_shape[ 0 ];
            const dim_t model_dim = config_.getModelDim();
            const dim_t hidden_dim = config_.getHiddenDimension();
            const dim_t NH = config_.getNumHeads();
            const dim_t NKV = numKVHeads();
            const dim_t HD = headDim();

            // Inference: non-attention components are built at the prefill chunk size;
            // the GQA layer is built at full context length so it sizes the KV cache.
            const int64_t chunk = context.getPrefillSize();

            const shape_t stream_shape = { B, chunk, model_dim };
            const shape_t qproj_shape = { B, chunk, NH * HD };
            const shape_t qknorm_shape = { B, chunk * NH, HD };   // QK-norm sees per-head rows
            const shape_t kknorm_shape = { B, chunk * NKV, HD };
            const shape_t gate_up_shape = { B, chunk, 2 * hidden_dim };
            const shape_t hidden_shape = { B, chunk, hidden_dim };
            // GQA build context: the op only reads B/T here (its geometry comes from
            // GqaConfig) and validates the trailing dim as the STANDARD (NH + 2*NKV)*HD
            // packing -- not the K=V-aware block packing -- so use that here regardless of kGlobal.
            const shape_t qkv_ctx_shape =
                { B, input_shape[ 1 ], (config_.getNumHeads() + 2 * NKV) * HD };

            BuildContext stream_ctx = context.withShape( stream_shape );
            BuildContext qproj_ctx = context.withShape( qproj_shape );
            BuildContext qknorm_ctx = context.withShape( qknorm_shape );
            BuildContext kknorm_ctx = context.withShape( kknorm_shape );
            BuildContext gate_up_ctx = context.withShape( gate_up_shape );
            BuildContext hidden_ctx = context.withShape( hidden_shape );
            BuildContext qkv_ctx = context.withShape( qkv_ctx_shape );

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
            input_norm_->build( stream_ctx );

            qkv_proj_ = this->template getComponentAs<LinearType>( n + ".qkv_proj" );
            install( qkv_proj_, workspace_.qkv );
            qkv_proj_->build( stream_ctx );

            q_norm_ = this->template getComponentAs<RmsNormType>( n + ".q_norm" );
            install( q_norm_, workspace_.q_normed );
            q_norm_->build( qknorm_ctx );

            k_norm_ = this->template getComponentAs<RmsNormType>( n + ".k_norm" );
            install( k_norm_, workspace_.k_normed );
            k_norm_->build( kknorm_ctx );

            v_norm_ = this->template getComponentAs<RmsNormType>( n + ".v_norm" );
            install( v_norm_, workspace_.v_normed );
            v_norm_->build( kknorm_ctx );

            rope_ = this->template getComponentAs<RopeType>( n + ".rope" );
            rope_->build( qproj_ctx );

            attn_ = this->template getComponentAs<AttentionType>( n + ".gqa" );
            install( attn_, workspace_.attn );
            attn_->build( qkv_ctx );

            o_proj_ = this->template getComponentAs<LinearType>( n + ".o_proj" );
            install( o_proj_, workspace_.o );
            o_proj_->build( context.withShape( qproj_shape ) );

            post_attn_norm_ = this->template getComponentAs<RmsNormType>( n + ".post_attn_norm" );
            install( post_attn_norm_, workspace_.o_normed );
            post_attn_norm_->build( stream_ctx );

            res1_ = this->template getComponentAs<ResidualType>( n + ".res_1" );
            install( res1_, workspace_.res1 );
            res1_->build( stream_ctx );

            pre_ffn_norm_ = this->template getComponentAs<RmsNormType>( n + ".pre_ffn_norm" );
            install( pre_ffn_norm_, workspace_.ffn_in );
            pre_ffn_norm_->build( stream_ctx );

            fc_gate_up_ = this->template getComponentAs<LinearType>( n + ".fc_gate_up" );
            install( fc_gate_up_, workspace_.gate_up );
            fc_gate_up_->build( stream_ctx );

            geglu_ = this->template getComponentAs<GeGLUType>( n + ".geglu" );
            install( geglu_, workspace_.ffn_act );
            geglu_->build( gate_up_ctx );

            fc_down_ = this->template getComponentAs<LinearType>( n + ".fc_down" );
            install( fc_down_, workspace_.ffn_down );
            fc_down_->build( hidden_ctx );

            post_ffn_norm_ = this->template getComponentAs<RmsNormType>( n + ".post_ffn_norm" );
            install( post_ffn_norm_, workspace_.ffn_normed );
            post_ffn_norm_->build( stream_ctx );

            res2_ = this->template getComponentAs<ResidualType>( n + ".res_2" );
            install( res2_, workspace_.stream );
            res2_->build( stream_ctx );

            // Split scratch (sized at the prefill chunk; decode uses T=1 sub-views).
            // With a transformer-installed workspace, validate coverage and view
            // prefixes instead of self-allocating.
            if ( workspace_installed_ )
            {
                const int64_t q_needed = B * chunk * NH * HD;
                const int64_t kv_needed = B * chunk * NKV * HD;

                if ( !q_ || q_->size() < q_needed || !k_ || k_->size() < kv_needed
                     || ( !kGlobal && ( !v_ || v_->size() < kv_needed ) ) )
                    throw std::invalid_argument( std::format(
                        "GemmaBlock '{}': installed shared scratch is smaller than the block geometry requires", n ) );
            }
            else
            {
                auto device = this->getExecutionContext()->getDeviceId();
                q_ = std::make_shared<TensorType>( device, qproj_shape, n + ".q" );
                k_ = std::make_shared<TensorType>( device, shape_t{ B, chunk, NKV * HD }, n + ".k" );

                if constexpr ( !kGlobal )
                    v_ = std::make_shared<TensorType>( device, shape_t{ B, chunk, NKV * HD }, n + ".v" );
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
        GemmaConfig config_;
        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };

        // Host staging to read the [1] layer_scalar to a float at load time:
        // pinned on CUDA, pageable on CPU.
#ifdef MILA_HAS_CUDA
        using HostStagingMR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaPinnedMemoryResource, CpuMemoryResource>;
#else
        using HostStagingMR = CpuMemoryResource;
#endif

        std::shared_ptr<RmsNormType> input_norm_{ nullptr };
        std::shared_ptr<LinearType> qkv_proj_{ nullptr };
        std::shared_ptr<RmsNormType> q_norm_{ nullptr };
        std::shared_ptr<RmsNormType> k_norm_{ nullptr };
        std::shared_ptr<RmsNormType> v_norm_{ nullptr };
        std::shared_ptr<RopeType> rope_{ nullptr };
        std::shared_ptr<AttentionType> attn_{ nullptr };
        std::shared_ptr<LinearType> o_proj_{ nullptr };
        std::shared_ptr<RmsNormType> post_attn_norm_{ nullptr };
        std::shared_ptr<ResidualType> res1_{ nullptr };
        std::shared_ptr<RmsNormType> pre_ffn_norm_{ nullptr };
        std::shared_ptr<LinearType> fc_gate_up_{ nullptr };
        std::shared_ptr<GeGLUType> geglu_{ nullptr };
        std::shared_ptr<LinearType> fc_down_{ nullptr };
        std::shared_ptr<RmsNormType> post_ffn_norm_{ nullptr };
        std::shared_ptr<ResidualType> res2_{ nullptr };

        // Split scratch: self-allocated at build, or the transformer workspace's
        // q/k/v set (installSharedWorkspace) which all layers view prefixes of.
        std::shared_ptr<TensorType> q_{ nullptr };
        std::shared_ptr<TensorType> k_{ nullptr };
        std::shared_ptr<TensorType> v_{ nullptr };

        // The full installed workspace: onBuilding routes the component-output
        // slots into the children via installSharedOutput. Empty (all null) when
        // the block is standalone-built.
        GemmaBlockWorkspace<TDeviceType, TPrecision> workspace_{};
        bool workspace_installed_{ false };

        // Gemma 4 Unified per-layer learned output scale (hidden_states *= layer_scalar).
        // Default 1.0 (identity) until loaded from the checkpoint via loadParameter.
        float layer_scalar_{ 1.0f };

        void createGraph()
        {
            const std::string n = this->getName();
            const dim_t model_dim = config_.getModelDim();
            const dim_t hidden_dim = config_.getHiddenDimension();
            const float eps = config_.getRMSNormEpsilon();
            const dim_t HD = headDim();

            // NORM CONVENTION: Gemma 4 RMSNorm applies the RAW stored weight (x_norm * weight),
            // NOT Gemma 3's x_norm * (1 + weight) -- confirmed against the Gemma4Unified source
            // (RMSNorm.forward: _norm(x) * weight) and token-for-token HF parity. The converter
            // writes the weights raw, so withUnitOffset stays at its default 0 (Llama-safe).
            // The ~18x residual blow-up was a separate bug (the missing per-layer layer_scalar
            // multiply, applied at the end of prefill/decode), not the norm convention.
            auto rms = [&]( const shape_t& shape )
            {
                return RmsNormConfig( shape ).withEpsilon( eps ).withBias( false );
            };

            // Stream-dim norms (sandwich) + per-head QK norms.
            this->addComponent( std::make_shared<RmsNormType>( n + ".input_norm", rms( shape_t{ model_dim } ) ) );
            this->addComponent( std::make_shared<RmsNormType>( n + ".q_norm", rms( shape_t{ HD } ) ) );
            this->addComponent( std::make_shared<RmsNormType>( n + ".k_norm", rms( shape_t{ HD } ) ) );
            // v_norm: Gemma 4 normalizes V per head (no learnable scale -- the converter writes a
            // unit weight so this standard RmsNorm computes normalize*1 == pure normalize). Applied
            // on every layer: sliding layers normalize the separate V projection, global K=V layers
            // normalize the raw k_proj (V = v_norm(k_proj)). Per-head shape matches q/k_norm
            // (head_dim, i.e. global_head_dim on the global blocks).
            this->addComponent( std::make_shared<RmsNormType>( n + ".v_norm", rms( shape_t{ HD } ) ) );
            this->addComponent( std::make_shared<RmsNormType>( n + ".post_attn_norm", rms( shape_t{ model_dim } ) ) );
            this->addComponent( std::make_shared<RmsNormType>( n + ".pre_ffn_norm", rms( shape_t{ model_dim } ) ) );
            this->addComponent( std::make_shared<RmsNormType>( n + ".post_ffn_norm", rms( shape_t{ model_dim } ) ) );

            // Fused QKV projection: model_dim -> packed (K=V global drops the V section).
            this->addComponent( std::make_shared<LinearType>(
                n + ".qkv_proj", LinearConfig( model_dim, packedQKVWidth() ).withBias( false ) ) );

            // RoPE over the Q-projection width; per-layer base + rotary_dim (proportional on global).
            auto rope_cfg = RopeConfig( qProjWidth(), config_.getNumHeads(), numKVHeads(), config_.getMaxSequenceLength() )
                .withBase( ropeTheta() )
                .withRotaryDim( static_cast<size_t>( rotaryDim() ) );
            this->addComponent( std::make_shared<RopeType>( n + ".rope", rope_cfg ) );

            // GQA: head_dim from the Q-width, window per-layer, scale 1.0 (QK-norm controls magnitude).
            auto gqa_cfg = GqaConfig( qProjWidth(), config_.getNumHeads(), numKVHeads() )
                .withWindow( window() )
                .withAttentionScale( 1.0f );
            this->addComponent( std::make_shared<AttentionType>( n + ".gqa", gqa_cfg ) );

            // Non-square output projection: (num_heads * head_dim) -> model_dim.
            this->addComponent( std::make_shared<LinearType>(
                n + ".o_proj", LinearConfig( qProjWidth(), model_dim ).withBias( false ) ) );

            this->addComponent( std::make_shared<ResidualType>( n + ".res_1", ResidualConfig{} ) );

            // GeGLU FFN: fused gate+up -> Swiglu<Gelu> -> down.
            this->addComponent( std::make_shared<LinearType>(
                n + ".fc_gate_up", LinearConfig( model_dim, 2 * hidden_dim ).withBias( false ) ) );
            this->addComponent( std::make_shared<GeGLUType>( n + ".geglu", SwigluConfig() ) );
            this->addComponent( std::make_shared<LinearType>(
                n + ".fc_down", LinearConfig( hidden_dim, model_dim ).withBias( false ) ) );

            this->addComponent( std::make_shared<ResidualType>( n + ".res_2", ResidualConfig{} ) );
        }

        void validateBuildContext( const BuildContext& context ) const
        {
            const auto& s = context.inputShape();

            if ( s.size() != 3 )
                throw std::invalid_argument( std::format(
                    "GemmaBlock: input must be rank 3 [B, T, model_dim], got rank {}", s.size() ) );

            if ( s.back() != config_.getModelDim() )
                throw std::invalid_argument( std::format(
                    "GemmaBlock: model_dim mismatch -- expected {}, got {}", config_.getModelDim(), s.back() ) );
        }
    };
}
