/**
 * @file Gemma.Block.ixx
 * @brief Gemma 4 decoder block (inference: prefill + decode), one type per attention kind.
 *
 * Modeled on LlamaBlock, with the Gemma 4 deltas:
 *  - Sandwich norm (4 RMSNorms): input -> attn -> post_attn -> +res -> pre_ffn -> ffn -> post_ffn -> +res.
 *  - QK-norm: per-head RMSNorm over head_dim on Q and K, applied BEFORE RoPE.
 *  - GeGLU FFN (Swiglu<..., Gelu>), decoupled head_dim, non-square o_proj.
 *  - Per-layer geometry via the compile-time kGlobal flag: global layers use
 *    global_head_dim / a single shared KV head / K=V (V aliased to K) / full attention /
 *    proportional partial-rotary; sliding layers use head_dim / num_kv_heads / window / full rotation.
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
// TEMPORARY (activation-diff instrumentation): console output for per-sub-step magnitudes.
#include <iostream>
#include <cmath>
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

    // TEMPORARY (remove once Gemma parity is localized): print per-sub-step LAST-TOKEN
    // magnitudes for the FIRST prefill call of each block kind (layer 0 local / first
    // global), to pinpoint which op in the block first blows the activation up. Pairs
    // with the transformer-level kGemmaDumpActivations.
    inline constexpr bool kGemmaBlockDumpActivations = true;

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
        // IDecoderLayer — inference path
        // ====================================================================

        TensorType& prefill( const TensorType& input, int position_offset ) override
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "GemmaBlock::prefill: must be built before prefill()." );

            const int64_t B = input.shape()[ 0 ];
            const int64_t T = input.shape()[ 1 ];
            const dim_t model_dim = config_.getModelDim();
            const dim_t NH = config_.getNumHeads();
            const dim_t NKV = numKVHeads();
            const dim_t HD = headDim();

            // TEMPORARY: dump per-sub-step magnitudes for the FIRST prefill call only
            // (static is per-instantiation, so this fires once for layer 0 (local) and
            // once for the first global block).
            static bool dumped = false;
            [[maybe_unused]] const bool do_dump = kGemmaBlockDumpActivations && !dumped;

            // Preserve the residual input (res0) — component buffers get overwritten downstream.
            auto res0 = res0_->view( shape_t{ B, T, model_dim }, 0 );
            copy( input, res0 );

            if constexpr ( kGemmaBlockDumpActivations )
                if ( do_dump )
                {
                    printBlockActivation( "input", input );

                    // The LOADED input_norm weight: a correct Gemma (1+w) norm weight is
                    // ~1 with small spread; large values / big outliers => corrupted load.
                    const auto norm_params = input_norm_->getParameters();
                    if ( !norm_params.empty() )
                        printBlockActivation( "input_norm.W", *static_cast<const TensorType*>( norm_params[ 0 ] ) );
                }

            // --- Attention sub-block ----------------------------------------
            auto& normed = input_norm_->forward( input );
            auto& qkv = qkv_proj_->forward( normed );

            if constexpr ( kGemmaBlockDumpActivations )
                if ( do_dump )
                {
                    printBlockActivation( "input_norm", normed );
                    printBlockActivation( "qkv_proj", qkv );
                }

            // Split fused QKV into per-projection contiguous buffers.
            auto q = q_->view( shape_t{ B, T, NH * HD }, 0 );
            auto k = k_->view( shape_t{ B, T, NKV * HD }, 0 );

            if constexpr ( kGlobal )
            {
                // K=V: the fused projection is [Q | K] (2-way split); V aliases K.
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

            if constexpr ( kGemmaBlockDumpActivations )
                if ( do_dump )
                {
                    printBlockActivation( "qk_norm(q)", q_roped );
                    printBlockActivation( "qk_norm(k)", k_roped );

                    // Loaded q_norm/k_norm weights: a correct Gemma (1+w) head-dim norm
                    // weight should match HF's q_norm/k_norm output RMS (1.02 / 0.12).
                    const auto qn = q_norm_->getParameters();
                    const auto kn = k_norm_->getParameters();
                    if ( !qn.empty() ) printBlockActivation( "q_norm.W", *static_cast<const TensorType*>( qn[ 0 ] ) );
                    if ( !kn.empty() ) printBlockActivation( "k_norm.W", *static_cast<const TensorType*>( kn[ 0 ] ) );

                    // DIRECTION trace, token 0 (BOS) AND last token (the self-attended query),
                    // for the V-magnitude bug: att matches HF, so output diff => V diff.
                    printBlockTokenN( "input_norm", normed, 0 );
                    printBlockTokenN( "input_norm", normed, T - 1 );
                    if constexpr ( !kGlobal )
                    {
                        auto v_full = v_->view( shape_t{ B, T, NKV * HD }, 0 );
                        printBlockTokenN( "V", v_full, 0 );
                        printBlockTokenN( "V", v_full, T - 1 );

                        // V per KV head for the last token: if the per-head l2 values match HF
                        // but in a different ORDER, the V KV-head ordering (vs K) is permuted.
                        Tensor<TensorDataType::FP32, DumpStagingMR> vh(
                            TDeviceType == DeviceType::Cuda ? this->getDeviceId() : Device::Cpu(), v_full.shape() );
                        copy( v_full, vh );
                        this->getExecutionContext()->synchronize();
                        const float* vrow = vh.data() + ( T - 1 ) * NKV * HD;
                        std::string line = "[V-KVHEAD] last-token l2 per kv:";
                        for ( int kv = 0; kv < NKV; ++kv )
                        {
                            double ss = 0.0;
                            for ( int i = 0; i < HD; ++i )
                                ss += static_cast<double>( vrow[ kv * HD + i ] ) * vrow[ kv * HD + i ];
                            line += std::format( " kv{}={:.2f}", kv, std::sqrt( ss ) );
                        }
                        std::cout << line << "\n";
                    }
                }

            rope_->prefill( q_roped, k_roped, position_offset );

            if constexpr ( kGemmaBlockDumpActivations )
                if ( do_dump ) printBlockActivation( "rope(q)", q_roped );

            // GQA: V is the separate v_ buffer (local) or aliased to K (global K=V).
            TensorType* attn_ptr = nullptr;
            if constexpr ( kGlobal )
            {
                attn_ptr = &attn_->prefill( q_roped, k_roped, k_roped, position_offset );
            }
            else
            {
                // Gemma 4: V is per-head normalized (v_norm, no learnable scale) before attention.
                auto v_perhead = v_->view( shape_t{ B, T * NKV, HD }, 0 );
                auto& v_normed = v_norm_->forward( v_perhead );
                auto v_view = v_normed.view( shape_t{ B, T, NKV * HD }, 0 );
                attn_ptr = &attn_->prefill( q_roped, k_roped, v_view, position_offset );
            }
            auto& attn = *attn_ptr;

            auto& o = o_proj_->forward( attn );
            auto& o_normed = post_attn_norm_->forward( o );
            auto& res1 = res1_->forward( res0, o_normed );

            if constexpr ( kGemmaBlockDumpActivations )
                if ( do_dump )
                {
                    printBlockActivation( "attn", attn );
                    printBlockActivation( "o_proj", o );
                    printBlockActivation( "post_attn_norm", o_normed );
                    printBlockActivation( "res1(+attn)", res1 );

                    // Attention-output DIRECTION (last-token head) -- res1 L2 matches HF but its
                    // DIRECTION is off (post_attn_norm[1] sign-flips: Mila +0.19 vs HF -0.86),
                    // the signature of an attention-output permutation (unpermute head ordering).
                    // HF post_attn_norm head = [-0.0001, -0.8633, -50.75, +0.1445] (reliable hook).
                    printBlockTokenN( "attn", attn, T - 1 );
                    printBlockTokenN( "o_proj", o, T - 1 );
                    printBlockTokenN( "post_attn_norm", o_normed, T - 1 );
                }

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

            if constexpr ( kGemmaBlockDumpActivations )
                if ( do_dump )
                {
                    printBlockActivation( "pre_ffn_norm", ffn_in );
                    printBlockActivation( "gate_up", gate_up );
                    printBlockActivation( "geglu", ffn_act );
                    printBlockActivation( "fc_down", ffn );
                    printBlockActivation( "post_ffn_norm", ffn_normed );
                    printBlockActivation( "res2(+ffn)", res2 );

                    // Loaded sandwich-norm weights: must match HF RAW post_attention/
                    // pre_feedforward/post_feedforward_layernorm.weight element-wise. Only
                    // input_norm.W + the QK-norms were verified after the raw-weight re-convert;
                    // if post_ffn_norm.W here differs from HF raw, the FFN residual inflation
                    // (res2 1634 vs HF 88) is a weight-LOAD bug, not the FFN body.
                    const auto pan = post_attn_norm_->getParameters();
                    const auto prn = pre_ffn_norm_->getParameters();
                    const auto pfn = post_ffn_norm_->getParameters();
                    if ( !pan.empty() ) printBlockActivation( "post_attn_norm.W", *static_cast<const TensorType*>( pan[ 0 ] ) );
                    if ( !prn.empty() ) printBlockActivation( "pre_ffn_norm.W", *static_cast<const TensorType*>( prn[ 0 ] ) );
                    if ( !pfn.empty() ) printBlockActivation( "post_ffn_norm.W", *static_cast<const TensorType*>( pfn[ 0 ] ) );

                    // Element-wise (last-token head) view of the FFN residual add: res2 = res1 +
                    // post_ffn_norm. HF cancels (res1 333 -> res2 88) so HF post_ffn_norm[i] ~ -res1[i];
                    // Mila inflates, so compare post_ffn_norm[i] direction here vs HF [HF-BLOCK] head.
                    printBlockTokenN( "res1", res1, T - 1 );
                    printBlockTokenN( "post_ffn_norm", ffn_normed, T - 1 );
                    printBlockTokenN( "res2", res2, T - 1 );

                    // FFN-body element-wise (last-token head) vs HF genuine activations:
                    // gate_up head[0:4] = gate_proj[0:4] (HF gate_proj head all NEGATIVE);
                    // geglu vs HF geglu [-0.0, +0.23, -0.009, -0.0]; fc_down vs HF mlp.
                    // The geglu magnitude jumps 1.25x (gate_up matches HF) -> GeGLU/gate split suspect.
                    printBlockTokenN( "gate_up", gate_up, T - 1 );
                    printBlockTokenN( "geglu", ffn_act, T - 1 );
                    printBlockTokenN( "fc_down", ffn, T - 1 );

                    dumped = true;
                }

            return res2;
        }

        TensorType& decode( const TensorType& input, int position ) override
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

            TensorType* attn_ptr = nullptr;
            if constexpr ( kGlobal )
            {
                attn_ptr = &attn_->decode( q_roped, k_roped, k_roped, position );
            }
            else
            {
                // Gemma 4: per-head V normalization (v_norm, no learnable scale) before attention.
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

        bool supportsKVCache() const noexcept override
        {
            return attn_ && attn_->supportsKVCache();
        }

        void resetKVCache() override
        {
            if ( attn_ )
                attn_->resetKVCache();
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

            for ( auto* t : { res0_.get(), q_.get(), k_.get(), v_.get() } )
            {
                if ( t )
                    stats.device_state_bytes += t->getStorageSize();
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
                Tensor<TensorDataType::FP32, DumpStagingMR> host(
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
            // packing — not the K=V-aware block packing — so use that here regardless of kGlobal.
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

            input_norm_ = this->template getComponentAs<RmsNormType>( n + ".input_norm" );
            input_norm_->build( stream_ctx );

            qkv_proj_ = this->template getComponentAs<LinearType>( n + ".qkv_proj" );
            qkv_proj_->build( stream_ctx );

            q_norm_ = this->template getComponentAs<RmsNormType>( n + ".q_norm" );
            q_norm_->build( qknorm_ctx );

            k_norm_ = this->template getComponentAs<RmsNormType>( n + ".k_norm" );
            k_norm_->build( kknorm_ctx );

            v_norm_ = this->template getComponentAs<RmsNormType>( n + ".v_norm" );
            v_norm_->build( kknorm_ctx );

            rope_ = this->template getComponentAs<RopeType>( n + ".rope" );
            rope_->build( qproj_ctx );

            attn_ = this->template getComponentAs<AttentionType>( n + ".gqa" );
            attn_->build( qkv_ctx );

            o_proj_ = this->template getComponentAs<LinearType>( n + ".o_proj" );
            o_proj_->build( context.withShape( qproj_shape ) );

            post_attn_norm_ = this->template getComponentAs<RmsNormType>( n + ".post_attn_norm" );
            post_attn_norm_->build( stream_ctx );

            res1_ = this->template getComponentAs<ResidualType>( n + ".res_1" );
            res1_->build( stream_ctx );

            pre_ffn_norm_ = this->template getComponentAs<RmsNormType>( n + ".pre_ffn_norm" );
            pre_ffn_norm_->build( stream_ctx );

            fc_gate_up_ = this->template getComponentAs<LinearType>( n + ".fc_gate_up" );
            fc_gate_up_->build( stream_ctx );

            geglu_ = this->template getComponentAs<GeGLUType>( n + ".geglu" );
            geglu_->build( gate_up_ctx );

            fc_down_ = this->template getComponentAs<LinearType>( n + ".fc_down" );
            fc_down_->build( hidden_ctx );

            post_ffn_norm_ = this->template getComponentAs<RmsNormType>( n + ".post_ffn_norm" );
            post_ffn_norm_->build( stream_ctx );

            res2_ = this->template getComponentAs<ResidualType>( n + ".res_2" );
            res2_->build( stream_ctx );

            // Scratch buffers (sized at the prefill chunk; decode uses T=1 sub-views).
            auto device = this->getExecutionContext()->getDeviceId();
            res0_ = std::make_unique<TensorType>( device, stream_shape, n + ".res0" );
            q_ = std::make_unique<TensorType>( device, qproj_shape, n + ".q" );
            k_ = std::make_unique<TensorType>( device, shape_t{ B, chunk, NKV * HD }, n + ".k" );

            if constexpr ( !kGlobal )
                v_ = std::make_unique<TensorType>( device, shape_t{ B, chunk, NKV * HD }, n + ".v" );
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

        // ---- TEMPORARY: per-sub-step magnitude print (remove after Gemma parity) ----
#ifdef MILA_HAS_CUDA
        using DumpStagingMR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaPinnedMemoryResource, CpuMemoryResource>;
#else
        using DumpStagingMR = CpuMemoryResource;
#endif

        void printBlockActivation( const std::string& step, const TensorType& tensor ) const
        {
            Tensor<TensorDataType::FP32, DumpStagingMR> host(
                TDeviceType == DeviceType::Cuda ? this->getDeviceId() : Device::Cpu(), tensor.shape() );

            copy( tensor, host );
            this->getExecutionContext()->synchronize();

            // Summarize the last row of the last dimension (last-token vector for a
            // rank-3 [B,T,W] activation; the whole vector for a rank-1 weight).
            const auto& shp = tensor.shape();
            const int64_t width = shp[ shp.ndim - 1 ];
            const size_t n = host.size();
            const float* row = host.data() + ( n - static_cast<size_t>( width ) );

            double sumsq = 0.0;
            float mn = row[ 0 ], mx = row[ 0 ];

            for ( int64_t i = 0; i < width; ++i )
            {
                const float v = row[ i ];
                sumsq += static_cast<double>( v ) * v;
                mn = std::min( mn, v );
                mx = std::max( mx, v );
            }

            std::cout << std::format(
                "[BLOCK-DUMP] {:<24} {:<10} w={:>5} l2={:>12.4f} min={:>+12.4f} max={:>+12.4f}\n",
                this->getName(), step, width, std::sqrt( sumsq ), mn, mx );
        }

        // TEMPORARY: dump one token's row (l2 + first head values), to compare DIRECTION
        // against HF (token 0 = BOS sink, last token = the self-attended query position).
        void printBlockTokenN( const std::string& step, const TensorType& tensor, int64_t token_idx ) const
        {
            Tensor<TensorDataType::FP32, DumpStagingMR> host(
                TDeviceType == DeviceType::Cuda ? this->getDeviceId() : Device::Cpu(), tensor.shape() );

            copy( tensor, host );
            this->getExecutionContext()->synchronize();

            const auto& shp = tensor.shape();
            const int64_t width = shp[ shp.ndim - 1 ];
            const float* row = host.data() + token_idx * width;

            double sumsq = 0.0;

            for ( int64_t i = 0; i < width; ++i )
                sumsq += static_cast<double>( row[ i ] ) * row[ i ];

            std::cout << std::format(
                "[BLOCK-T{}] {:<24} {:<12} w={:>5} l2={:>12.4f} head=[{:+.4f}, {:+.4f}, {:+.4f}, {:+.4f}]\n",
                token_idx, this->getName(), step, width, std::sqrt( sumsq ), row[ 0 ], row[ 1 ], row[ 2 ], row[ 3 ] );
        }

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

        std::unique_ptr<TensorType> res0_{ nullptr };
        std::unique_ptr<TensorType> q_{ nullptr };
        std::unique_ptr<TensorType> k_{ nullptr };
        std::unique_ptr<TensorType> v_{ nullptr };

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

            // NORM CONVENTION UNDER INVESTIGATION (Gemma 4 != Gemma 3). The Gemma3 source applies
            // x_norm * (1 + weight) to all RMSNorms, but applying that here ONE-HOTS the attention
            // (QK-norm (1+w) makes Q*2 / K*9 -> Q.K ~18x -> argmax softmax), whereas HF's att
            // (output_attentions) is SOFT -- so Gemma 4's QK-norm is effectively RAW. Using RAW
            // here (withUnitOffset default 0 = soft att, matches HF). The separate ~18x residual
            // blow-up (res2 1634 vs HF 88) is NOT the norm convention and is still open; resolve
            // against the actual Gemma-4 modeling source before re-touching withUnitOffset.
            auto rms = [&]( const shape_t& shape )
            {
                return RmsNormConfig( shape ).withEpsilon( eps ).withBias( false );
            };

            // Stream-dim norms (sandwich) + per-head QK norms.
            this->addComponent( std::make_shared<RmsNormType>( n + ".input_norm", rms( shape_t{ model_dim } ) ) );
            this->addComponent( std::make_shared<RmsNormType>( n + ".q_norm", rms( shape_t{ HD } ) ) );
            this->addComponent( std::make_shared<RmsNormType>( n + ".k_norm", rms( shape_t{ HD } ) ) );
            // v_norm: Gemma 4 normalizes V per head (no learnable scale -- the converter writes a
            // unit weight so this standard RmsNorm computes normalize*1 == pure normalize). Present
            // on every block so the weight loads; applied in the LOCAL attention path (global V is
            // step 2). Same per-head shape as q/k_norm.
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

            if ( s.back() != static_cast<int64_t>( config_.getModelDim() ) )
                throw std::invalid_argument( std::format(
                    "GemmaBlock: model_dim mismatch — expected {}, got {}", config_.getModelDim(), s.back() ) );
        }
    };
}
