/**
 * @file Llama.Block.ixx
 * @brief LLaMA transformer block — module partition of LlamaTransformer.
 *
 * Implements the correct Llama 3.x attention sub-graph using fused projections,
 * zero-copy tensor views, and in-place RoPE rotation:
 *
 *   input [B, T, model_dim]
 *     └─ RMSNorm (ln_1)
 *          └─ fc_qkv_proj  [model_dim → (n_heads + 2*n_kv_heads) * head_dim]  1 GEMM
 *               └─ view Q  [B, T, n_heads    * head_dim]  ──┐
 *               └─ view K  [B, T, n_kv_heads * head_dim]  ──┤ RoPE in-place
 *               └─ view V  [B, T, n_kv_heads * head_dim]    │ (V untouched)
 *                    └─ GroupedQueryAttention (packed QKV)
 *                         └─ fc_out_proj  [model_dim → model_dim]
 *                              └─ Residual (input + out_proj)            res_1
 *                                   └─ RMSNorm (ln_2)
 *                                        └─ fc_gate_up  [model_dim → 2*hidden_dim]  1 GEMM
 *                                             └─ SwiGLU  → [B, T, hidden_dim]
 *                                                  └─ fc_down  [hidden_dim → model_dim]
 *                                                       └─ Residual (res1 + ffn)    res_2
 *
 * Key design points:
 *  - Single GEMM for QKV (GQA-correct output dim).
 *  - RoPE applied in-place via zero-copy views — no concat/split ops needed.
 *  - Single GEMM for gate+up (SwiGLU kernel expects [gate | up] layout).
 *  - FFN composed directly from Linear + SwiGLU primitives; no MLP composite.
 *  - Component names match convert_llama32.py tensor name mapping.
 *
 * Weight loader note:
 *  HuggingFace stores fc_gate and fc_up as separate tensors. The weight loader
 *  must concatenate them along dim 0 into a single [2*hidden_dim, model_dim]
 *  matrix when loading into fc_gate_up.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <format>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <optional>

export module Dnn.Components.LlamaTransformer:Block;
import :Config;

import Dnn.ITensor;
import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorOps;
import Dnn.Component;
import Dnn.ComponentType;
import Dnn.CompositeComponent;
import Compute.Precision;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.ExecutionContextFactory;
import Dnn.Components.RmsNorm;
import Dnn.Components.Rope;
import Dnn.Components.Gqa;
import Dnn.Components.Residual;
import Dnn.Components.Linear;
import Dnn.Components.Swiglu;
import Serialization.ModelArchive;
import Serialization.Mode;

import Cuda.Debug;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    // TODO: Temporary. To be replaced by tuned chunk size based on empirical perf testing across devices and precisions.
    export constexpr int64_t kPrefillChunkSize = 64;

    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class LlamaBlock : public CompositeComponent<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using CompositeComponentBase = CompositeComponent<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using RmsNormType = RmsNorm<TDeviceType, TPrecision>;
        using RopeType = Rope<TDeviceType, TPrecision>;
        using AttentionType = GroupedQueryAttention<TDeviceType, TPrecision>;
        using ResidualType = Residual<TDeviceType, TPrecision>;
        using LinearType = Linear<TDeviceType, TPrecision>;
        using SwiGLUType = Swiglu<TDeviceType, TPrecision>;

        explicit LlamaBlock(
            const std::string& name,
            const LlamaConfig& config,
            std::optional<DeviceId> device_id = std::nullopt )
            : CompositeComponentBase( name ), config_( config )
        {
            config_.validate();
            createGraph();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                {
                    throw std::invalid_argument( "LlamaBlock: device type mismatch" );
                }

                owned_exec_context_ = createExecutionContext( device_id.value() );
                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~LlamaBlock() override = default;

        TensorType& forward( const TensorType& input )
        {
            // 1. Pre-attention RMSNorm.
            auto& rms1_out = rms1_->forward( input );
            this->getExecutionContext()->synchronize();

            // 2. Fused QKV projection — single GEMM.
            //    Output: [B, T, (n_heads + 2*n_kv_heads) * head_dim]
            auto& qkv_out = qkv_proj_->forward( rms1_out );
            this->getExecutionContext()->synchronize();

            // REVIEW: The use of tensor views here is incorrect. The Q,K,V splits of qkv_out are not actually contiguous in memory,
            // so the views are lying about the true layout.
            // See the prefill() implementation for the correct approach.
            // 
            // 3. Zero-copy views into the packed QKV buffer.
            auto Q = qkv_out.view( q_shape_, 0 );
            auto K = qkv_out.view( k_shape_, q_offset_ );
            // V occupies the remainder — GQA reads the full qkv_out buffer.

            // 4. RoPE: rotate Q and K in-place inside qkv_out. V untouched.
            rope_->forward( Q, K );
            this->getExecutionContext()->synchronize();

            // 5. GQA receives the packed buffer — Q and K are now rotated.
            auto& attn_out = attn_->forward( qkv_out );
            this->getExecutionContext()->synchronize();

            // 6. Output projection.
            auto& out_proj_out = out_proj_->forward( attn_out );
            this->getExecutionContext()->synchronize();

            // 7. First residual: input + out_proj.
            auto& res1_out = res1_->forward( input, out_proj_out );
            this->getExecutionContext()->synchronize();

            // 8. Post-attention RMSNorm.
            auto& rms2_out = rms2_->forward( res1_out );
            this->getExecutionContext()->synchronize();

            // 9. FFN — fused gate+up projection, SwiGLU, down projection.
            auto& gate_up_out = fc_gate_up_->forward( rms2_out );
            this->getExecutionContext()->synchronize();

            auto& swiglu_out = swiglu_->forward( gate_up_out );
            this->getExecutionContext()->synchronize();

            auto& ffn_out = fc_down_->forward( swiglu_out );
            this->getExecutionContext()->synchronize();

            // 10. Second residual: res1 + ffn.
            auto& res2_out = res2_->forward( res1_out, ffn_out );
            this->getExecutionContext()->synchronize();

            // Cache activations for backward.
            last_rms1_out_ = &rms1_out;
            last_qkv_out_ = &qkv_out;
            last_attn_out_ = &attn_out;
            last_out_proj_out_ = &out_proj_out;
            last_res1_out_ = &res1_out;
            last_rms2_out_ = &rms2_out;
            last_gate_up_out_ = &gate_up_out;
            last_swiglu_out_ = &swiglu_out;
            last_ffn_out_ = &ffn_out;
            last_res2_out_ = &res2_out;

            // FIXME: forward_executed_ = this->isTraining();

            return res2_out;
        }

        // ====================================================================
        // Decode (inference / KV-cache path)
        // ====================================================================

        TensorType& prefill( const TensorType& input, int position_offset )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error(
                    "LlamaBlock::prefill: must be built before prefill()." );
            }

            int64_t B = input.shape()[ 0 ];
            int64_t T_actual = input.shape()[ 1 ];

            const int64_t n_heads = config_.getNumHeads();
            const int64_t n_kv = config_.getNumKVHeads();
            const int64_t head_dim = config_.getModelDim() / n_heads;

            // Preserve skip connection
            auto res1_view = res1_prefill_->view( shape_t{ B, T_actual, config_.getModelDim() }, 0 );
            copy( input, res1_view );
            // DEBUG: this->getExecutionContext()->synchronize();

            // Pre-attention RMSNorm
            auto& rms1_out = rms1_->forward( input );
            // DEBUG: this->getExecutionContext()->synchronize();

            // Fused QKV projection
            auto& qkv_out = qkv_proj_->forward( rms1_out );
            // DEBUG: this->getExecutionContext()->synchronize();

            // Split the fused QKV output into separate Q, K and V for RoPE and GQA
            // Create T-trimmed views into the pre-allocated chunk buffers so split()'s
            // shape validation (B and T must match) succeeds when T_actual < kPrefillChunkSize.
            auto q_view = q_->view( shape_t{ B, T_actual, n_heads * head_dim }, 0 );
            auto k_view = k_->view( shape_t{ B, T_actual, n_kv * head_dim }, 0 );
            auto v_view = v_->view( shape_t{ B, T_actual, n_kv * head_dim }, 0 );

            split(
                qkv_out,
                q_view, k_view, v_view,
                this->getExecutionContext() );

            //this->getExecutionContext()->synchronize();

            // RoPE
            rope_->prefill( q_view, k_view, position_offset );
            //this->getExecutionContext()->synchronize();

            // GQA prefill
            auto& attn_out = attn_->prefill( q_view, k_view, v_view, position_offset );
            //this->getExecutionContext()->synchronize();

            // 5. Output projection
            auto& out_proj_out = out_proj_->forward( attn_out );
            //this->getExecutionContext()->synchronize();

            // 6. First residual
            auto& res1_out = res1_->forward( res1_view, out_proj_out );
            //this->getExecutionContext()->synchronize();

            // 7. Post-attention RMSNorm
            auto& rms2_out = rms2_->forward( res1_out );
            //this->getExecutionContext()->synchronize();

            // 8. Fused gate+up projection
            auto& gate_up_out = fc_gate_up_->forward( rms2_out );
            //this->getExecutionContext()->synchronize();

            // 9. SwiGLU
            auto& swiglu_out = swiglu_->forward( gate_up_out );
            //this->getExecutionContext()->synchronize();

            // 10. Down projection
            auto& ffn_out = fc_down_->forward( swiglu_out );
            //this->getExecutionContext()->synchronize();

            // 11. Second residual
            auto& res2_out = res2_->forward( res1_out, ffn_out );
            //this->getExecutionContext()->synchronize();
            
            return res2_out;
        }

        TensorType& decode( const TensorType& input, int position )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error(
                    "LlamaBlock::decode: must be built before decode()." );
            }

            int64_t B = input.shape()[ 0 ];
            int64_t T_actual = input.shape()[ 1 ];
            const int64_t n_heads = config_.getNumHeads();
            const int64_t n_kv = config_.getNumKVHeads();
            const int64_t head_dim = config_.getModelDim() / n_heads;

            // Pre-attention RMSNorm, T=1
            auto& rms1_out = rms1_->forward( input );
            //this->getExecutionContext()->synchronize();

            // 2. Fused QKV projection, T=1
            auto& qkv_out = qkv_proj_->forward( rms1_out );
            //this->getExecutionContext()->synchronize();

            // Split fused QKV into separate contiguous tensors
            auto q_decode = q_->view( shape_t{ B, 1, n_heads * head_dim }, 0 );
            auto k_decode = k_->view( shape_t{ B, 1, n_kv * head_dim }, 0 );
            auto v_decode = v_->view( shape_t{ B, 1, n_kv * head_dim }, 0 );

            split( qkv_out, q_decode, k_decode, v_decode, this->getExecutionContext() );
            //this->getExecutionContext()->synchronize();

            // RoPE with correct absolute position
            rope_->decode( q_decode, k_decode, position );
            //this->getExecutionContext()->synchronize();

            // GQA decode — KV cache lookup at position
            auto& attn_out = attn_->decode( q_decode, k_decode, v_decode, position );
            //this->getExecutionContext()->synchronize();

            // Output projection, T=1
            auto& out_proj_out = out_proj_->forward( attn_out );
            //this->getExecutionContext()->synchronize();

            // First residual, input + out_proj_out, T=1
            auto& res1_out = res1_->forward( input, out_proj_out );
            //this->getExecutionContext()->synchronize();

            // Post-attention RMSNorm — T=1
            auto& rms2_out = rms2_->forward( res1_out );
            //this->getExecutionContext()->synchronize();

            // Fused gate+up projection — T=1
            auto& gate_up_out = fc_gate_up_->forward( rms2_out );
            //this->getExecutionContext()->synchronize();

            // SwiGLU — T=1
            auto& swiglu_out = swiglu_->forward( gate_up_out );
            //this->getExecutionContext()->synchronize();

            // Down projection — T=1
            auto& ffn_out = fc_down_->forward( swiglu_out );
            //this->getExecutionContext()->synchronize();

            // Second residual — res1_out + ffn_out, T=1
            auto& res2_out = res2_->forward( res1_out, ffn_out );
            //this->getExecutionContext()->synchronize();

            return res2_out;
        }

        // ====================================================================
        // Backward
        // ====================================================================

        TensorType& backward( const TensorType& input, const TensorType& output_grad )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "LlamaBlock must be built before backward()." );
            }

            if ( !this->isTraining() )
            {
                throw std::runtime_error( "LlamaBlock must be in training mode to call backward()." );
            }

            if ( !forward_executed_ )
            {
                throw std::runtime_error( "LlamaBlock::backward: call forward() in training mode first." );
            }

            // 10. Backward through res2.
            auto [d_res1_from_res2, d_ffn] =
                res2_->backward( *last_res1_out_, *last_ffn_out_, output_grad );

            // 9. Backward through FFN.
            auto& d_swiglu = fc_down_->backward( *last_swiglu_out_, d_ffn );

            auto& d_gate_up = swiglu_->backward( *last_gate_up_out_, d_swiglu );

            auto& d_rms2 = fc_gate_up_->backward( *last_rms2_out_, d_gate_up );

            // 8. Backward through rms2.
            auto& d_res1_from_rms2 = rms2_->backward( *last_res1_out_, d_rms2 );

            // Accumulate res1 gradients.
            zero( *d_res1_accum_ );
            add( d_res1_from_res2, d_res1_from_rms2, *d_res1_accum_ );

            // 7. Backward through res1.
            auto [d_input_from_res1, d_out_proj] =
                res1_->backward( input, *last_out_proj_out_, *d_res1_accum_ );

            // 6. Backward through out_proj.
            auto& d_attn = out_proj_->backward( *last_attn_out_, d_out_proj );
            this->getExecutionContext()->synchronize();

            // 5. Backward through GQA — gradient w.r.t. packed QKV buffer.
            auto& d_qkv = attn_->backward( *last_qkv_out_, d_attn );
            this->getExecutionContext()->synchronize();

            // 4. Backward through RoPE — in-place inverse rotation on Q and K
            //    gradient slices. V gradient passes through unchanged.
            auto d_Q = d_qkv.view( q_shape_, 0 );
            auto d_K = d_qkv.view( k_shape_, q_offset_ );

            rope_->backward( d_Q, d_K );
            this->getExecutionContext()->synchronize();

            // 3. Backward through fused QKV projection.
            auto& d_rms1 = qkv_proj_->backward( *last_rms1_out_, d_qkv );

            // 2. Backward through rms1.
            auto& d_input_from_rms1 = rms1_->backward( input, d_rms1 );

            // Accumulate total input gradient.
            zero( *d_input_ );
            add( d_input_from_res1, d_input_from_rms1, *d_input_ );

            forward_executed_ = false;

            return *d_input_;
        }

        // ====================================================================
        // KV cache
        // ====================================================================

        bool supportsKVCache() const noexcept
        {
            return attn_ && attn_->supportsKVCache();
        }

        void resetKVCache()
        {
            if ( attn_ )
            {
                attn_->resetKVCache();
            }
        }

        // ====================================================================
        // Gradient management
        // ====================================================================

        void zeroGradients() override
        {
            if ( d_res1_accum_ )
            {
                zero( *d_res1_accum_ );
            }

            if ( d_input_ )
            {
                zero( *d_input_ );
            }

            rope_->zeroGradients();
            attn_->zeroGradients();
            qkv_proj_->zeroGradients();
            out_proj_->zeroGradients();
            rms1_->zeroGradients();
            rms2_->zeroGradients();
            res1_->zeroGradients();
            res2_->zeroGradients();
            fc_gate_up_->zeroGradients();
            swiglu_->zeroGradients();
            fc_down_->zeroGradients();
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            rms1_->save_( archive, mode );
            qkv_proj_->save_( archive, mode );
            attn_->save_( archive, mode );
            out_proj_->save_( archive, mode );
            res1_->save_( archive, mode );
            rms2_->save_( archive, mode );
            fc_gate_up_->save_( archive, mode );
            swiglu_->save_( archive, mode );
            fc_down_->save_( archive, mode );
            res2_->save_( archive, mode );
        }

        void load_( ModelArchive& archive, SerializationMode mode )
        {
            rms1_->load_( archive, mode );
            qkv_proj_->load_( archive, mode );
            attn_->load_( archive, mode );
            out_proj_->load_( archive, mode );
            res1_->load_( archive, mode );
            rms2_->load_( archive, mode );
            fc_gate_up_->load_( archive, mode );
            swiglu_->load_( archive, mode );
            fc_down_->load_( archive, mode );
            res2_->load_( archive, mode );
        }

        const ComponentType getType() const override
        {
            return ComponentType::Transformer;
        }

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;

            for ( const auto& child : this->getComponents() )
            {
                stats += child->getMemoryStats();
            }

            if ( d_res1_accum_ != nullptr )
            {
                stats.device_gradient_bytes += d_res1_accum_->getStorageSize();
            }

            if ( d_input_ != nullptr )
            {
                stats.device_gradient_bytes += d_input_->getStorageSize();
            }

            return stats;
        }

    protected:

        void onBuilding( const BuildContext& context ) override
        {
            validateBuildContext( context );

            const auto& input_shape = context.inputShape();

            const int64_t B = input_shape[ 0 ];
            const int64_t context_length = input_shape[ 1 ];
            const int64_t n_heads = config_.getNumHeads();
            const int64_t n_kv = config_.getNumKVHeads();
            const int64_t head_dim = config_.getModelDim() / n_heads;
            const int64_t hidden_dim = config_.getHiddenDimension() > 0
                ? config_.getHiddenDimension()
                : config_.getModelDim() * 4;

            // Decode view shapes — T=1, always
            q_shape_ = { B, 1, n_heads * head_dim };
            k_shape_ = { B, 1, n_kv * head_dim };
            q_offset_ = static_cast<size_t>(B * 1 * n_heads * head_dim);

            // GQA — context_length for KV cache sizing, correct QKV trailing dim
            const shape_t qkv_shape = { B, context_length, (n_heads + 2 * n_kv) * head_dim };
            BuildContext qkv_context = context.withShape( qkv_shape );

            if ( context.isInferenceMode() )
            {
                // Non-attention components built at kPrefillChunkSize —
                // owned_output_ sized to hold a full prefill chunk.
                // decode() uses resolveOutputView() to extract T=1 slice.
                const shape_t prefill_shape = { B, kPrefillChunkSize, config_.getModelDim() };
                const shape_t gate_up_prefill = { B, kPrefillChunkSize, 2 * hidden_dim };
                const shape_t hidden_prefill = { B, kPrefillChunkSize, hidden_dim };

                BuildContext prefill_context = context.withShape( prefill_shape );
                BuildContext gate_up_context = context.withShape( gate_up_prefill );
                BuildContext hidden_context = context.withShape( hidden_prefill );

                // Prefill view shapes — kPrefillChunkSize
                q_prefill_shape_ = { B, kPrefillChunkSize, n_heads * head_dim };
                k_prefill_shape_ = { B, kPrefillChunkSize, n_kv * head_dim };
                q_prefill_offset_ = static_cast<size_t>( B * kPrefillChunkSize * n_heads * head_dim);

                rms1_ = this->template getComponentAs<RmsNormType>( this->getName() + ".rmsn_1" );
                rms1_->build( prefill_context );

                qkv_proj_ = this->template getComponentAs<LinearType>( this->getName() + ".fc_qkv_proj" );
                qkv_proj_->build( prefill_context );

                rope_ = this->template getComponentAs<RopeType>( this->getName() + ".rope" );
                rope_->build( prefill_context );

                attn_ = this->template getComponentAs<AttentionType>( this->getName() + ".gqa" );
                attn_->build( qkv_context );

                out_proj_ = this->template getComponentAs<LinearType>( this->getName() + ".fc_out_proj" );
                out_proj_->build( prefill_context );

                res1_ = this->template getComponentAs<ResidualType>( this->getName() + ".res_1" );
                res1_->build( prefill_context );

                rms2_ = this->template getComponentAs<RmsNormType>(
                    this->getName() + ".rmsn_2" );
                rms2_->build( prefill_context );

                fc_gate_up_ = this->template getComponentAs<LinearType>(
                    this->getName() + ".fc_gate_up" );
                fc_gate_up_->build( prefill_context );

                swiglu_ = this->template getComponentAs<SwiGLUType>(
                    this->getName() + ".sglu" );
                swiglu_->build( gate_up_context );

                fc_down_ = this->template getComponentAs<LinearType>(
                    this->getName() + ".fc_down" );
                fc_down_->build( hidden_context );

                res2_ = this->template getComponentAs<ResidualType>(
                    this->getName() + ".res_2" );
                res2_->build( prefill_context );

                // Skip connection buffer — preserves input across attention block
                // during prefill. Cannot reuse component buffers as they get
                // overwritten by subsequent components.
                auto device = this->getExecutionContext()->getDeviceId();

                res1_prefill_ = std::make_unique<TensorType>( device, shape_t{ B, kPrefillChunkSize, config_.getModelDim() }, this->getName() + ".res_1.prefill" );

                q_ = std::make_unique<TensorType>( device, shape_t{ B, kPrefillChunkSize, n_heads * head_dim }, this->getName() + ".q" );
                k_ = std::make_unique<TensorType>( device, shape_t{ B, kPrefillChunkSize, n_kv * head_dim }, this->getName() + ".k" );
                v_ = std::make_unique<TensorType>( device, shape_t{ B, kPrefillChunkSize, n_kv * head_dim }, this->getName() + ".v" );
            }
            else
            {
                // Training — build all components at full T
                const shape_t training_shape = { B, context_length, config_.getModelDim() };
                const shape_t gate_up_shape = { B, context_length, 2 * hidden_dim };
                const shape_t hidden_shape = { B, context_length, hidden_dim };

                BuildContext training_context = context.withShape( training_shape );
                BuildContext gate_up_context = context.withShape( gate_up_shape );
                BuildContext hidden_context = context.withShape( hidden_shape );

                rms1_ = this->template getComponentAs<RmsNormType>( this->getName() + ".rmsn_1" );
                rms1_->build( training_context );

                qkv_proj_ = this->template getComponentAs<LinearType>( this->getName() + ".fc_qkv_proj" );
                qkv_proj_->build( training_context );

                rope_ = this->template getComponentAs<RopeType>( this->getName() + ".rope" );
                rope_->build( training_context );

                attn_ = this->template getComponentAs<AttentionType>( this->getName() + ".gqa" );
                attn_->build( qkv_context );

                out_proj_ = this->template getComponentAs<LinearType>(
                    this->getName() + ".fc_out_proj" );
                out_proj_->build( training_context );

                res1_ = this->template getComponentAs<ResidualType>(
                    this->getName() + ".res_1" );
                res1_->build( training_context );

                rms2_ = this->template getComponentAs<RmsNormType>(
                    this->getName() + ".rmsn_2" );
                rms2_->build( training_context );

                fc_gate_up_ = this->template getComponentAs<LinearType>(
                    this->getName() + ".fc_gate_up" );
                fc_gate_up_->build( training_context );

                swiglu_ = this->template getComponentAs<SwiGLUType>(
                    this->getName() + ".sglu" );
                swiglu_->build( gate_up_context );

                fc_down_ = this->template getComponentAs<LinearType>(
                    this->getName() + ".fc_down" );
                fc_down_->build( hidden_context );

                res2_ = this->template getComponentAs<ResidualType>(
                    this->getName() + ".res_2" );
                res2_->build( training_context );

                // Training backward scratch buffers
                auto device = this->getExecutionContext()->getDeviceId();

                d_res1_accum_ = std::make_unique<TensorType>(
                    device, training_shape,
                    this->getName() + ".d_res1_accum" );
                zero( *d_res1_accum_ );

                d_input_ = std::make_unique<TensorType>(
                    device, training_shape,
                    this->getName() + ".d_input" );
                zero( *d_input_ );
            }
        }

        void onTrainingModeChanging( TrainingMode training_mode ) override
        {
            rope_->setTrainingMode( training_mode );
            attn_->setTrainingMode( training_mode );
            rms1_->setTrainingMode( training_mode );
            rms2_->setTrainingMode( training_mode );
            qkv_proj_->setTrainingMode( training_mode );
            out_proj_->setTrainingMode( training_mode );
            res1_->setTrainingMode( training_mode );
            res2_->setTrainingMode( training_mode );
            fc_gate_up_->setTrainingMode( training_mode );
            swiglu_->setTrainingMode( training_mode );
            fc_down_->setTrainingMode( training_mode );

            forward_executed_ = false;
        }

    private:
        LlamaConfig config_;
        shape_t cached_input_shape_;
        bool forward_executed_{ false };

        shape_t q_prefill_shape_;
        shape_t k_prefill_shape_;
        size_t q_prefill_offset_;

        // Pre-computed at build — reused every forward/backward call.
        shape_t q_shape_;
        shape_t k_shape_;
        size_t q_offset_{ 0 };

        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };

        // Components
        std::shared_ptr<RmsNormType> rms1_{ nullptr };
        std::shared_ptr<LinearType> qkv_proj_{ nullptr };
        std::shared_ptr<RopeType> rope_{ nullptr };
        std::shared_ptr<AttentionType> attn_{ nullptr };
        std::shared_ptr<LinearType> out_proj_{ nullptr };
        std::shared_ptr<ResidualType> res1_{ nullptr };
        std::shared_ptr<RmsNormType> rms2_{ nullptr };
        std::shared_ptr<LinearType> fc_gate_up_{ nullptr };
        std::shared_ptr<SwiGLUType> swiglu_{ nullptr };
        std::shared_ptr<LinearType> fc_down_{ nullptr };
        std::shared_ptr<ResidualType> res2_{ nullptr };

        // Inference-only prefill buffers
        std::unique_ptr<TensorType> res1_prefill_{ nullptr };
        std::unique_ptr<TensorType> q_{ nullptr };
        std::unique_ptr<TensorType> k_{ nullptr };
        std::unique_ptr<TensorType> v_{ nullptr };

        // Backward scratch.
        std::unique_ptr<TensorType> d_res1_accum_{ nullptr };
        std::unique_ptr<TensorType> d_input_{ nullptr };

        // Cached forward activations (valid between forward() and backward()).
        TensorType* last_rms1_out_{ nullptr };
        TensorType* last_qkv_out_{ nullptr };
        TensorType* last_attn_out_{ nullptr };
        TensorType* last_out_proj_out_{ nullptr };
        TensorType* last_res1_out_{ nullptr };
        TensorType* last_rms2_out_{ nullptr };
        TensorType* last_gate_up_out_{ nullptr };
        TensorType* last_swiglu_out_{ nullptr };
        TensorType* last_ffn_out_{ nullptr };
        TensorType* last_res2_out_{ nullptr };

        // ====================================================================
        // Graph construction
        // ====================================================================

        void createGraph()
        {
            const int64_t model_dim = config_.getModelDim();
            const int64_t n_heads = config_.getNumHeads();
            const int64_t n_kv = config_.getNumKVHeads();
            const int64_t head_dim = model_dim / n_heads;
            const int64_t qkv_dim = (n_heads + 2 * n_kv) * head_dim;
            const int64_t hidden_dim = config_.getHiddenDimension() > 0 ? config_.getHiddenDimension() : model_dim * 4;

            const std::string name = this->getName();

            // Pre-attention RMSNorm.
            auto preattn_norm_config = RmsNormConfig( shape_t{ model_dim } )
                .withEpsilon( config_.getRMSNormEpsilon() )
                .withBias( false );
            auto rmsn_1 = std::make_shared<RmsNormType>( name + ".rmsn_1", preattn_norm_config );
            this->addComponent( rmsn_1 );

            // First residual.
            auto res1_cfg = ResidualConfig{};
            auto res1 = std::make_shared<ResidualType>( name + ".res_1", res1_cfg );
            this->addComponent( res1 );

            // Fused QKV projection: model_dim → (n_heads + 2*n_kv) * head_dim
            auto qkv_cfg = LinearConfig( model_dim, qkv_dim )
                .withBias( false );

            auto qkv_proj = std::make_shared<LinearType>( name + ".fc_qkv_proj", qkv_cfg );
            this->addComponent( qkv_proj );

            // RoPE.
            auto rope_cfg = RopeConfig( model_dim, n_heads, n_kv, config_.getMaxSequenceLength() )
                .withBase( config_.getRoPETheta() );

            auto rope = std::make_shared<RopeType>( name + ".rope", rope_cfg );
            this->addComponent( rope );

            // GQA.
            auto gqa_cfg = GqaConfig( model_dim, n_heads, n_kv );
            auto attn = std::make_shared<AttentionType>( name + ".gqa", gqa_cfg );
            this->addComponent( attn );

            // ----------------------------------------------------------------
            // Llama 3.2 FFN Block
            // ----------------------------------------------------------------

            // Fused gate+up projection: model_dim → 2 * hidden_dim  [gate | up]
            auto gate_up_cfg = LinearConfig( model_dim, 2 * hidden_dim )
                .withBias( false );

            auto fc_gate_up = std::make_shared<LinearType>( name + ".fc_gate_up", gate_up_cfg );
            this->addComponent( fc_gate_up );

            // SwiGLU: 2 * hidden_dim → hidden_dim.
            auto swiglu_cfg = SwigluConfig();
            auto swiglu = std::make_shared<SwiGLUType>( name + ".sglu", swiglu_cfg );
            this->addComponent( swiglu );

            // Down projection: hidden_dim → model_dim.
            auto fc_down_cfg = LinearConfig( hidden_dim, model_dim )
                .withBias( false );

            auto fc_down = std::make_shared<LinearType>( name + ".fc_down", fc_down_cfg );
            this->addComponent( fc_down );

            // ----------------------------------------------------------------

            // Output projection.
            auto out_proj_cfg = LinearConfig( model_dim, model_dim )
                .withBias( false );

            auto out_proj = std::make_shared<LinearType>( name + ".fc_out_proj", out_proj_cfg );
            this->addComponent( out_proj );

            // Post-attention RMSNorm.
            auto ln2_cfg = RmsNormConfig(shape_t{ model_dim } )
                .withEpsilon( config_.getRMSNormEpsilon() )
                .withBias( false );

            auto ln2 = std::make_shared<RmsNormType>( name + ".rmsn_2", ln2_cfg, std::nullopt );
            this->addComponent( ln2 );

            // Second residual.
            auto res2_cfg = ResidualConfig{};
            auto res2 = std::make_shared<ResidualType>( name + ".res_2", res2_cfg, std::nullopt );
            this->addComponent( res2 );
        }

        void validateBuildContext( const BuildContext& context ) const
        {
            const auto& input_shape = context.inputShape();

            if ( input_shape.size() != 3 )
            {
                throw std::invalid_argument( std::format(
                    "LlamaBlock: input must be rank 3 [B, T, model_dim], got rank {}",
                    input_shape.size() ) );
            }

            if ( input_shape.back() != static_cast<int64_t>(config_.getModelDim()) )
            {
                throw std::invalid_argument( std::format(
                    "LlamaBlock: model dim mismatch — expected {}, got {}",
                    config_.getModelDim(), input_shape.back() ) );
            }
        }

        void validateInputShape( const shape_t& input_shape ) const
        {
            if ( input_shape.size() != 3 )
            {
                throw std::invalid_argument(
                    "LlamaBlock: input must be [B, T, model_dim]" );
            }

            if ( input_shape.back() != static_cast<int64_t>(config_.getModelDim()) )
            {
                std::ostringstream oss;
                oss << "LlamaBlock: model_dim mismatch — expected "
                    << config_.getModelDim() << ", got " << input_shape.back();
                throw std::invalid_argument( oss.str() );
            }
        }
    };
}
