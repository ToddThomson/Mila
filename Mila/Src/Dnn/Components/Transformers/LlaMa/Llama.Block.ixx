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
import Dnn.Components.GroupedQueryAttention;
import Dnn.Components.Residual;
import Dnn.Components.Linear;
import Dnn.Components.Swiglu;
import Serialization.ModelArchive;
import Serialization.Mode;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

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

        // ====================================================================
        // Forward
        // ====================================================================

        TensorType& forward( const TensorType& input )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "LlamaBlock must be built before forward()." );
            }

            // 1. Pre-attention RMSNorm.
            auto& rms1_out = rms1_->forward( input );
            this->getExecutionContext()->synchronize();

            // 2. Fused QKV projection — single GEMM.
            //    Output: [B, T, (n_heads + 2*n_kv_heads) * head_dim]
            auto& qkv_out = qkv_proj_->forward( rms1_out );
            this->getExecutionContext()->synchronize();

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

            forward_executed_ = this->isTraining();

            return res2_out;
        }

        // ====================================================================
        // Decode (inference / KV-cache path)
        // ====================================================================

        TensorType& decode( const TensorType& input, int position )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "LlamaBlock must be built before decode()." );
            }

            auto& rms1_out = rms1_->forward( input );
            this->getExecutionContext()->synchronize();

            auto& qkv_out = qkv_proj_->forward( rms1_out );
            this->getExecutionContext()->synchronize();

            auto Q = qkv_out.view( q_shape_, 0 );
            auto K = qkv_out.view( k_shape_, q_offset_ );

            rope_->forward( Q, K );
            this->getExecutionContext()->synchronize();

            auto& attn_out = attn_->decode( qkv_out, position );
            this->getExecutionContext()->synchronize();

            auto& out_proj_out = out_proj_->forward( attn_out );
            this->getExecutionContext()->synchronize();

            auto& res1_out = res1_->forward( input, out_proj_out );
            this->getExecutionContext()->synchronize();

            auto& rms2_out = rms2_->forward( res1_out );
            this->getExecutionContext()->synchronize();

            auto& gate_up_out = fc_gate_up_->forward( rms2_out );
            this->getExecutionContext()->synchronize();

            auto& swiglu_out = swiglu_->forward( gate_up_out );
            this->getExecutionContext()->synchronize();

            auto& ffn_out = fc_down_->forward( swiglu_out );
            this->getExecutionContext()->synchronize();

            auto& res2_out = res2_->forward( res1_out, ffn_out );
            this->getExecutionContext()->synchronize();

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

        // ====================================================================
        // Build
        // ====================================================================

        void onBuilding( const shape_t& input_shape ) override
        {
            validateInputShape( input_shape );

            cached_input_shape_ = input_shape;

            const int64_t B = input_shape[ 0 ];
            const int64_t T = input_shape[ 1 ];
            const int64_t n_heads = config_.getNumHeads();
            const int64_t n_kv = config_.getNumKVHeads();
            const int64_t head_dim = config_.getModelDim() / n_heads;

            const int64_t hidden_dim = config_.getHiddenDimension() > 0
                ? config_.getHiddenDimension()
                : config_.getModelDim() * 4;

            // Leading shape { B, T } is used to derive Q and K view shapes for RoPE.
            auto leading_shape = shape_t{ B, T };

            // Pre-computed view shapes and Q→K offset — reused every forward/backward.
            q_shape_ = { B, T, n_heads * head_dim };
            k_shape_ = { B, T, n_kv * head_dim };
            q_offset_ = static_cast<size_t>(B * T * n_heads * head_dim);

            const shape_t qkv_shape = { B, T, (n_heads + 2 * n_kv) * head_dim };
            const shape_t gate_up_shape = { B, T, 2 * hidden_dim };
            const shape_t hidden_shape = { B, T, hidden_dim };

            // 1. Pre-attention RMSNorm.
            rms1_ = this->template getComponentAs<RmsNormType>( this->getName() + ".ln_1" );
            rms1_->build( input_shape );

            // 2. Fused QKV projection.
            qkv_proj_ = this->template getComponentAs<LinearType>( this->getName() + ".fc_qkv_proj" );
            qkv_proj_->build( input_shape );

            // 3. RoPE — paired build with Q and K shapes.
            rope_ = this->template getComponentAs<RopeType>( this->getName() + ".rope" );
            rope_->build( leading_shape );

            // 4. GQA.
            attn_ = this->template getComponentAs<AttentionType>( this->getName() + ".attn" );
            attn_->build( qkv_shape );

            // 5. Output projection.
            out_proj_ = this->template getComponentAs<LinearType>( this->getName() + ".fc_out_proj" );
            out_proj_->build( input_shape );

            // 6. First residual.
            res1_ = this->template getComponentAs<ResidualType>( this->getName() + ".res_1" );
            res1_->build( input_shape );

            // 7. Post-attention RMSNorm.
            rms2_ = this->template getComponentAs<RmsNormType>( this->getName() + ".ln_2" );
            rms2_->build( input_shape );

            // 8. Fused gate+up projection.
            fc_gate_up_ = this->template getComponentAs<LinearType>( this->getName() + ".fc_gate_up" );
            fc_gate_up_->build( input_shape );

            // 9. SwiGLU.
            swiglu_ = this->template getComponentAs<SwiGLUType>( this->getName() + ".swiglu" );
            swiglu_->build( gate_up_shape );

            // 10. Down projection.
            fc_down_ = this->template getComponentAs<LinearType>( this->getName() + ".fc_down" );
            fc_down_->build( hidden_shape );

            // 11. Second residual.
            res2_ = this->template getComponentAs<ResidualType>( this->getName() + ".res_2" );
            res2_->build( input_shape );

            // Backward scratch buffers.
            auto device = this->getDeviceId();

            d_res1_accum_ = std::make_unique<TensorType>( device, input_shape );
            d_res1_accum_->setName( this->getName() + ".d_res1_accum" );
            zero( *d_res1_accum_ );

            d_input_ = std::make_unique<TensorType>( device, input_shape );
            d_input_->setName( this->getName() + ".d_input" );
            zero( *d_input_ );
        }

        void onTrainingChanging( bool is_training ) override
        {
            if ( rope_ )
            {
                rope_->setTraining( is_training );
            }

            if ( attn_ )
            {
                attn_->setTraining( is_training );
            }

            if ( rms1_ )
            {
                rms1_->setTraining( is_training );
            }

            if ( rms2_ )
            {
                rms2_->setTraining( is_training );
            }

            if ( qkv_proj_ )
            {
                qkv_proj_->setTraining( is_training );
            }

            if ( out_proj_ )
            {
                out_proj_->setTraining( is_training );
            }

            if ( res1_ )
            {
                res1_->setTraining( is_training );
            }

            if ( res2_ )
            {
                res2_->setTraining( is_training );
            }

            if ( fc_gate_up_ )
            {
                fc_gate_up_->setTraining( is_training );
            }

            if ( swiglu_ )
            {
                swiglu_->setTraining( is_training );
            }

            if ( fc_down_ )
            {
                fc_down_->setTraining( is_training );
            }

            forward_executed_ = false;
        }

    private:
        LlamaConfig config_;
        shape_t cached_input_shape_;
        bool forward_executed_{ false };

        // Pre-computed at build — reused every forward/backward call.
        shape_t q_shape_;
        shape_t k_shape_;
        size_t q_offset_{ 0 };

        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };

        // Components — named to match convert_llama32.py tensor mapping.
        std::shared_ptr<RmsNormType>   rms1_{ nullptr };
        std::shared_ptr<LinearType>    qkv_proj_{ nullptr };
        std::shared_ptr<RopeType>      rope_{ nullptr };
        std::shared_ptr<AttentionType> attn_{ nullptr };
        std::shared_ptr<LinearType>    out_proj_{ nullptr };
        std::shared_ptr<ResidualType>  res1_{ nullptr };
        std::shared_ptr<RmsNormType>   rms2_{ nullptr };
        std::shared_ptr<LinearType>    fc_gate_up_{ nullptr };
        std::shared_ptr<SwiGLUType>    swiglu_{ nullptr };
        std::shared_ptr<LinearType>    fc_down_{ nullptr };
        std::shared_ptr<ResidualType>  res2_{ nullptr };

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
            const int64_t hidden_dim = config_.getHiddenDimension() > 0
                ? config_.getHiddenDimension()
                : model_dim * 4;

            const std::string name = this->getName();

            // Pre-attention RMSNorm.
            auto ln1_cfg = RmsNormConfig( shape_t{ model_dim } )
                .withEpsilon( config_.getRMSNormEpsilon() )
                .withBias( false );

            auto ln1 = std::make_shared<RmsNormType>( name + ".ln_1", ln1_cfg, std::nullopt );
            this->addComponent( ln1 );

            // Fused QKV projection: model_dim → (n_heads + 2*n_kv) * head_dim
            auto qkv_cfg = LinearConfig( model_dim, qkv_dim )
                .withBias( false );

            auto qkv_proj = std::make_shared<LinearType>( name + ".fc_qkv_proj", qkv_cfg, std::nullopt );
            this->addComponent( qkv_proj );

            // RoPE.
            auto rope_cfg = RopeConfig( model_dim, n_heads, n_kv, config_.getMaxSequenceLength() )
                .withBase( config_.getRoPETheta() );

            auto rope = std::make_shared<RopeType>( name + ".rope", rope_cfg, std::nullopt );
            this->addComponent( rope );

            // GQA.
            auto gqa_cfg = GroupedQueryAttentionConfig( model_dim, n_heads, n_kv );

            auto attn = std::make_shared<AttentionType>( name + ".attn", gqa_cfg, std::nullopt );
            this->addComponent( attn );

            // Output projection.
            auto out_proj_cfg = LinearConfig( model_dim, model_dim )
                .withBias( false );

            auto out_proj = std::make_shared<LinearType>( name + ".fc_out_proj", out_proj_cfg, std::nullopt );
            this->addComponent( out_proj );

            // First residual.
            auto res1_cfg = ResidualConfig{};
            auto res1 = std::make_shared<ResidualType>( name + ".res_1", res1_cfg, std::nullopt );
            this->addComponent( res1 );

            // Post-attention RMSNorm.
            auto ln2_cfg = RmsNormConfig(shape_t{ model_dim } )
                .withEpsilon( config_.getRMSNormEpsilon() )
                .withBias( false );

            auto ln2 = std::make_shared<RmsNormType>( name + ".ln_2", ln2_cfg, std::nullopt );
            this->addComponent( ln2 );

            // Fused gate+up projection: model_dim → 2*hidden_dim  [gate | up]
            auto gate_up_cfg = LinearConfig( model_dim, 2 * hidden_dim )
                .withBias( false );

            auto fc_gate_up = std::make_shared<LinearType>( name + ".fc_gate_up", gate_up_cfg, std::nullopt );
            this->addComponent( fc_gate_up );

            // SwiGLU: 2*hidden_dim → hidden_dim.
            auto swiglu_cfg = SwigluConfig();
            auto swiglu = std::make_shared<SwiGLUType>( name + ".swiglu", swiglu_cfg, std::nullopt );
            this->addComponent( swiglu );

            // Down projection: hidden_dim → model_dim.
            auto fc_down_cfg = LinearConfig( hidden_dim, model_dim )
                .withBias( false );

            auto fc_down = std::make_shared<LinearType>( name + ".fc_down", fc_down_cfg, std::nullopt );
            this->addComponent( fc_down );

            // Second residual.
            auto res2_cfg = ResidualConfig{};
            auto res2 = std::make_shared<ResidualType>( name + ".res_2", res2_cfg, std::nullopt );
            this->addComponent( res2 );
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
