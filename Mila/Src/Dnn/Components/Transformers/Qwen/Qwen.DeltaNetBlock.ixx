/**
 * @file Qwen.DeltaNetBlock.ixx
 * @brief Qwen 3.8 Gated DeltaNet decoder block (inference: prefill + decode).
 *
 * The other of the two block kinds in the 4-layer interleave -- three of these per
 * QwenAttentionBlock. A separate class rather than a flag, because the two kinds differ in
 * the MIXER itself and share no projection shape (Specifications/Qwen3.8.md section 8).
 *
 * Same pre-norm skeleton as the attention block; only the mixer differs:
 *
 *   x  <- x + out_proj( gated_norm( delta_rule( conv(qk), conv(v), a, b ), z ) )
 *   x  <- x + down( swiglu( gate_up( post_norm(x) ) ) )
 *
 * THREE THINGS HAVE NO PRECEDENT IN THE TREE, and each is why a piece was built first:
 *
 *  - The mixer's memory is a fixed-size RECURRENT STATE, not a growing KV cache. Nothing
 *    here scales with context length, which is the property that lets 48 of 64 layers cost
 *    nothing extra at long context (section 3).
 *  - A short depthwise CAUSAL CONVOLUTION sits between the projections and the mixer,
 *    carrying its own rolling window of the last kernel_width - 1 positions.
 *  - The output norm is GATED and, unlike every other norm in this family, uses the RAW
 *    weight rather than the unit-offset form. Both conventions live in one model; see the
 *    note at createGraph().
 *
 * PER-ROLE PRECISION. The plan splits this block's projections three ways, and the split is
 * not cosmetic -- section 5's parameter counts only add up this way:
 *
 *   DeltaNetQueryKey        q, k                  1.007 B
 *   DeltaNetValueGateOutput v, z, out_proj        4.531 B
 *   DeltaNetGating          a, b                  0.024 B, NEVER quantized
 *
 * `DeltaNetGating` is `NoWeightQuant` on purpose: a and b drive the forget gate, where error
 * compounds EXPONENTIALLY over the sequence rather than linearly.
 *
 * WEIGHT LAYOUT CONTRACT. The checkpoint fuses q, k and v into one `in_proj_qkv`
 * [10240, 5120]; this block takes them as TWO projections, [q|k] at 4096 and [v] at 6144,
 * because one tensor cannot carry two storage policies and the plan puts q/k a half-step
 * above v. The converter owns that split, and splits `conv1d.weight` [10240, 4] the same
 * way. Splitting the convolution is exact rather than approximate: it is DEPTHWISE, so no
 * channel ever reads another, and two convolutions over a partition of the channels compute
 * what one convolution over all of them computes.
 *
 * Inference-only, like the other blocks: implements IDecoderLayer, not forward/backward.
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

export module Dnn.Components.QwenDeltaNetBlock;

export import Dnn.Components.QwenConfig;
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
import Dnn.Components.RmsNorm;
import Dnn.Components.Residual;
import Dnn.Components.Linear;
import Dnn.Components.Swiglu;
import Dnn.Components.Activation;
import Dnn.Components.AttentionOutputGate;
import Dnn.Components.CausalConv1d;
import Dnn.Components.GatedDeltaRule;
import Serialization.ModelArchive;
import Serialization.Mode;
import Dnn.Quantization.Weight.Policies;
import Dnn.Quantization.Weight.PrecisionPlan;
import Dnn.Components.QwenPrecisionPlan;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;
    using namespace Mila::Dnn::Quant::Weight;

    /**
     * @brief One Gated DeltaNet decoder block.
     *
     * @tparam TWeightPlan A precision plan, or a bare policy meaning "uniform". Lifted
     *                     through QwenPrecisionPlanFor rather than the generic
     *                     PrecisionPlanFor, because the generic uniform plan deliberately
     *                     does not name this family's DeltaNet roles.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision,
        typename TWeightPlan = QwenReferencePrecisionPlan>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
              && FeedForwardPrecisionRoles<QwenPrecisionPlanFor<TWeightPlan>>
              && DeltaNetPrecisionRoles<QwenPrecisionPlanFor<TWeightPlan>>
    class QwenDeltaNetBlock
        : public CompositeComponent<TDeviceType, TPrecision>,
          public IDecoderLayer<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using CompositeComponentBase = CompositeComponent<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;

        /// The resolved plan. Named so a caller can ask a block what it decided.
        using PrecisionPlan = QwenPrecisionPlanFor<TWeightPlan>;

        using RmsNormType = RmsNorm<TDeviceType, TPrecision>;
        using ResidualType = Residual<TDeviceType, TPrecision>;
        using SwigluType = Swiglu<TDeviceType, TPrecision, ActivationType::Silu>;
        using ConvType = CausalConv1d<TDeviceType, TPrecision>;
        using ConvActivationType = Activation<TDeviceType, TPrecision, ActivationType::Silu>;
        using DeltaRuleType = GatedDeltaRule<TDeviceType, TPrecision>;

        // The output gate. Named for attention because that is where it first appeared, and
        // mechanically generic (out = silu(gate) * value) -- the rename is an open question,
        // not a reason to write a second implementation of one concept.
        using OutputGateType = AttentionOutputGate<TDeviceType, TPrecision, ActivationType::Silu>;

        using QueryKeyProjectionType = Linear<TDeviceType, TPrecision, typename PrecisionPlan::DeltaNetQueryKey>;
        using ValueProjectionType = Linear<TDeviceType, TPrecision, typename PrecisionPlan::DeltaNetValueGateOutput>;
        using GateProjectionType = Linear<TDeviceType, TPrecision, typename PrecisionPlan::DeltaNetValueGateOutput>;
        using OutputProjectionType = Linear<TDeviceType, TPrecision, typename PrecisionPlan::DeltaNetValueGateOutput>;
        using GatingProjectionType = Linear<TDeviceType, TPrecision, typename PrecisionPlan::DeltaNetGating>;
        using FeedForwardGateUpType = Linear<TDeviceType, TPrecision, typename PrecisionPlan::FeedForwardGateUp>;
        using FeedForwardDownType = Linear<TDeviceType, TPrecision, typename PrecisionPlan::FeedForwardDown>;

        explicit QwenDeltaNetBlock( const std::string& name, const QwenConfig& config,
            std::optional<DeviceId> device_id = std::nullopt )
            : CompositeComponentBase( name ), config_( config )
        {
            config_.validate();

            createGraph();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                {
                    throw std::invalid_argument( "QwenDeltaNetBlock: device type mismatch" );
                }

                owned_exec_context_ = createExecutionContext( device_id.value() );
                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~QwenDeltaNetBlock() override = default;

        // ====================================================================
        // Geometry
        // ====================================================================

        dim_t queryKeyWidth() const noexcept
        {
            return 2 * config_.getLinearNumKeyHeads() * config_.getLinearHeadDim();
        }

        dim_t valueWidth() const noexcept
        {
            return config_.getLinearNumValueHeads() * config_.getLinearHeadDim();
        }

        dim_t gatingWidth() const noexcept { return config_.getLinearNumValueHeads(); }

        // ====================================================================
        // IDecoderLayer -- inference path
        // ====================================================================

        TensorType& prefill( const TensorType& input, dim_t position_offset ) override
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "QwenDeltaNetBlock::prefill: must be built before prefill()." );

            return run( input, input.shape()[ 0 ], input.shape()[ 1 ], position_offset, false );
        }

        TensorType& decode( const TensorType& input, dim_t position ) override
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "QwenDeltaNetBlock::decode: must be built before decode()." );

            return run( input, input.shape()[ 0 ], 1, position, true );
        }

        /**
         * @brief No-op: this block has no grouped-query attention and no shared GQA workspace.
         *
         * The interface names a concrete GqaState because it was written for a family whose
         * blocks all attend (Qwen3.8.md section 7 records the mismatch). Accepting and
         * ignoring it is honest here -- there is no attention transient to wire -- and the
         * alternative, generalizing the interface, is a change across three families.
         */
        void setState( const GqaState& ) override
        {
        }

        /**
         * @brief False: the mixer's memory is a recurrent state, not a key/value cache.
         *
         * Not a limitation being reported -- a different mechanism. The state is already
         * fixed-size and always live; there is no cache path to support or refuse.
         */
        bool supportsKVCache() const noexcept override
        {
            return false;
        }

        /// Start a new generation session: drop the recurrent state and the conv window.
        void resetKVCache() override
        {
            if ( delta_rule_ )
                delta_rule_->resetState();

            if ( conv_qk_ )
                conv_qk_->resetState();

            if ( conv_v_ )
                conv_v_->resetState();
        }

        /**
         * @brief Always false: a recurrent state cannot be rewound.
         *
         * A KV cache stores each position separately, so dropping a suffix is exact. This
         * block's state is a LOSSY SUMMARY of every position it has seen -- the information
         * needed to undo the last N steps is not in it. Returning true would silently
         * corrupt a prefix-reuse session rather than fail it, so prompt-prefix reuse is
         * unavailable for any model containing these layers.
         */
        bool rewindKvCache( dim_t ) override
        {
            return false;
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

            return stats;
        }

        MemoryStats getRequiredMemory( const BuildContext& context ) const override
        {
            validateBuildContext( context );

            const BlockContexts contexts = resolveBlockContexts( context );
            const std::string n = this->getName();

            MemoryStats stats;

            auto required = [&]( const auto& component, const BuildContext& child_context )
            {
                return component->getRequiredMemory( child_context );
            };

            stats += required( this->template getComponentAs<RmsNormType>( n + ".input_norm" ), contexts.stream );
            stats += required( this->template getComponentAs<QueryKeyProjectionType>( n + ".fc_in_proj_qk" ), contexts.stream );
            stats += required( this->template getComponentAs<ValueProjectionType>( n + ".fc_in_proj_v" ), contexts.stream );
            stats += required( this->template getComponentAs<GateProjectionType>( n + ".fc_in_proj_z" ), contexts.stream );
            stats += required( this->template getComponentAs<GatingProjectionType>( n + ".fc_in_proj_a" ), contexts.stream );
            stats += required( this->template getComponentAs<GatingProjectionType>( n + ".fc_in_proj_b" ), contexts.stream );
            stats += required( this->template getComponentAs<ConvType>( n + ".conv_qk" ), contexts.query_key );
            stats += required( this->template getComponentAs<ConvType>( n + ".conv_v" ), contexts.value );
            stats += required( this->template getComponentAs<ConvActivationType>( n + ".conv_act_qk" ), contexts.query_key );
            stats += required( this->template getComponentAs<ConvActivationType>( n + ".conv_act_v" ), contexts.value );
            stats += required( this->template getComponentAs<DeltaRuleType>( n + ".delta_rule" ), contexts.value );
            stats += required( this->template getComponentAs<RmsNormType>( n + ".norm_gate" ), contexts.per_head );
            stats += required( this->template getComponentAs<OutputGateType>( n + ".output_gate" ), contexts.value );
            stats += required( this->template getComponentAs<OutputProjectionType>( n + ".fc_out_proj" ), contexts.value );
            stats += required( this->template getComponentAs<ResidualType>( n + ".res_1" ), contexts.stream );
            stats += required( this->template getComponentAs<RmsNormType>( n + ".post_attn_norm" ), contexts.stream );
            stats += required( this->template getComponentAs<FeedForwardGateUpType>( n + ".fc_gate_up" ), contexts.stream );
            stats += required( this->template getComponentAs<SwigluType>( n + ".swiglu" ), contexts.gate_up );
            stats += required( this->template getComponentAs<FeedForwardDownType>( n + ".fc_down" ), contexts.hidden );
            stats += required( this->template getComponentAs<ResidualType>( n + ".res_2" ), contexts.stream );

            stats.device_state_bytes += storageBytes<TPrecision>(
                contexts.batch * contexts.chunk * (config_.getLinearNumKeyHeads() * config_.getLinearHeadDim()) * 2 );

            return stats;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "QwenDeltaNetBlock: " << this->getName() << std::endl;
            oss << "Device: " << deviceTypeToString( this->getDeviceType() ) << std::endl;
            oss << "DeltaNet: " << config_.getLinearNumKeyHeads() << " key heads / "
                << config_.getLinearNumValueHeads() << " value heads, head_dim "
                << config_.getLinearHeadDim() << std::endl;
            oss << "Conv kernel: " << config_.getLinearConvKernelDim() << std::endl;

            return oss.str();
        }

    protected:

        struct BlockContexts
        {
            BuildContext stream;
            BuildContext query_key;
            BuildContext value;
            BuildContext per_head;
            BuildContext gate_up;
            BuildContext hidden;

            dim_t batch{ 0 };
            dim_t chunk{ 0 };
        };

        BlockContexts resolveBlockContexts( const BuildContext& context ) const
        {
            const auto& input_shape = context.inputShape();

            BlockContexts contexts;
            contexts.batch = input_shape[ 0 ];
            contexts.chunk = context.getPrefillSize();

            const dim_t B = contexts.batch;
            const dim_t chunk = contexts.chunk;
            const dim_t model_dim = config_.getModelDim();
            const dim_t hidden_dim = config_.getHiddenDimension();

            contexts.stream = context.withShape( shape_t{ B, chunk, model_dim } );
            contexts.query_key = context.withShape( shape_t{ B, chunk, queryKeyWidth() } );
            contexts.value = context.withShape( shape_t{ B, chunk, valueWidth() } );
            contexts.per_head = context.withShape( shape_t{
                B, chunk * config_.getLinearNumValueHeads(), config_.getLinearHeadDim() } );
            contexts.gate_up = context.withShape( shape_t{ B, chunk, 2 * hidden_dim } );
            contexts.hidden = context.withShape( shape_t{ B, chunk, hidden_dim } );

            return contexts;
        }

        void onBuilding( const BuildContext& context ) override
        {
            validateBuildContext( context );

            const BlockContexts contexts = resolveBlockContexts( context );
            const std::string n = this->getName();

            input_norm_ = this->template getComponentAs<RmsNormType>( n + ".input_norm" );
            input_norm_->build( contexts.stream );

            qk_proj_ = this->template getComponentAs<QueryKeyProjectionType>( n + ".fc_in_proj_qk" );
            qk_proj_->build( contexts.stream );

            v_proj_ = this->template getComponentAs<ValueProjectionType>( n + ".fc_in_proj_v" );
            v_proj_->build( contexts.stream );

            z_proj_ = this->template getComponentAs<GateProjectionType>( n + ".fc_in_proj_z" );
            z_proj_->build( contexts.stream );

            a_proj_ = this->template getComponentAs<GatingProjectionType>( n + ".fc_in_proj_a" );
            a_proj_->build( contexts.stream );

            b_proj_ = this->template getComponentAs<GatingProjectionType>( n + ".fc_in_proj_b" );
            b_proj_->build( contexts.stream );

            conv_qk_ = this->template getComponentAs<ConvType>( n + ".conv_qk" );
            conv_qk_->build( contexts.query_key );

            conv_v_ = this->template getComponentAs<ConvType>( n + ".conv_v" );
            conv_v_->build( contexts.value );

            conv_act_qk_ = this->template getComponentAs<ConvActivationType>( n + ".conv_act_qk" );
            conv_act_qk_->build( contexts.query_key );

            conv_act_v_ = this->template getComponentAs<ConvActivationType>( n + ".conv_act_v" );
            conv_act_v_->build( contexts.value );

            delta_rule_ = this->template getComponentAs<DeltaRuleType>( n + ".delta_rule" );
            delta_rule_->build( contexts.value );

            norm_gate_ = this->template getComponentAs<RmsNormType>( n + ".norm_gate" );
            norm_gate_->build( contexts.per_head );

            output_gate_ = this->template getComponentAs<OutputGateType>( n + ".output_gate" );
            output_gate_->build( contexts.value );

            out_proj_ = this->template getComponentAs<OutputProjectionType>( n + ".fc_out_proj" );
            out_proj_->build( contexts.value );

            res1_ = this->template getComponentAs<ResidualType>( n + ".res_1" );
            res1_->build( contexts.stream );

            post_attn_norm_ = this->template getComponentAs<RmsNormType>( n + ".post_attn_norm" );
            post_attn_norm_->build( contexts.stream );

            fc_gate_up_ = this->template getComponentAs<FeedForwardGateUpType>( n + ".fc_gate_up" );
            fc_gate_up_->build( contexts.stream );

            swiglu_ = this->template getComponentAs<SwigluType>( n + ".swiglu" );
            swiglu_->build( contexts.gate_up );

            fc_down_ = this->template getComponentAs<FeedForwardDownType>( n + ".fc_down" );
            fc_down_->build( contexts.hidden );

            res2_ = this->template getComponentAs<ResidualType>( n + ".res_2" );
            res2_->build( contexts.stream );

            auto device = this->getExecutionContext()->getDeviceId();
            const dim_t head_width = config_.getLinearNumKeyHeads() * config_.getLinearHeadDim();

            q_ = std::make_shared<TensorType>(
                device, shape_t{ contexts.batch, contexts.chunk, head_width }, n + ".q" );
            k_ = std::make_shared<TensorType>(
                device, shape_t{ contexts.batch, contexts.chunk, head_width }, n + ".k" );
        }

        void onTrainingModeChanging( TrainingMode training_mode ) override
        {
            for ( const auto& child : this->getComponents() )
                child->setTrainingMode( training_mode );
        }

    private:
        QwenConfig config_;
        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };

        std::shared_ptr<RmsNormType> input_norm_{ nullptr };
        std::shared_ptr<QueryKeyProjectionType> qk_proj_{ nullptr };
        std::shared_ptr<ValueProjectionType> v_proj_{ nullptr };
        std::shared_ptr<GateProjectionType> z_proj_{ nullptr };
        std::shared_ptr<GatingProjectionType> a_proj_{ nullptr };
        std::shared_ptr<GatingProjectionType> b_proj_{ nullptr };
        std::shared_ptr<ConvType> conv_qk_{ nullptr };
        std::shared_ptr<ConvType> conv_v_{ nullptr };
        std::shared_ptr<ConvActivationType> conv_act_qk_{ nullptr };
        std::shared_ptr<ConvActivationType> conv_act_v_{ nullptr };
        std::shared_ptr<DeltaRuleType> delta_rule_{ nullptr };
        std::shared_ptr<RmsNormType> norm_gate_{ nullptr };
        std::shared_ptr<OutputGateType> output_gate_{ nullptr };
        std::shared_ptr<OutputProjectionType> out_proj_{ nullptr };
        std::shared_ptr<ResidualType> res1_{ nullptr };
        std::shared_ptr<RmsNormType> post_attn_norm_{ nullptr };
        std::shared_ptr<FeedForwardGateUpType> fc_gate_up_{ nullptr };
        std::shared_ptr<SwigluType> swiglu_{ nullptr };
        std::shared_ptr<FeedForwardDownType> fc_down_{ nullptr };
        std::shared_ptr<ResidualType> res2_{ nullptr };

        // The q/k split of the convolved query-key stream.
        std::shared_ptr<TensorType> q_{ nullptr };
        std::shared_ptr<TensorType> k_{ nullptr };

        TensorType& run( const TensorType& input, dim_t B, dim_t T, dim_t position, bool is_decode )
        {
            const dim_t head_width = config_.getLinearNumKeyHeads() * config_.getLinearHeadDim();
            const dim_t heads = config_.getLinearNumValueHeads();
            const dim_t head_dim = config_.getLinearHeadDim();

            // --- DeltaNet mixer ---------------------------------------------
            auto& normed = input_norm_->forward( input );

            auto& qk = qk_proj_->forward( normed );
            auto& v_projected = v_proj_->forward( normed );
            auto& z = z_proj_->forward( normed );
            auto& a = a_proj_->forward( normed );
            auto& b = b_proj_->forward( normed );

            // Short causal convolution, then SiLU, over the projected streams. Splitting
            // the convolution across [q|k] and [v] is exact because it is depthwise.
            auto& conv_qk_out = is_decode
                ? conv_qk_->decode( qk, position )
                : conv_qk_->prefill( qk, position );
            auto& conv_v_out = is_decode
                ? conv_v_->decode( v_projected, position )
                : conv_v_->prefill( v_projected, position );

            auto& qk_activated = conv_act_qk_->forward( conv_qk_out );
            auto& v_activated = conv_act_v_->forward( conv_v_out );

            auto q = q_->view( shape_t{ B, T, head_width }, 0 );
            auto k = k_->view( shape_t{ B, T, head_width }, 0 );

            split( qk_activated, q, k, this->getExecutionContext() );

            auto& core = is_decode
                ? delta_rule_->decode( q, k, v_activated, a, b, position )
                : delta_rule_->prefill( q, k, v_activated, a, b, position );

            // Gated output norm, per value head. RAW weight -- see createGraph().
            auto core_per_head = core.view( shape_t{ B, T * heads, head_dim }, 0 );
            auto& core_normed = norm_gate_->forward( core_per_head );

            auto core_flat = core_normed.view( shape_t{ B, T, heads * head_dim }, 0 );
            auto& gated = output_gate_->forward( z, core_flat );

            auto& mixed = out_proj_->forward( gated );
            auto& res1 = res1_->forward( input, mixed );

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

            // TWO NORM CONVENTIONS IN ONE BLOCK, and the difference is load-bearing. The
            // stream norms use Qwen's unit-offset form, x_norm * (1 + weight), with weights
            // stored zero-centered. The GATED norm inside the mixer does not: its reference
            // (Qwen3_5RMSNormGated) multiplies by the raw weight, initialized to ones. Give
            // norm_gate the unit offset and it silently adds one to every scale.
            auto stream_rms = [&]( const shape_t& shape )
            {
                return RmsNormConfig( shape ).withEpsilon( eps ).withBias( false ).withUnitOffset( 1.0f );
            };

            auto raw_rms = [&]( const shape_t& shape )
            {
                return RmsNormConfig( shape ).withEpsilon( eps ).withBias( false );
            };

            this->addComponent( std::make_shared<RmsNormType>( n + ".input_norm", stream_rms( shape_t{ model_dim } ) ) );
            this->addComponent( std::make_shared<RmsNormType>( n + ".post_attn_norm", stream_rms( shape_t{ model_dim } ) ) );

            // Three projection roles, three storage policies -- see the file header.
            this->addComponent( std::make_shared<QueryKeyProjectionType>(
                n + ".fc_in_proj_qk", LinearConfig( model_dim, queryKeyWidth() ).withBias( false ) ) );
            this->addComponent( std::make_shared<ValueProjectionType>(
                n + ".fc_in_proj_v", LinearConfig( model_dim, valueWidth() ).withBias( false ) ) );
            this->addComponent( std::make_shared<GateProjectionType>(
                n + ".fc_in_proj_z", LinearConfig( model_dim, valueWidth() ).withBias( false ) ) );
            this->addComponent( std::make_shared<GatingProjectionType>(
                n + ".fc_in_proj_a", LinearConfig( model_dim, gatingWidth() ).withBias( false ) ) );
            this->addComponent( std::make_shared<GatingProjectionType>(
                n + ".fc_in_proj_b", LinearConfig( model_dim, gatingWidth() ).withBias( false ) ) );

            const dim_t kernel = config_.getLinearConvKernelDim();

            this->addComponent( std::make_shared<ConvType>(
                n + ".conv_qk", CausalConv1dConfig( queryKeyWidth(), kernel ) ) );
            this->addComponent( std::make_shared<ConvType>(
                n + ".conv_v", CausalConv1dConfig( valueWidth(), kernel ) ) );

            this->addComponent( std::make_shared<ConvActivationType>(
                n + ".conv_act_qk", ActivationConfig() ) );
            this->addComponent( std::make_shared<ConvActivationType>(
                n + ".conv_act_v", ActivationConfig() ) );

            this->addComponent( std::make_shared<DeltaRuleType>( n + ".delta_rule",
                GatedDeltaRuleConfig( config_.getLinearNumKeyHeads(), config_.getLinearNumValueHeads(),
                    config_.getLinearHeadDim(), config_.getLinearHeadDim() ) ) );

            this->addComponent( std::make_shared<RmsNormType>(
                n + ".norm_gate", raw_rms( shape_t{ config_.getLinearHeadDim() } ) ) );

            this->addComponent( std::make_shared<OutputGateType>(
                n + ".output_gate", AttentionOutputGateConfig() ) );

            this->addComponent( std::make_shared<OutputProjectionType>(
                n + ".fc_out_proj", LinearConfig( valueWidth(), model_dim ).withBias( false ) ) );

            this->addComponent( std::make_shared<ResidualType>( n + ".res_1", ResidualConfig{} ) );

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
                    "QwenDeltaNetBlock: input must be rank 3 [B, T, model_dim], got rank {}", s.size() ) );

            if ( s.back() != config_.getModelDim() )
                throw std::invalid_argument( std::format(
                    "QwenDeltaNetBlock: model_dim mismatch -- expected {}, got {}",
                    config_.getModelDim(), s.back() ) );
        }
    };
}
