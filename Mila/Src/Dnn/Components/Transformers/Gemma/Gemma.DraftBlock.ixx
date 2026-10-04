/**
 * @file Gemma.DraftBlock.ixx
 * @brief One layer of Gemma 4's draft model: a Gemma block that projects queries only and attends the target's cache.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <format>
#include <stdexcept>
#include <optional>
#include <type_traits>

export module Dnn.Components.GemmaDraftBlock;

export import Dnn.Components.GemmaConfig;
export import Compute.KvCacheView;

import Dnn.ITensor;
import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TensorOps;
import Dnn.Component;
import Dnn.ComponentType;
import Dnn.CompositeComponent;
import Compute.Device;
import Compute.DeviceAllocation;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.ExecutionContextFactory;
import Compute.CpuMemoryResource;
import Compute.Observation;
#ifdef MILA_HAS_CUDA
import Compute.CudaPinnedMemoryResource;
#endif
import Dnn.Components.RmsNorm;
import Dnn.Components.Rope;
import Dnn.Components.SharedKvAttention;
import Dnn.Components.Residual;
import Dnn.Components.Linear;
import Dnn.Components.GemmaDenseFeedForward;
import Serialization.ModelArchive;
import Serialization.Mode;
import Serialization.Tensor;
import Serialization.SafeTensors;
import Dnn.Quantization.Weight.Policies;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;
    using namespace Mila::Dnn::Quant::Weight;

    /**
     * @brief A draft-model layer; kGlobal selects the full-attention geometry, as in GemmaBlock.
     *
     * HF Gemma4 decoder layer with `is_kv_shared_layer` (Gemma4Mtp.md section 2):
     *   a = o_proj( attend( RoPE( q_norm( q_proj( input_norm(x) ) ) ), target cache ) )
     *   x = x + post_attn_norm(a)
     *   x = ( x + ffn(x) ) * layer_scalar
     *
     * There is no k_proj, v_proj, k_norm or v_norm: keys and values are the target's, as its cache holds them.
     * Decode only, one token per step, at the context's decode position; the config carries the target's
     * key/value head counts and window, which the cache must match.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision, bool kGlobal,
        WeightQuantPolicy TWeightQuantization = NoWeightQuant>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class GemmaDraftBlock : public CompositeComponent<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using CompositeComponentBase = CompositeComponent<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using RmsNormType = RmsNorm<TDeviceType, TPrecision>;
        using RopeType = Rope<TDeviceType, TPrecision>;
        using AttentionType = SharedKvAttention<TDeviceType, TPrecision>;
        using ResidualType = Residual<TDeviceType, TPrecision>;
        using LinearType = Linear<TDeviceType, TPrecision, TWeightQuantization>;
        using FeedForwardType = GemmaDenseFeedForward<TDeviceType, TPrecision, TWeightQuantization>;

        explicit GemmaDraftBlock( const std::string& name, const GemmaConfig& config, std::optional<DeviceId> device_id = std::nullopt )
            : CompositeComponentBase( name ), config_( config )
        {
            config_.validate();

            if ( config_.hasMixtureOfExperts() )
                throw std::invalid_argument( std::format( "GemmaDraftBlock '{}': a draft layer's feed-forward is dense", name ) );

            createGraph();

            if ( device_id.has_value() )
            {
                if ( device_id->type != TDeviceType )
                    throw std::invalid_argument( "GemmaDraftBlock: device type mismatch" );

                owned_exec_context_ = createExecutionContext( device_id.value() );
                this->setExecutionContext( owned_exec_context_.get() );
            }
        }

        ~GemmaDraftBlock() override = default;

        static constexpr bool isGlobal() noexcept { return kGlobal; }

        dim_t headDim() const noexcept { return kGlobal ? config_.getGlobalHeadDim() : config_.getHeadDim(); }
        dim_t numKVHeads() const noexcept { return kGlobal ? config_.getNumGlobalKVHeads() : config_.getNumKVHeads(); }
        dim_t window() const noexcept { return kGlobal ? dim_t{ 0 } : config_.getWindow(); }
        float ropeTheta() const noexcept { return kGlobal ? config_.getRoPEThetaGlobal() : config_.getRoPEThetaLocal(); }
        dim_t rotaryDim() const noexcept { return kGlobal ? config_.getGlobalRotaryDim() : dim_t{ 0 }; }
        dim_t qProjWidth() const noexcept { return config_.getNumHeads() * headDim(); }

        /**
         * @brief One step at the context's decode position `position`, attending `cache` up to the position before it.
         *
         * @param input    [B, 1, model_dim].
         * @param position The context's decode position, at which the query rotates.
         * @param cache    The target layer's cache this layer attends.
         * @return The block's output [B, 1, model_dim], owned by its last component.
         */
        TensorType& decode( const TensorType& input, dim_t position, const KvCacheView& cache )
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "GemmaDraftBlock::decode: must be built before decode()." );

            const int64_t B = input.shape()[ 0 ];
            const dim_t NH = config_.getNumHeads();
            const dim_t HD = headDim();

            auto& normed = input_norm_->forward( input );
            auto& q = q_proj_->forward( normed );

            // QK-norm per head: [B, 1, NH*HD] viewed as [B, NH, HD], one normalization group per head.
            auto q_perhead = q.view( shape_t{ B, NH, HD }, 0 );
            auto& q_normed = q_norm_->forward( q_perhead );
            auto q_roped = q_normed.view( shape_t{ B, 1, NH * HD }, 0 );

            rope_->decodeQuery( q_roped, position );

            auto& attn = attn_->decode( q_roped, cache );

            auto& o = o_proj_->forward( attn );
            auto& o_normed = post_attn_norm_->forward( o );
            auto& res1 = res1_->forward( input, o_normed );

            auto& ffn_out = feed_forward_->forward( res1 );
            auto& res2 = res2_->forward( res1, ffn_out );

            scale( res2, layer_scalar_, res2, this->getExecutionContext() );

            this->publish( ComputePass::Decode, "output", res2 );

            return res2;
        }

        const ComponentType getType() const override
        {
            return ComponentType::Transformer;
        }

        std::vector<ObservableStage> getObservableStages() const override
        {
            return { { "output", ComputePassMask{ ComputePass::Decode } } };
        }

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;

            for ( const auto& child : this->getComponents() )
                stats += child->getMemoryStats();

            return stats;
        }

        /**
         * @brief What onBuilding() would allocate for [B, T, model_dim], without allocating.
         *
         * T bounds the positions RoPE rotates and the attention reads; every other child is built one token wide.
         */
        MemoryStats getRequiredMemory( const BuildContext& context ) const override
        {
            validateBuildContext( context );

            const BlockBuildContexts contexts = resolveBlockBuildContexts( context );
            const std::string n = this->getName();

            MemoryStats stats;
            stats += this->template getComponentAs<RmsNormType>( n + ".input_norm" )->getRequiredMemory( contexts.stream );
            stats += this->template getComponentAs<LinearType>( n + ".q_proj" )->getRequiredMemory( contexts.stream );
            stats += this->template getComponentAs<RmsNormType>( n + ".q_norm" )->getRequiredMemory( contexts.qknorm );
            stats += this->template getComponentAs<RopeType>( n + ".rope" )->getRequiredMemory( contexts.context );
            stats += this->template getComponentAs<AttentionType>( n + ".attn" )->getRequiredMemory( contexts.context );
            stats += this->template getComponentAs<LinearType>( n + ".o_proj" )->getRequiredMemory( contexts.qproj );
            stats += this->template getComponentAs<RmsNormType>( n + ".post_attn_norm" )->getRequiredMemory( contexts.stream );
            stats += this->template getComponentAs<ResidualType>( n + ".res_1" )->getRequiredMemory( contexts.stream );
            stats += this->template getComponentAs<FeedForwardType>( n + ".ffn" )->getRequiredMemory( contexts.stream );
            stats += this->template getComponentAs<ResidualType>( n + ".res_2" )->getRequiredMemory( contexts.stream );

            return stats;
        }

        std::vector<std::string> getParameterNames() const override
        {
            return { "layer_scalar" };
        }

        /// Children through the base recursion, then this block's own layer_scalar, as GemmaBlock does.
        void saveFlatTensors(
            Serialization::SafeTensorsWriter& writer,
            const std::string& prefix,
            Serialization::TensorSavePass pass ) const override
        {
            CompositeComponentBase::saveFlatTensors( writer, prefix, pass );

            Tensor<TensorDataType::FP32, HostStagingMR> scalar(
                TDeviceType == DeviceType::Cuda ? this->getDeviceId() : Device::Cpu(), shape_t{ 1 } );

            scalar.data()[ 0 ] = layer_scalar_;

            this->saveParameterToWriter( writer, prefix + ".layer_scalar", scalar, pass );
        }

        void save_( ModelArchive& archive, SerializationMode mode ) const override
        {
            CompositeComponentBase::save_( archive, mode );

            Tensor<TensorDataType::FP32, HostStagingMR> scalar(
                TDeviceType == DeviceType::Cuda ? this->getDeviceId() : Device::Cpu(), shape_t{ 1 } );

            scalar.data()[ 0 ] = layer_scalar_;

            this->saveParameterToArchive( archive, "layer_scalar", scalar );
        }

        void load_( ModelArchive& archive, SerializationMode mode ) override
        {
            CompositeComponentBase::load_( archive, mode );

            const std::string prefix = "tensors/layer_scalar";

            if ( !archive.hasFile( prefix + "/data.bin" ) )
            {
                throw std::runtime_error( std::format(
                    "GemmaDraftBlock '{}': archive has no blob for 'layer_scalar'", this->getName() ) );
            }

            auto blob = readTensorBlob<HostStagingMR>( archive, prefix, this->getDeviceId().index );

            loadParameter( "layer_scalar", blob );
        }

        /// The block's own parameter is its [1] FP32 layer_scalar; every other name goes to the children.
        void loadParameter( const std::string& name, const ITensorBlob& blob ) override
        {
            if ( name == "layer_scalar" )
            {
                Tensor<TensorDataType::FP32, MR> buf(
                    this->getDeviceId(), shape_t{ 1 }, this->getName() + ".layer_scalar" );
                this->template loadParameterFromBlob<TensorDataType::FP32, MR>(
                    "layer_scalar", blob, buf, shape_t{ 1 } );

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

        struct BlockBuildContexts
        {
            BuildContext stream;
            BuildContext qproj;
            BuildContext qknorm;
            BuildContext context;
        };

        BlockBuildContexts resolveBlockBuildContexts( const BuildContext& context ) const
        {
            const dim_t B = context.inputShape()[ 0 ];
            const dim_t T = context.inputShape()[ 1 ];
            const dim_t NH = config_.getNumHeads();
            const dim_t HD = headDim();

            // One token per step: every per-token child is built one row wide.
            const BuildContext token = context.withPrefillSize( 1 );

            return {
                token.withShape( shape_t{ B, 1, config_.getModelDim() } ),
                token.withShape( shape_t{ B, 1, NH * HD } ),
                token.withShape( shape_t{ B, NH, HD } ),
                token.withShape( shape_t{ B, T, NH * HD } ),
            };
        }

        void onBuilding( const BuildContext& context ) override
        {
            validateBuildContext( context );

            const BlockBuildContexts contexts = resolveBlockBuildContexts( context );
            const std::string n = this->getName();

            input_norm_ = this->template getComponentAs<RmsNormType>( n + ".input_norm" );
            input_norm_->build( contexts.stream );

            q_proj_ = this->template getComponentAs<LinearType>( n + ".q_proj" );
            q_proj_->build( contexts.stream );

            q_norm_ = this->template getComponentAs<RmsNormType>( n + ".q_norm" );
            q_norm_->build( contexts.qknorm );

            rope_ = this->template getComponentAs<RopeType>( n + ".rope" );
            rope_->build( contexts.context );

            attn_ = this->template getComponentAs<AttentionType>( n + ".attn" );
            attn_->build( contexts.context );

            o_proj_ = this->template getComponentAs<LinearType>( n + ".o_proj" );
            o_proj_->build( contexts.qproj );

            post_attn_norm_ = this->template getComponentAs<RmsNormType>( n + ".post_attn_norm" );
            post_attn_norm_->build( contexts.stream );

            res1_ = this->template getComponentAs<ResidualType>( n + ".res_1" );
            res1_->build( contexts.stream );

            feed_forward_ = this->template getComponentAs<FeedForwardType>( n + ".ffn" );
            feed_forward_->build( contexts.stream );

            res2_ = this->template getComponentAs<ResidualType>( n + ".res_2" );
            res2_->build( contexts.stream );
        }

        void onTrainingModeChanging( TrainingMode training_mode ) override
        {
            for ( const auto& child : this->getComponents() )
                child->setTrainingMode( training_mode );
        }

    private:
        GemmaConfig config_;
        std::unique_ptr<IExecutionContext> owned_exec_context_{ nullptr };

#ifdef MILA_HAS_CUDA
        using HostStagingMR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaPinnedMemoryResource, CpuMemoryResource>;
#else
        using HostStagingMR = CpuMemoryResource;
#endif

        std::shared_ptr<RmsNormType> input_norm_{ nullptr };
        std::shared_ptr<LinearType> q_proj_{ nullptr };
        std::shared_ptr<RmsNormType> q_norm_{ nullptr };
        std::shared_ptr<RopeType> rope_{ nullptr };
        std::shared_ptr<AttentionType> attn_{ nullptr };
        std::shared_ptr<LinearType> o_proj_{ nullptr };
        std::shared_ptr<RmsNormType> post_attn_norm_{ nullptr };
        std::shared_ptr<ResidualType> res1_{ nullptr };
        std::shared_ptr<FeedForwardType> feed_forward_{ nullptr };
        std::shared_ptr<ResidualType> res2_{ nullptr };

        // Per-layer learned output scale; 1.0 until loaded.
        float layer_scalar_{ 1.0f };

        void createGraph()
        {
            const std::string n = this->getName();
            const dim_t model_dim = config_.getModelDim();
            const float eps = config_.getRMSNormEpsilon();
            const dim_t HD = headDim();

            // Gemma 4 norms multiply by the raw stored weight (unit offset 0), as in GemmaBlock.
            auto rms = [&]( const shape_t& shape )
            {
                return RmsNormConfig( shape ).withEpsilon( eps ).withBias( false );
            };

            this->addComponent( std::make_shared<RmsNormType>( n + ".input_norm", rms( shape_t{ model_dim } ) ) );
            this->addComponent( std::make_shared<LinearType>(
                n + ".q_proj", LinearConfig( model_dim, qProjWidth() ).withBias( false ) ) );
            this->addComponent( std::make_shared<RmsNormType>( n + ".q_norm", rms( shape_t{ HD } ) ) );

            // Queries only: zero key/value heads, since the keys were rotated by the target layer that cached them.
            auto rope_cfg = RopeConfig( qProjWidth(), config_.getNumHeads(), 0, config_.getMaxSequenceLength() )
                .withBase( ropeTheta() )
                .withRotaryDim( static_cast<size_t>( rotaryDim() ) );
            this->addComponent( std::make_shared<RopeType>( n + ".rope", rope_cfg ) );

            auto attention_cfg = GqaConfig( qProjWidth(), config_.getNumHeads(), numKVHeads() )
                .withWindow( window() )
                .withAttentionScale( 1.0f );
            this->addComponent( std::make_shared<AttentionType>( n + ".attn", attention_cfg ) );

            this->addComponent( std::make_shared<LinearType>(
                n + ".o_proj", LinearConfig( qProjWidth(), model_dim ).withBias( false ) ) );
            this->addComponent( std::make_shared<RmsNormType>( n + ".post_attn_norm", rms( shape_t{ model_dim } ) ) );
            this->addComponent( std::make_shared<ResidualType>( n + ".res_1", ResidualConfig{} ) );
            this->addComponent( std::make_shared<FeedForwardType>( n + ".ffn", config_ ) );
            this->addComponent( std::make_shared<ResidualType>( n + ".res_2", ResidualConfig{} ) );
        }

        void validateBuildContext( const BuildContext& context ) const
        {
            const auto& s = context.inputShape();

            if ( s.size() != 3 || s.back() != config_.getModelDim() )
                throw std::invalid_argument( std::format(
                    "GemmaDraftBlock: build shape must be [B, T, {}]", config_.getModelDim() ) );
        }
    };
}
