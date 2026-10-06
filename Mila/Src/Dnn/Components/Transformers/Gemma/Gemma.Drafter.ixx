/**
 * @file Gemma.Drafter.ixx
 * @brief Gemma 4's draft model: four query-only Gemma layers over the target's caches, proposing tokens to verify.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <string_view>
#include <format>
#include <stdexcept>
#include <optional>
#include <utility>

export module Dnn.Components.GemmaDrafter;

export import Dnn.Components.GemmaConfig;
export import Dnn.Components.GemmaDraftBlock;
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
import Compute.DeviceAllocation;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.ExecutionContext;
import Compute.IExecutionContext;
import Compute.CpuMemoryResource;
import Compute.Observation;
#ifdef MILA_HAS_CUDA
import Compute.CudaPinnedMemoryResource;
#endif
import Dnn.Components.RmsNorm;
import Dnn.Components.Linear;
import Serialization.ModelArchive;
import Serialization.Mode;
import Serialization.WeightsReader;
import Serialization.Metadata;
import Dnn.Quantization.Weight.Policies;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;
    using namespace Mila::Dnn::Quant::Weight;

    /**
     * @brief The draft model Google ships beside each Gemma 4 model (Gemma4Mtp.md section 2).
     *
     * One step, at the context's decode position p:
     *   x = pre_projection( [ target embedding of the last token | target final-normed hidden ] )
     *   x = layers( x ), each attending the target's cache of its layer type up to p - 1
     *   h = final_norm( x );  logits = head( h );  next hidden = post_projection( h )
     *
     * It runs on the target's execution context (Gemma4Mtp.md 4.1), keeps no cache, and decodes one token at a
     * time for batch 1. Its sliding layers precede one global layer, as every Gemma 4 drafter's do. Its weights are
     * a file of their own, every tensor named under the component's name.
     *
     * @tparam THeadQuantization The head's format, apart from the layers': its vocabulary rows are most of a step's
     *                           bytes. A BF16 head loaded into a quantized one is quantized on load.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision, WeightQuantPolicy TWeightQuantization = NoWeightQuant,
        WeightQuantPolicy THeadQuantization = NoWeightQuant>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class GemmaDrafter : public CompositeComponent<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using CompositeComponentBase = CompositeComponent<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using RmsNormType = RmsNorm<TDeviceType, TPrecision>;
        using LinearType = Linear<TDeviceType, TPrecision, TWeightQuantization>;
        using HeadType = Linear<TDeviceType, TPrecision, THeadQuantization>;
        using LocalBlockType = GemmaDraftBlock<TDeviceType, TPrecision, false, TWeightQuantization>;
        using GlobalBlockType = GemmaDraftBlock<TDeviceType, TPrecision, true, TWeightQuantization>;

        /**
         * @param name              The component's name, `drafter` where its file's tensor names resolve.
         * @param config            The drafter's geometry, as configFromMetadata() builds it.
         * @param target_model_dim  The target's residual width, which the projections in and out join.
         * @param target_context    The target's execution context, which the drafter runs on (Gemma4Mtp.md 4.1);
         *                          null when the target adds the drafter as a child and supplies it then.
         */
        GemmaDrafter( const std::string& name, const GemmaConfig& config, dim_t target_model_dim,
            IExecutionContext* target_context = nullptr )
            : CompositeComponentBase( name ), config_( config ), target_model_dim_( target_model_dim )
        {
            config_.validate();

            if ( config_.getNumLayers() < 1 || config_.getSlidingWindowPattern() != config_.getNumLayers() )
            {
                throw std::invalid_argument( std::format(
                    "GemmaDrafter '{}': expected sliding layers ending in one global layer (pattern == layers), got {} "
                    "layers in a pattern of {}", name, config_.getNumLayers(), config_.getSlidingWindowPattern() ) );
            }

            createGraph();

            if ( target_context != nullptr )
                this->setExecutionContext( target_context );
        }

        /**
         * @brief The drafter's geometry: its own from `drafter`, the target's RoPE and key/value heads from `target`.
         *
         * @throws std::invalid_argument When a head size or the window differs from the target's: the drafter's
         *         queries attend the target's cache, so they must be the cache's head size and band.
         */
        static GemmaConfig configFromMetadata( const WeightsMetadata& drafter, const GemmaConfig& target )
        {
            if ( drafter.head_dim != target.getHeadDim() || drafter.global_head_dim != target.getGlobalHeadDim()
                 || drafter.window != target.getWindow() )
            {
                throw std::invalid_argument( std::format(
                    "GemmaDrafter: the drafter's head sizes {} / {} and window {} must be the target's {} / {} and {}",
                    drafter.head_dim, drafter.global_head_dim, drafter.window,
                    target.getHeadDim(), target.getGlobalHeadDim(), target.getWindow() ) );
            }

            GemmaConfig config( static_cast<dim_t>( drafter.embedding_dim ), static_cast<dim_t>( drafter.num_layers ) );

            config.withVocabularyLength( static_cast<dim_t>( drafter.vocab_size ) )
                .withMaxSequenceLength( target.getMaxSequenceLength() )
                .withNumHeads( static_cast<dim_t>( drafter.num_heads ) )
                .withNumKVHeads( target.getNumKVHeads() )
                .withHeadDim( static_cast<dim_t>( drafter.head_dim ) )
                .withGlobalHeadDim( static_cast<dim_t>( drafter.global_head_dim ) )
                .withNumGlobalKVHeads( target.getNumGlobalKVHeads() )
                .withHiddenDimension( static_cast<dim_t>( drafter.hidden_dim ) )
                .withRMSNormEpsilon( drafter.norm_epsilon )
                .withWindow( static_cast<dim_t>( drafter.window ) )
                .withSlidingWindowPattern( static_cast<dim_t>( drafter.sliding_window_pattern ) )
                .withGlobalRotaryDim( target.getGlobalRotaryDim() )
                .withRoPETheta( target.getRoPEThetaLocal() )
                .withGlobalRoPETheta( target.getRoPEThetaGlobal() )
                .withTieWordEmbeddings( true );

            return config;
        }

        /**
         * @brief Start a draft from the target's final-normed hidden state at the position before the first step's.
         *
         * The next decode() reads it, and each decode() leaves its own next hidden state where the step after it
         * reads, so a draft's steps take no hidden state of their own and read it from one place.
         *
         * @param hidden [1, 1, target_model_dim].
         */
        void startFrom( const TensorType& hidden )
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "GemmaDrafter::startFrom: must be built before startFrom()." );

            auto hidden_half = input_->view( shape_t{ 1, 1, target_model_dim_ }, target_model_dim_ );

            copy( hidden, hidden_half, this->getExecutionContext() );
        }

        /**
         * @brief One draft step at the context's decode position `position`, from the hidden state startFrom() or
         *        the step before left.
         *
         * @param embedding  The target's token embedding of the last token, [1, 1, target_model_dim], scaled as the
         *                   target scales it.
         * @param position   The context's decode position: where the target's chosen token sits, unprocessed.
         * @param sliding    The target's last sliding layer's cache.
         * @param global     The target's last global layer's cache.
         * @return Logits [1, 1, vocab].
         */
        TensorType& decode( const TensorType& embedding, dim_t position, const KvCacheView& sliding,
            const KvCacheView& global )
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "GemmaDrafter::decode: must be built before decode()." );

            auto embedding_half = input_->view( shape_t{ 1, 1, target_model_dim_ }, 0 );
            auto hidden_half = input_->view( shape_t{ 1, 1, target_model_dim_ }, target_model_dim_ );

            copy( embedding, embedding_half, this->getExecutionContext() );

            TensorType* x = &pre_projection_->forward( *input_ );

            for ( auto& block : local_blocks_ )
                x = &block->decode( *x, position, sliding );

            x = &global_block_->decode( *x, position, global );

            auto& normed = final_norm_->forward( *x );

            next_hidden_ = &post_projection_->forward( normed );
            copy( *next_hidden_, hidden_half, this->getExecutionContext() );

            return head_->forward( normed );
        }

        /// The hidden state the last decode() produced, which the next decode() reads, [1, 1, target_model_dim].
        const TensorType& nextHidden() const
        {
            if ( next_hidden_ == nullptr )
                throw std::logic_error( "GemmaDrafter::nextHidden: no step has run" );

            return *next_hidden_;
        }

        /// The name every tensor in a drafter's weights file is under (Tools/Converters/Gemma/convert_drafter.py).
        static constexpr std::string_view kWeightsPrefix = "drafter.";

        /**
         * @brief Load the drafter's own weights file into this component, whatever this component is named: a
         *        network names its drafter under its own name, the file names every tensor under `drafter`.
         *
         * @param reader The drafter's file, as Tools/Converters/Gemma/convert_drafter.py writes it.
         */
        void loadParameters( WeightsReader& reader )
        {
            const int device_index = this->getExecutionContext()->getDeviceId().index;

            auto consume = [&]( const std::string& full_name, const Serialization::ITensorBlob& blob )
            {
                auto [component_path, param_name] = parseParameterPath( full_name );

                if ( !component_path.starts_with( kWeightsPrefix ) )
                {
                    throw std::runtime_error( std::format(
                        "GemmaDrafter '{}': tensor '{}' is not under '{}'; not a drafter's weights file",
                        this->getName(), full_name, kWeightsPrefix ) );
                }

                this->findComponent( component_path.substr( kWeightsPrefix.size() ) )->loadParameter( param_name, blob );

                // The reader reuses its pinned staging slot when this returns.
                if constexpr ( TDeviceType == DeviceType::Cuda )
                {
                    this->getExecutionContext()->synchronize();
                }
            };

#ifdef MILA_HAS_CUDA
            if constexpr ( TDeviceType == DeviceType::Cuda )
            {
                reader.streamTensorBlobs<CudaPinnedMemoryResource>( consume, device_index );
            }
            else
#endif
            {
                reader.streamTensorBlobs<CpuMemoryResource>( consume );
            }

            this->getExecutionContext()->synchronize();
            this->getExecutionContext()->releaseLoadStaging();
        }

        const ComponentType getType() const override
        {
            return ComponentType::Transformer;
        }

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;

            for ( const auto& child : this->getComponents() )
                stats += child->getMemoryStats();

            if ( input_ != nullptr )
                stats.device_state_bytes += occupiedTensorBytes( *input_ );

            return stats;
        }

        /// What onBuilding() would allocate for [1, T, model_dim], T the longest context a step attends.
        MemoryStats getRequiredMemory( const BuildContext& context ) const override
        {
            validateBuildContext( context );

            const Contexts contexts = resolveContexts( context );
            const std::string n = this->getName();

            MemoryStats stats;
            stats += this->template getComponentAs<LinearType>( n + ".pre_projection" )->getRequiredMemory( contexts.joined );

            for ( dim_t i = 0; i + 1 < config_.getNumLayers(); ++i )
                stats += this->template getComponentAs<LocalBlockType>( layerName( i ) )->getRequiredMemory( context );

            stats += this->template getComponentAs<GlobalBlockType>( layerName( config_.getNumLayers() - 1 ) )
                ->getRequiredMemory( context );
            stats += this->template getComponentAs<RmsNormType>( n + ".final_norm" )->getRequiredMemory( contexts.token );
            stats += this->template getComponentAs<HeadType>( n + ".head" )->getRequiredMemory( contexts.token );
            stats += this->template getComponentAs<LinearType>( n + ".post_projection" )->getRequiredMemory( contexts.token );

            stats.device_state_bytes += occupiedDeviceBytes(
                storageBytes<TPrecision>( 2 * target_model_dim_ ), context.getAllocationGranularity() );

            return stats;
        }

        std::string toString() const override
        {
            return std::format( "GemmaDrafter '{}': {} layers, width {}, joins target width {}\n",
                this->getName(), config_.getNumLayers(), config_.getModelDim(), target_model_dim_ );
        }

        const GemmaConfig& getConfig() const noexcept
        {
            return config_;
        }

    protected:

        struct Contexts
        {
            BuildContext joined;
            BuildContext token;
        };

        Contexts resolveContexts( const BuildContext& context ) const
        {
            const BuildContext token = context.withPrefillSize( 1 );

            return {
                token.withShape( shape_t{ 1, 1, 2 * target_model_dim_ } ),
                token.withShape( shape_t{ 1, 1, config_.getModelDim() } ),
            };
        }

        void onBuilding( const BuildContext& context ) override
        {
            validateBuildContext( context );

            const Contexts contexts = resolveContexts( context );
            const std::string n = this->getName();

            pre_projection_ = this->template getComponentAs<LinearType>( n + ".pre_projection" );
            pre_projection_->build( contexts.joined );

            local_blocks_.clear();

            for ( dim_t i = 0; i + 1 < config_.getNumLayers(); ++i )
            {
                auto block = this->template getComponentAs<LocalBlockType>( layerName( i ) );
                block->build( context );
                local_blocks_.push_back( block );
            }

            global_block_ = this->template getComponentAs<GlobalBlockType>( layerName( config_.getNumLayers() - 1 ) );
            global_block_->build( context );

            final_norm_ = this->template getComponentAs<RmsNormType>( n + ".final_norm" );
            final_norm_->build( contexts.token );

            head_ = this->template getComponentAs<HeadType>( n + ".head" );
            head_->build( contexts.token );

            post_projection_ = this->template getComponentAs<LinearType>( n + ".post_projection" );
            post_projection_->build( contexts.token );

            input_ = std::make_unique<TensorType>( this->getExecutionContext()->getDeviceId(),
                shape_t{ 1, 1, 2 * target_model_dim_ }, n + ".input" );

            next_hidden_ = nullptr;
        }

        void onTrainingModeChanging( TrainingMode training_mode ) override
        {
            for ( const auto& child : this->getComponents() )
                child->setTrainingMode( training_mode );
        }

    private:
        GemmaConfig config_;
        dim_t target_model_dim_{ 0 };

        std::shared_ptr<LinearType> pre_projection_{ nullptr };
        std::vector<std::shared_ptr<LocalBlockType>> local_blocks_;
        std::shared_ptr<GlobalBlockType> global_block_{ nullptr };
        std::shared_ptr<RmsNormType> final_norm_{ nullptr };
        std::shared_ptr<HeadType> head_{ nullptr };
        std::shared_ptr<LinearType> post_projection_{ nullptr };

        // [1, 1, 2 * target_model_dim]: the embedding then the hidden state, one projection's input as HF joins them.
        std::unique_ptr<TensorType> input_{ nullptr };
        const TensorType* next_hidden_{ nullptr };

        std::string layerName( dim_t index ) const
        {
            return this->getName() + ".layer_" + std::to_string( index );
        }

        /// "drafter.layer_0.q_proj.weight" into the component path and the parameter, split at the last dot.
        static std::pair<std::string, std::string> parseParameterPath( const std::string& full_name )
        {
            const auto last_dot = full_name.rfind( '.' );

            if ( last_dot == std::string::npos )
                throw std::runtime_error( std::format( "GemmaDrafter: invalid parameter path '{}'", full_name ) );

            return { full_name.substr( 0, last_dot ), full_name.substr( last_dot + 1 ) };
        }

        void createGraph()
        {
            const std::string n = this->getName();
            const dim_t model_dim = config_.getModelDim();

            this->addComponent( std::make_shared<LinearType>(
                n + ".pre_projection", LinearConfig( 2 * target_model_dim_, model_dim ).withBias( false ) ) );

            for ( dim_t i = 0; i + 1 < config_.getNumLayers(); ++i )
                this->addComponent( std::make_shared<LocalBlockType>( layerName( i ), config_ ) );

            this->addComponent( std::make_shared<GlobalBlockType>( layerName( config_.getNumLayers() - 1 ), config_ ) );

            this->addComponent( std::make_shared<RmsNormType>( n + ".final_norm",
                RmsNormConfig( shape_t{ model_dim } ).withEpsilon( config_.getRMSNormEpsilon() ).withBias( false ) ) );

            this->addComponent( std::make_shared<HeadType>(
                n + ".head", LinearConfig( model_dim, config_.getVocabSize() ).withBias( false ) ) );

            this->addComponent( std::make_shared<LinearType>(
                n + ".post_projection", LinearConfig( model_dim, target_model_dim_ ).withBias( false ) ) );
        }

        void validateBuildContext( const BuildContext& context ) const
        {
            const auto& s = context.inputShape();

            if ( s.size() != 3 || s[ 0 ] != 1 || s.back() != config_.getModelDim() )
                throw std::invalid_argument( std::format(
                    "GemmaDrafter: build shape must be [1, T, {}]; the drafter decodes batch 1", config_.getModelDim() ) );
        }
    };
}
