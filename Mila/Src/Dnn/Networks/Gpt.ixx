/**
 * @file Gpt.ixx
 * @brief GPT-2 style transformer network (decoder-only) for autoregressive language modeling.
 *
 * Device-templated network implementing a GPT-2 style transformer decoder.
 */

module;
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <type_traits>
#include <stdexcept>
#include <cstdint>
#include <ostream>
#include <iostream>
#include <format>
#include <optional>
#include <cassert>

export module Dnn.Networks.Gpt;
export import :Config;
export import :Presets;

import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.Network;
import Dnn.Components.Linear;
import Dnn.Components.LayerNorm;
import Dnn.Components.LearnedEncoder;
import Dnn.Components.GptBlock;
import Dnn.Component;
import Dnn.ComponentType;
import Dnn.ActivationType;
import Compute.Precision;
import Compute.DeviceType;
import Compute.DeviceId;
import Compute.DeviceTypeTraits;
import Compute.ExecutionContext;
import Compute.ExecutionContextFactory;
import Serialization.ModelArchive;

namespace Mila::Dnn::Networks
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief GPT-2 style transformer (decoder-only) for autoregressive token prediction.
     *
     * Template parameters:
     *  - TDeviceType: device type (Cpu/Cuda)
     *  - TPrecision: tensor precision
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Gpt : public Network<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using NetworkBase = Network<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using LinearType = Linear<TDeviceType, TPrecision>;
        using LayerNormType = LayerNorm<TDeviceType, TPrecision>;
        using TransformerBlockType = GptBlock<TDeviceType, TPrecision>;
        using EncoderType = LearnedEncoder<TDeviceType, dtype_t::INT32, TPrecision>;
        using TokenIndexType = Tensor<dtype_t::INT32, MR>;
        using ComponentPtr = typename NetworkBase::ComponentPtr;

        /**
         * @brief Construct Gpt type transformer.
         *
         * @param name Network name
         * @param config GPT transformer configuration
         * @param device_id Device identifier for execution
         *
         * @throws std::invalid_argument on invalid config or device mismatch
         */
        explicit Gpt( const std::string& name, const GptConfig& config, DeviceId device_id )
            : NetworkBase( name ), owned_context_( createExecutionContext( device_id ) ), config_( config )
        {
            config_.validate();

            if ( device_id.type != TDeviceType )
            {
                throw std::invalid_argument(
                    std::format( "GptTransformer: device type mismatch: expected {}, got {}",
                        deviceTypeToString( TDeviceType ),
                        deviceTypeToString( device_id.type ) ) );
            }

            createGraph();

            this->setExecutionContext( owned_context_.get() );
        }

        ~Gpt() override = default;

        // --------------------------------------------------------------------
        // Factory methods
        // --------------------------------------------------------------------

        static std::unique_ptr<Gpt<TDeviceType, TPrecision>> fromPretrained(
            const std::filesystem::path& model_path,
            std::size_t batch_size,      // User specifies runtime dimensions
            std::size_t seq_length,      // Must be ? max_seq_length from weights
            DeviceId device_id = DeviceId{ TDeviceType, 0 } )
        {
            ModelReader reader( model_path );
            const auto& metadata = reader.getMetadata();

            GptConfig config = createConfigFromMetadata( metadata );

            std::unique_ptr<Gpt<TDeviceType, TPrecision>> gpt =
                std::make_unique<Gpt<TDeviceType, TPrecision>>(
                    metadata.model_name,
                    config,
                    device_id
                );

            // Build with max sequence length (position embeddings support full range)
            shape_t build_shape = { 1, config.getMaxSequenceLength() };
            gpt->build( build_shape );

            return gpt;
        }

        /**
         * @brief Load GptTransformer from archive.
         *
         * Reads metadata, constructs network, builds with saved shape and loads weights.
         */
        /*static std::unique_ptr<GptTransformer> Load( ModelArchive& archive, DeviceId device_id )
        {
            auto scope = ModelArchive::ScopedScope( archive, "network" );

            SerializationMetadata meta = archive.readMetadata( "transformer_meta.json" );
          
            return transformer;
        }*/

        // ====================================================================
        // Compute API (component-owned outputs)
        // ====================================================================

        TensorType& forward( const TokenIndexType& input )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "GptTransformer must be built before calling forward." );
            }

            encoder_out_ptr_ = &encoder_->forward( input );
            this->getExecutionContext()->synchronize();

            if ( block_input_ptrs_.empty() || block_input_ptrs_.size() != transformer_blocks_.size() )
            {
                throw std::runtime_error( "GptTransformer: forward internal state not initialized" );
            }

            block_input_ptrs_[ 0 ] = encoder_out_ptr_;

            for ( size_t i = 0; i < transformer_blocks_.size(); ++i )
            {
                auto& block_out = transformer_blocks_[ i ]->forward( *block_input_ptrs_[ i ] );
                this->getExecutionContext()->synchronize();

                block_output_ptrs_[ i ] = &block_out;

                if ( i + 1 < transformer_blocks_.size() )
                {
                    block_input_ptrs_[ i + 1 ] = &block_out;
                }
            }

            normalized_ptr_ = &final_layernorm_->forward( *block_output_ptrs_.back() );
            this->getExecutionContext()->synchronize();

            logits_ptr_ = &lm_head_->forward( *normalized_ptr_ );
            this->getExecutionContext()->synchronize();

            copy( *dynamic_cast<const TensorType*>(logits_ptr_), *owned_output_ );
            this->getExecutionContext()->synchronize();

            return *owned_output_;
        }

        TokenIndexType& backward( const TokenIndexType& input, const TensorType& output_grad )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "GptTransformer must be built before calling backward." );
            }

            if ( !this->isTraining() )
            {
                throw std::runtime_error( "GptTransformer: backward requires training mode (setTraining(true))." );
            }

            if ( !encoder_out_ptr_ )
            {
                throw std::runtime_error( "GptTransformer: forward activations not present for backward. Call forward() before backward()." );
            }

            for ( size_t i = 0; i < transformer_blocks_.size(); ++i )
            {
                if ( !block_input_ptrs_[ i ] || !block_output_ptrs_[ i ] )
                {
                    throw std::runtime_error( std::format( "GptTransformer: missing cached activation for block {}", i ) );
                }
            }

            if ( !normalized_ptr_ || !logits_ptr_ )
            {
                throw std::runtime_error( "GptTransformer: missing final activations for backward" );
            }

            auto& normalized_grad_ptr = lm_head_->backward( *normalized_ptr_, output_grad );
            this->getExecutionContext()->synchronize();

            auto& last_block_grad_ptr = final_layernorm_->backward( *block_output_ptrs_.back(), normalized_grad_ptr );
            this->getExecutionContext()->synchronize();

            TensorType* curr_grad = &last_block_grad_ptr;

            for ( int64_t i = static_cast<int64_t>(transformer_blocks_.size()) - 1; i >= 0; --i )
            {
                auto& block_grad_ptr = transformer_blocks_[ static_cast<size_t>(i) ]->backward(
                    *block_input_ptrs_[ static_cast<size_t>(i) ], *curr_grad );

                curr_grad = &block_grad_ptr;

                this->getExecutionContext()->synchronize();
            }

            auto& input_grad_ptr = encoder_->backward( input, *curr_grad );
            this->getExecutionContext()->synchronize();

            return input_grad_ptr;
        }

        void zeroGradients() override
        {
            if ( !this->isBuilt() )
            {
                return;
            }

            encoder_->zeroGradients();

            for ( auto& block : transformer_blocks_ )
            {
                block->zeroGradients();
            }

            final_layernorm_->zeroGradients();
            lm_head_->zeroGradients();
        }

        const ComponentType getType() const override
        {
            return ComponentType::Gpt2;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << std::endl;
            oss << "Gpt Network: " << this->getName() << std::endl;
            oss << "Device: " << this->getDeviceId().toString() << std::endl;

            oss << "Architecture:" << std::endl;
            oss << "  Vocabulary: " << config_.getVocabSize() << " tokens" << std::endl;
            oss << "  Max sequence length: " << config_.getMaxSequenceLength() << std::endl;
            oss << "  Embedding Size/Dim: " << config_.getEmbeddingSize() << std::endl;
            oss << "  Number of heads: " << config_.getNumHeads() << std::endl;
            oss << "  Number of layers: " << config_.getNumLayers() << std::endl;
            oss << "  MLP Hidden Size/Dim: " << config_.getHiddenSize() << std::endl;

            if ( this->isBuilt() )
            {
                oss << "  Parameters: " << this->parameterCount() << std::endl;
                oss << "  Batch size: " << batch_size_ << std::endl;
                oss << "  Sequence length: " << seq_length_ << std::endl;

                oss << "  Input shape: ("; for ( size_t i = 0; i < input_shape_.size(); ++i ) {
                    oss << input_shape_[ i ]; if ( i != input_shape_.size() - 1 ) oss << ", ";
                } oss << ")" << std::endl;

                oss << "  Output shape: ("; for ( size_t i = 0; i < output_shape_.size(); ++i ) {
                    oss << output_shape_[ i ]; if ( i != output_shape_.size() - 1 ) oss << ", ";
                } oss << ")" << std::endl;
            }

            oss << "  Sub-Modules:" << std::endl;

            if ( encoder_ )
            {
                oss << "    - encoder: " << encoder_->getName() << std::endl;
            }

            oss << "    - transformer_blocks: " << transformer_blocks_.size() << " layers" << std::endl;

            if ( final_layernorm_ )
            {
                oss << "    - final_layernorm: " << final_layernorm_->getName() << std::endl;
            }

            if ( lm_head_ )
            {
                oss << "    - lm_head: " << lm_head_->getName() << std::endl;
            }

            oss << std::endl;

            return oss.str();
        }

        IExecutionContext* getExecutionContext() const
        {
            return NetworkBase::getExecutionContext();
        }

        /**
         * @brief Initialize this transformer's components from a GPT-2 checkpoint.
         *
         * Delegates to small helpers that load checkpoint blobs and apply them to
         * the encoder, per-layer blocks, and final layer-norm.
         */
        /*void initializeFromCheckpoint( const std::string& checkpoint_path )
        {
            static_assert(std::is_same_v<typename TensorDataTypeTraits<TPrecision>::value_type, float>,
                "initializeFromCheckpoint requires TPrecision == dtype_t::FP32.");

            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "GptTransformer: must be built before initializing from checkpoint." );
            }

            ModelConfig ckpt_cfg{};
            ParameterTensors params{};
            std::array<size_t, NumberOfParameterTensors> param_sizes{};

            readCheckpoint( checkpoint_path, ckpt_cfg, params, param_sizes );

            validateCheckpointCompatibility( ckpt_cfg );

            applyEncoderParams( params );

            for ( size_t layer = 0; layer < static_cast<size_t>( config_.num_layers ); ++layer )
            {
                applyLayerParams( layer, params );
            }

            applyFinalLayerNorm( params );
        }*/

    protected:
        void save_( ModelArchive& archive, SerializationMode /*mode*/ ) const override
        {
            SerializationMetadata meta;
            meta.set( "type", "GptTransformer" )
                .set( "version", int64_t( 1 ) )
                .set( "name", this->getName() )
                .set( "vocab_size", config_.getVocabSize() )
                .set( "max_seq_length", config_.getMaxSequenceLength() )
                .set( "embedding_dim", config_.getEmbeddingSize() )
                .set( "num_heads", config_.getNumHeads() )
                .set( "num_layers", config_.getNumLayers()  )
                .set( "mlp_hidden_dim", config_.getHiddenSize() );

            if ( this->isBuilt() )
            {
                meta.set( "input_shape", input_shape_ )
                    .set( "embedding_shape", embedding_shape_ )
                    .set( "output_shape", output_shape_ )
                    .set( "batch_size", batch_size_ )
                    .set( "seq_length", seq_length_ );
            }

            archive.writeMetadata( "transformer_meta.json", meta );
        }

        void onTrainingChanging( bool is_training ) override
        {
            NetworkBase::onTrainingChanging( is_training );
        }

        void onBuilding( const shape_t& input_shape ) override
        {
            validateInputShape( input_shape );

            input_shape_ = input_shape;
            batch_size_ = input_shape[ 0 ];
            seq_length_ = input_shape[ 1 ];

            embedding_shape_ = { batch_size_, seq_length_, config_.getEmbeddingSize() };
            output_shape_ = { batch_size_, seq_length_, config_.getVocabSize() };

            encoder_ = this->template getComponentAs<EncoderType>( this->getName() + ".lenc" );
            encoder_->build( input_shape );

            for ( int64_t i = 0; i < config_.getNumLayers(); ++i )
            {
                std::string block_name = this->getName() + ".tf_layer_" + std::to_string( i );
                auto block = this->template getComponentAs<TransformerBlockType>( block_name );
                block->build( embedding_shape_ );
                transformer_blocks_.push_back( block );
            }

            final_layernorm_ = this->template getComponentAs<LayerNormType>( this->getName() + ".ln_final" );
            final_layernorm_->build( embedding_shape_ );

            lm_head_ = this->template getComponentAs<LinearType>( this->getName() + ".lm_head" );
            lm_head_->build( embedding_shape_ );

            auto device_id = this->getDeviceId();

            owned_output_ = std::make_shared<TensorType>( device_id, output_shape_ );
            owned_output_->setName( this->getName() + ".output" );

            block_input_ptrs_.assign( transformer_blocks_.size(), nullptr );
            block_output_ptrs_.assign( transformer_blocks_.size(), nullptr );

            encoder_out_ptr_ = nullptr;
            normalized_ptr_ = nullptr;
            logits_ptr_ = nullptr;
        }

    private:
        std::unique_ptr<IExecutionContext> owned_context_{ nullptr };

        GptConfig config_;

        shape_t input_shape_;
        shape_t embedding_shape_;
        shape_t output_shape_;
        int64_t batch_size_{ 0 };
        int64_t seq_length_{ 0 };

        std::shared_ptr<EncoderType> encoder_{ nullptr };
        std::vector<std::shared_ptr<TransformerBlockType>> transformer_blocks_;
        std::shared_ptr<LayerNormType> final_layernorm_{ nullptr };
        std::shared_ptr<LinearType> lm_head_{ nullptr };

        std::shared_ptr<TensorType> owned_output_{ nullptr };

        TensorType* encoder_out_ptr_{ nullptr };
        std::vector<TensorType*> block_input_ptrs_;
        std::vector<TensorType*> block_output_ptrs_;

        TensorType* normalized_ptr_{ nullptr };
        TensorType* logits_ptr_{ nullptr };

        void validateInputShape( const shape_t& input_shape ) const
        {
            if ( input_shape.size() != 2 )
            {
                throw std::invalid_argument(
                    "GptTransformer: input must have rank 2 (batch_size, seq_length)" );
            }

            if ( input_shape[ 1 ] > config_.getMaxSequenceLength() )
            {
                throw std::invalid_argument(
                    std::format( "GptTransformer: sequence length {} exceeds maximum {}",
                        input_shape[ 1 ], config_.getMaxSequenceLength() ));
            }
        }

        void createGraph()
        {
            LearnedEncoderConfig enc_cfg;
            enc_cfg.withVocabularyLength( static_cast<size_t>(config_.getVocabSize() ))
                .withMaxSequenceLength( static_cast<size_t>(config_.getMaxSequenceLength() ))
                .withChannels( static_cast<size_t>(config_.getEmbeddingSize()));

            enc_cfg.validate();

            auto encoder = std::make_shared<EncoderType>(
                this->getName() + ".lenc", enc_cfg );

            this->addComponent( encoder );

            for ( int64_t i = 0; i < config_.getNumLayers(); ++i )
            {
                GptBlockConfig block_cfg( static_cast<dim_t>( config_.getEmbeddingSize() ),
                    static_cast<dim_t>( config_.getNumHeads() ));

                block_cfg.withHiddenSize( static_cast<dim_t>( config_.getHiddenSize() ))
                    .withBias( false )
                    .withActivation( ActivationType::Gelu )
                    .withResidualScale( 1.0f / sqrtf( static_cast<float>( config_.getNumLayers() ) ) );

                auto layer = std::make_shared<TransformerBlockType>(
                    this->getName() + ".tf_layer_" + std::to_string( i ), block_cfg, std::nullopt );

                this->addComponent( layer );
            }

            auto ln_config = LayerNormConfig()
                .withNormalizedShape( { config_.getEmbeddingSize() });

            auto final_layernorm = std::make_shared<LayerNormType>(
                this->getName() + ".ln_final", ln_config, std::nullopt );

            this->addComponent( final_layernorm );

            auto lm_head_config = LinearConfig( config_.getEmbeddingSize(), config_.getVocabSize() )
                .withBias( false );

            auto lm_head = std::make_shared<LinearType>(
                this->getName() + ".lm_head", lm_head_config, std::nullopt );

            this->addComponent( lm_head );
        }

        /**
         * @brief Create GptConfig from Mila metadata.
         */
        static auto createConfigFromMetadata( const ModelMetadata& metadata ) -> GptConfig
        {
            dim_t embedding_size = static_cast<dim_t>(metadata.embedding_dim);
            dim_t num_layers = static_cast<dim_t>(metadata.num_layers);

            GptConfig config = GptConfig( embedding_size, num_layers );

            config.withVocabSize( static_cast<dim_t>( metadata.vocab_size ) )
                .withMaxSequenceLength( static_cast<dim_t>( metadata.max_seq_length ) )
                .withNumHeads( static_cast<dim_t>(metadata.num_heads) )
                .withHiddenSize( static_cast<dim_t>(metadata.hidden_dim) )
                .withBias( metadata.use_bias );

            return config;
        }
    };
}