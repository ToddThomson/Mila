/**
 * @file Llama.ixx
 * @brief LLaMA-style decoder-only transformer network.
 *
 * Device-templated network implementing a LLaMA-style autoregressive decoder.
 */

module;
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <cstdint>
#include <format>
#include <optional>
#include <filesystem>

export module Dnn.Components.LlamaTransformer;
export import :Config;
export import :Presets;
export import :Block;

import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.Network;
import Dnn.Component;
import Dnn.ComponentType;
import Dnn.Components.TokenEmbedding;
import Dnn.Components.Linear;
import Dnn.Components.RmsNorm;
import Dnn.Components.Rope;
import Dnn.ActivationType;
import Compute.Precision;
import Compute.Device;
import Compute.DeviceType;
import Compute.DeviceId;
import Compute.DeviceTypeTraits;
import Compute.CpuMemoryResource;
import Compute.ExecutionContext;
import Compute.ExecutionContextFactory;
import Serialization.ModelArchive;
import Serialization.Tensor;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief LLaMA-style transformer (decoder-only) for autoregressive token prediction.
     *
     * Graph: TokenEmbedding → RoPE → LlamaBlock × N → RmsNorm → Linear (lm_head).
     * RoPE is applied to the full embedding stream after the token lookup; each
     * LlamaBlock receives rotary-encoded embeddings as input.
     *
     * Template parameters:
     *  - TDeviceType: device type (Cpu/Cuda)
     *  - TPrecision: tensor precision
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class LlamaTransformer : public Network<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using NetworkBase = Network<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using TokenEmbeddingType = TokenEmbedding<TDeviceType, dtype_t::INT32, TPrecision>;
        using RopeType = Rope<TDeviceType, TPrecision>;
        using LinearType = Linear<TDeviceType, TPrecision>;
        using RmsNormType = RmsNorm<TDeviceType, TPrecision>;
        using TransformerBlockType = LlamaBlock<TDeviceType, TPrecision>;
        using TokenIndexType = Tensor<dtype_t::INT32, MR>;
        using ComponentPtr = typename NetworkBase::ComponentPtr;

        explicit LlamaTransformer( const std::string& name, const LlamaConfig& config, DeviceId device_id )
            : NetworkBase( name ), owned_context_( createExecutionContext( device_id ) ), config_( config )
        {
            config_.validate();

            if ( device_id.type != TDeviceType )
            {
                throw std::invalid_argument(
                    std::format( "LlamaTransformer: device type mismatch: expected {}, got {}",
                        deviceTypeToString( TDeviceType ),
                        deviceTypeToString( device_id.type ) ) );
            }

            createGraph();

            this->setExecutionContext( owned_context_.get() );
        }

        ~LlamaTransformer() override = default;

        static std::unique_ptr<LlamaTransformer<TDeviceType, TPrecision>> fromPretrained(
            const std::filesystem::path& model_path,
            std::size_t batch_size,
            std::size_t seq_length,
            DeviceId device_id = DeviceId{ TDeviceType, 0 },
            bool strict = true )
        {
            PretrainedModelReader reader( model_path );
            const auto& metadata = reader.getPretrainedMetadata();

            LlamaConfig config = createConfigFromMetadata( metadata );

            auto llama = std::make_unique<LlamaTransformer<TDeviceType, TPrecision>>(
                metadata.model_name,
                config,
                device_id );

            if ( batch_size == 0 )
                throw std::invalid_argument( "LlamaTransformer::fromPretrained: batch_size must be > 0" );

            if ( seq_length == 0 )
                throw std::invalid_argument( "LlamaTransformer::fromPretrained: seq_length must be > 0" );

            std::size_t runtime_seq = std::min<std::size_t>(
                seq_length, static_cast<std::size_t>(config.getMaxSequenceLength()) );

            llama->build( shape_t{
                static_cast<dim_t>(batch_size),
                static_cast<dim_t>(runtime_seq) } );

            llama->loadParameters( reader, strict );

            return llama;
        }

        // ====================================================================
        // Forward / Decode
        // ====================================================================

        TensorType& forward( const TokenIndexType& input )
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "LlamaTransformer must be built before calling forward()." );

            auto& embed_out = token_embedding_->forward( input );
            this->getExecutionContext()->synchronize();

            token_embed_out_ptr_ = &embed_out;

            encoder_out_ptr_ = &rope_->forward( embed_out );
            this->getExecutionContext()->synchronize();

            if ( block_input_ptrs_.empty() || block_input_ptrs_.size() != transformer_blocks_.size() )
                throw std::runtime_error( "LlamaTransformer: forward internal state not initialized" );

            block_input_ptrs_[ 0 ] = encoder_out_ptr_;

            for ( size_t i = 0; i < transformer_blocks_.size(); ++i )
            {
                auto& block_out = transformer_blocks_[ i ]->forward( *block_input_ptrs_[ i ] );
                this->getExecutionContext()->synchronize();

                block_output_ptrs_[ i ] = &block_out;

                if ( i + 1 < transformer_blocks_.size() )
                    block_input_ptrs_[ i + 1 ] = &block_out;
            }

            normalized_ptr_ = &final_rmsnorm_->forward( *block_output_ptrs_.back() );
            this->getExecutionContext()->synchronize();

            logits_ptr_ = &lm_head_->forward( *normalized_ptr_ );
            this->getExecutionContext()->synchronize();

            return *logits_ptr_;
        }

        TensorType& decode( const TokenIndexType& input, int position )
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "LlamaTransformer must be built before calling decode()." );

            auto& embed_out = token_embedding_->forward( input );
            this->getExecutionContext()->synchronize();

            token_embed_out_ptr_ = &embed_out;

            // FIXME: rope_ applies rotation at sequence offset 0 for T=1 input.
            //        Add Rope::decode(input, position) to apply the correct
            //        rotary frequencies for the given KV cache position.
            encoder_out_ptr_ = &rope_->forward( embed_out );
            this->getExecutionContext()->synchronize();

            if ( block_input_ptrs_.empty() || block_input_ptrs_.size() != transformer_blocks_.size() )
                throw std::runtime_error( "LlamaTransformer: decode internal state not initialized" );

            block_input_ptrs_[ 0 ] = encoder_out_ptr_;

            for ( size_t i = 0; i < transformer_blocks_.size(); ++i )
            {
                auto& block_out = transformer_blocks_[ i ]->decode( *block_input_ptrs_[ i ], position );
                this->getExecutionContext()->synchronize();

                block_output_ptrs_[ i ] = &block_out;

                if ( i + 1 < transformer_blocks_.size() )
                    block_input_ptrs_[ i + 1 ] = &block_out;
            }

            normalized_ptr_ = &final_rmsnorm_->forward( *block_output_ptrs_.back() );
            this->getExecutionContext()->synchronize();

            logits_ptr_ = &lm_head_->decode( *normalized_ptr_ );
            this->getExecutionContext()->synchronize();

            return *logits_ptr_;
        }

        // ====================================================================
        // Backward
        // ====================================================================

        TokenIndexType& backward( const TokenIndexType& input, const TensorType& output_grad )
        {
            if ( !this->isBuilt() )
                throw std::runtime_error( "LlamaTransformer must be built before calling backward()." );

            if ( !this->isTraining() )
                throw std::runtime_error( "LlamaTransformer: backward requires training mode (setTraining(true))." );

            if ( !token_embed_out_ptr_ || !encoder_out_ptr_ )
                throw std::runtime_error( "LlamaTransformer: forward activations not present. Call forward() before backward()." );

            for ( size_t i = 0; i < transformer_blocks_.size(); ++i )
            {
                if ( !block_input_ptrs_[ i ] || !block_output_ptrs_[ i ] )
                    throw std::runtime_error( std::format( "LlamaTransformer: missing cached activation for block {}", i ) );
            }

            if ( !normalized_ptr_ || !logits_ptr_ )
                throw std::runtime_error( "LlamaTransformer: missing final activations for backward" );

            auto& normalized_grad = lm_head_->backward( *normalized_ptr_, output_grad );
            this->getExecutionContext()->synchronize();

            auto& last_block_grad = final_rmsnorm_->backward( *block_output_ptrs_.back(), normalized_grad );
            this->getExecutionContext()->synchronize();

            TensorType* curr_grad = &last_block_grad;

            for ( int64_t i = static_cast<int64_t>(transformer_blocks_.size()) - 1; i >= 0; --i )
            {
                auto& block_grad = transformer_blocks_[ static_cast<size_t>(i) ]->backward(
                    *block_input_ptrs_[ static_cast<size_t>(i) ], *curr_grad );

                curr_grad = &block_grad;

                this->getExecutionContext()->synchronize();
            }

            // Backprop through RoPE then token embedding.
            auto& rope_in_grad = rope_->backward( *token_embed_out_ptr_, *curr_grad );
            this->getExecutionContext()->synchronize();

            auto& input_grad = token_embedding_->backward( input, rope_in_grad );
            this->getExecutionContext()->synchronize();

            return input_grad;
        }

        // ====================================================================
        // Gradient management
        // ====================================================================

        void zeroGradients() override
        {
            if ( !this->isBuilt() )
                return;

            token_embedding_->zeroGradients();
            rope_->zeroGradients();

            for ( auto& block : transformer_blocks_ )
                block->zeroGradients();

            final_rmsnorm_->zeroGradients();
            lm_head_->zeroGradients();
        }

        // ====================================================================
        // Accessors / Diagnostics
        // ====================================================================

        const ComponentType getType() const override
        {
            return ComponentType::Llama;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << std::endl;
            oss << "Llama Network: " << this->getName() << std::endl;
            oss << "Device: " << this->getDeviceId().toString() << std::endl;

            oss << "Architecture:" << std::endl;
            oss << "  Vocabulary: " << config_.getVocabSize() << " tokens" << std::endl;
            oss << "  Max sequence length: " << config_.getMaxSequenceLength() << std::endl;
            oss << "  Model dim: " << config_.getModelDim() << std::endl;
            oss << "  Number of heads: " << config_.getNumHeads() << std::endl;
            oss << "  Number of KV heads: " << config_.getNumKVHeads() << std::endl;
            oss << "  Number of layers: " << config_.getNumLayers() << std::endl;
            oss << "  MLP hidden dim: " << config_.getHiddenDimension() << std::endl;
            oss << "  RoPE theta: " << config_.getRoPETheta() << std::endl;
            oss << "  RoPE scaling factor: " << config_.getRoPEScalingFactor() << std::endl;

            if ( this->isBuilt() )
            {
                oss << "  Parameters: " << this->parameterCount() << std::endl;
                oss << "  Batch size: " << batch_size_ << std::endl;
                oss << "  Sequence length: " << seq_length_ << std::endl;
            }

            return oss.str();
        }

        IExecutionContext* getExecutionContext() const
        {
            return NetworkBase::getExecutionContext();
        }

        void loadParameters( PretrainedModelReader& reader, bool strict )
        {
            for ( const auto& full_name : reader.getTensorNames() )
            {
                auto [component_path, param_name] = parseParameterPath( full_name );

                auto target = this->findComponent( component_path );

                if ( !target )
                {
                    if ( strict )
                        throw std::runtime_error( "Component not found: " + component_path );

                    continue;
                }

                auto blob = reader.readTensorBlob( full_name );

                target->loadParameter( param_name, blob );
            }
        }

    protected:

        void save_( ModelArchive& archive, SerializationMode /*mode*/ ) const override
        {
            SerializationMetadata meta;
            meta.set( "type", "LlamaTransformer" )
                .set( "version", int64_t( 1 ) )
                .set( "name", this->getName() )
                .set( "vocab_size", config_.getVocabSize() )
                .set( "max_seq_length", config_.getMaxSequenceLength() )
                .set( "model_dim", config_.getModelDim() )
                .set( "num_heads", config_.getNumHeads() )
                .set( "num_kv_heads", config_.getNumKVHeads() )
                .set( "num_layers", config_.getNumLayers() )
                .set( "hidden_dim", config_.getHiddenDimension() )
                .set( "rope_theta", static_cast<double>(config_.getRoPETheta()) )
                .set( "rope_scaling", static_cast<double>(config_.getRoPEScalingFactor()) )
                .set( "use_bias", config_.useBias() );

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

            embedding_shape_ = { batch_size_, seq_length_, config_.getModelDim() };
            output_shape_ = { batch_size_, seq_length_, config_.getVocabSize() };

            transformer_blocks_.clear();
            transformer_blocks_.reserve( static_cast<size_t>(config_.getNumLayers()) );

            token_embedding_ = this->template getComponentAs<TokenEmbeddingType>( this->getName() + ".wte" );
            token_embedding_->build( input_shape );

            rope_ = this->template getComponentAs<RopeType>( this->getName() + ".rope" );
            rope_->build( embedding_shape_ );

            for ( int64_t i = 0; i < config_.getNumLayers(); ++i )
            {
                std::string block_name = this->getName() + ".tf_layer_" + std::to_string( i );
                auto block = this->template getComponentAs<TransformerBlockType>( block_name );
                block->build( embedding_shape_ );
                transformer_blocks_.push_back( block );
            }

            final_rmsnorm_ = this->template getComponentAs<RmsNormType>( this->getName() + ".rms_final" );
            final_rmsnorm_->build( embedding_shape_ );

            lm_head_ = this->template getComponentAs<LinearType>( this->getName() + ".lm_head" );
            lm_head_->build( embedding_shape_ );

            block_input_ptrs_.assign( transformer_blocks_.size(), nullptr );
            block_output_ptrs_.assign( transformer_blocks_.size(), nullptr );

            token_embed_out_ptr_ = nullptr;
            encoder_out_ptr_ = nullptr;
            normalized_ptr_ = nullptr;
            logits_ptr_ = nullptr;
        }

    private:

        std::unique_ptr<IExecutionContext> owned_context_{ nullptr };

        LlamaConfig config_;

        shape_t input_shape_;
        shape_t embedding_shape_;
        shape_t output_shape_;
        int64_t batch_size_{ 0 };
        int64_t seq_length_{ 0 };

        std::shared_ptr<TokenEmbeddingType> token_embedding_{ nullptr };
        std::shared_ptr<RopeType>           rope_{ nullptr };
        std::vector<std::shared_ptr<TransformerBlockType>> transformer_blocks_;
        std::shared_ptr<RmsNormType> final_rmsnorm_{ nullptr };
        std::shared_ptr<LinearType>  lm_head_{ nullptr };

        // Activation pointers — valid between forward() and the next backward().
        TensorType* token_embed_out_ptr_{ nullptr };  // rope's input
        TensorType* encoder_out_ptr_{ nullptr };       // rope's output / blocks' input
        std::vector<TensorType*> block_input_ptrs_;
        std::vector<TensorType*> block_output_ptrs_;
        TensorType* normalized_ptr_{ nullptr };
        TensorType* logits_ptr_{ nullptr };

        // ====================================================================
        // Graph construction
        // ====================================================================

        void createGraph()
        {
            // 1. Token embedding — pure vocabulary lookup, no positional information.
            TokenEmbeddingConfig tok_cfg;
            tok_cfg.withVocabSize( static_cast<size_t>(config_.getVocabSize()) )
                .withEmbeddingDim( static_cast<size_t>(config_.getModelDim()) );

            auto tok_emb = std::make_shared<TokenEmbeddingType>(
                this->getName() + ".wte", tok_cfg );

            this->addComponent( tok_emb );

            // 2. RoPE — rotary positional encoding applied to the embedding stream.
            RopeConfig rope_cfg;
            rope_cfg.withChannels( static_cast<size_t>(config_.getModelDim()) )
                .withNumHeads( static_cast<size_t>(config_.getNumHeads()) )
                .withNumKvHeads( static_cast<size_t>(config_.getNumKVHeads()) )
                .withMaxSequenceLength( static_cast<size_t>(config_.getMaxSequenceLength()) )
                .withBase( config_.getRoPETheta() );

            auto rope = std::make_shared<RopeType>(
                this->getName() + ".rope", rope_cfg );

            this->addComponent( rope );

            // 3. Transformer blocks.
            for ( int64_t i = 0; i < config_.getNumLayers(); ++i )
            {
                LlamaConfig block_cfg( config_.getModelDim(), 1 );
                block_cfg.withNumHeads( config_.getNumHeads() )
                    .withNumKVHeads( config_.getNumKVHeads() )
                    .withHiddenDimension( config_.getHiddenDimension() )
                    .withBias( config_.useBias() )
                    .withRoPETheta( config_.getRoPETheta() )
                    .withMaxSequenceLength( config_.getMaxSequenceLength() );

                auto layer = std::make_shared<TransformerBlockType>(
                    this->getName() + ".tf_layer_" + std::to_string( i ), block_cfg, std::nullopt );

                this->addComponent( layer );
            }

            // 4. Final RMSNorm.
            auto rms_config = RmsNormConfig()
                .withNormalizedShape( { config_.getModelDim() } )
                .withEpsilon( config_.getRMSNormEpsilon() )
                .withBias( false );

            auto final_rmsnorm = std::make_shared<RmsNormType>(
                this->getName() + ".rms_final", rms_config, std::nullopt );

            this->addComponent( final_rmsnorm );

            // 5. Language model head — projects model_dim → vocab_size, no bias.
            auto lm_head_config = LinearConfig( config_.getModelDim(), config_.getVocabSize() )
                .withBias( false )
                .withRowMajor( true );

            auto lm_head = std::make_shared<LinearType>(
                this->getName() + ".lm_head", lm_head_config, std::nullopt );

            this->addComponent( lm_head );
        }

        // ====================================================================
        // Helpers
        // ====================================================================

        std::pair<std::string, std::string> parseParameterPath( const std::string& full_name ) const
        {
            auto last_dot = full_name.rfind( '.' );

            if ( last_dot == std::string::npos )
                throw std::runtime_error( std::format( "Invalid parameter path: {}", full_name ) );

            return { full_name.substr( 0, last_dot ), full_name.substr( last_dot + 1 ) };
        }

        void validateInputShape( const shape_t& input_shape ) const
        {
            if ( input_shape.size() != 2 )
                throw std::invalid_argument(
                    "LlamaTransformer: input must have rank 2 (batch_size, seq_length)" );

            if ( input_shape[ 1 ] > config_.getMaxSequenceLength() )
                throw std::invalid_argument(
                    std::format( "LlamaTransformer: sequence length {} exceeds maximum {}",
                        input_shape[ 1 ], config_.getMaxSequenceLength() ) );
        }

        static LlamaConfig createConfigFromMetadata( const PretrainedMetadata& metadata )
        {
            LlamaConfig config( static_cast<dim_t>(metadata.embedding_dim),
                                static_cast<dim_t>(metadata.num_layers) );

            config.withVocabularyLength( static_cast<dim_t>(metadata.vocab_size) )
                .withMaxSequenceLength( static_cast<dim_t>(metadata.max_seq_length) )
                .withNumHeads( static_cast<dim_t>(metadata.num_heads) )
                .withNumKVHeads( static_cast<dim_t>(metadata.num_kv_heads) )
                .withHiddenDimension( static_cast<dim_t>(metadata.hidden_dim) )
                .withRoPETheta( metadata.rope_theta )
                // FIXME: .withRoPEScalingFactor( metadata.rope_scaling )
                .withBias( metadata.use_bias );

            return config;
        }
    };
}