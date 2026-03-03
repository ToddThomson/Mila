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
     * The network-level structure follows LLaMA conventions (RMSNorm and SwiGLU).
     * This implementation uses `GptBlock` as the per-layer block backend.
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
        using LinearType = Linear<TDeviceType, TPrecision>;
        using RmsNormType = RmsNorm<TDeviceType, TPrecision>;
        using TransformerBlockType = LlamaBlock<TDeviceType, TPrecision>;
        using EmbeddingType = Rope<TDeviceType, TPrecision>;
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
            {
                throw std::invalid_argument( "LlamaTransformer::fromPretrained: batch_size must be > 0" );
            }

            std::size_t runtime_seq = seq_length;
            if ( runtime_seq == 0 )
            {
                throw std::invalid_argument( "LlamaTransformer::fromPretrained: seq_length must be > 0" );
            }

            runtime_seq = std::min<std::size_t>( runtime_seq, static_cast<std::size_t>(config.getMaxSequenceLength()) );

            shape_t build_shape = {
                static_cast<dim_t>(batch_size),
                static_cast<dim_t>(runtime_seq)
            };
            llama->build( build_shape );

            llama->loadParameters( reader, strict );

            return llama;
        }

        TensorType& forward( const TokenIndexType& input )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "LlamaTransformer must be built before calling forward()." );
            }

            encoder_out_ptr_ = &encoder_->forward( input );
            this->getExecutionContext()->synchronize();

            if ( block_input_ptrs_.empty() || block_input_ptrs_.size() != transformer_blocks_.size() )
            {
                throw std::runtime_error( "LlamaTransformer: forward internal state not initialized" );
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

            normalized_ptr_ = &final_rmsnorm_->forward( *block_output_ptrs_.back() );
            this->getExecutionContext()->synchronize();

            logits_ptr_ = &lm_head_->forward( *normalized_ptr_ );
            this->getExecutionContext()->synchronize();

            return *logits_ptr_;
        }

        TokenIndexType& backward( const TokenIndexType& input, const TensorType& output_grad )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "LlamaTransformer must be built before calling backward()." );
            }

            if ( !this->isTraining() )
            {
                throw std::runtime_error( "LlamaTransformer: backward requires training mode (setTraining(true))." );
            }

            if ( !encoder_out_ptr_ )
            {
                throw std::runtime_error( "LlamaTransformer: forward activations not present for backward. Call forward() before backward()." );
            }

            for ( size_t i = 0; i < transformer_blocks_.size(); ++i )
            {
                if ( !block_input_ptrs_[ i ] || !block_output_ptrs_[ i ] )
                {
                    throw std::runtime_error( std::format( "LlamaTransformer: missing cached activation for block {}", i ) );
                }
            }

            if ( !normalized_ptr_ || !logits_ptr_ )
            {
                throw std::runtime_error( "LlamaTransformer: missing final activations for backward" );
            }

            auto& normalized_grad_ptr = lm_head_->backward( *normalized_ptr_, output_grad );
            this->getExecutionContext()->synchronize();

            auto& last_block_grad_ptr = final_rmsnorm_->backward( *block_output_ptrs_.back(), normalized_grad_ptr );
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

        TensorType& decode( const TokenIndexType& input, int position )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "LlamaTransformer must be built before calling decode()." );
            }

            encoder_out_ptr_ = &encoder_->decode( input, position );
            this->getExecutionContext()->synchronize();

            if ( block_input_ptrs_.empty() || block_input_ptrs_.size() != transformer_blocks_.size() )
            {
                throw std::runtime_error( "LlamaTransformer: decode internal state not initialized" );
            }

            block_input_ptrs_[ 0 ] = encoder_out_ptr_;

            for ( size_t i = 0; i < transformer_blocks_.size(); ++i )
            {
                auto& block_out = transformer_blocks_[ i ]->decode( *block_input_ptrs_[ i ], position );
                this->getExecutionContext()->synchronize();

                block_output_ptrs_[ i ] = &block_out;

                if ( i + 1 < transformer_blocks_.size() )
                {
                    block_input_ptrs_[ i + 1 ] = &block_out;
                }
            }

            normalized_ptr_ = &final_rmsnorm_->forward( *block_output_ptrs_.back() );
            this->getExecutionContext()->synchronize();

            logits_ptr_ = &lm_head_->decode( *normalized_ptr_ );
            this->getExecutionContext()->synchronize();

            return *logits_ptr_;
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

            final_rmsnorm_->zeroGradients();
            lm_head_->zeroGradients();
        }

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
            oss << "  Embedding Size/Dim: " << config_.getEmbeddingSize() << std::endl;
            oss << "  Number of heads: " << config_.getNumHeads() << std::endl;
            oss << "  Number of KV heads: " << config_.getNumKVHeads() << std::endl;
            oss << "  Number of layers: " << config_.getNumLayers() << std::endl;
            oss << "  MLP Hidden Size/Dim: " << config_.getHiddenDimension() << std::endl;
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
                    {
                        throw std::runtime_error( "Component not found: " + component_path );
                    }

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
                .set( "embedding_dim", config_.getEmbeddingSize() )
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

            embedding_shape_ = { batch_size_, seq_length_, config_.getEmbeddingSize() };
            output_shape_ = { batch_size_, seq_length_, config_.getVocabSize() };

            transformer_blocks_.clear();
            transformer_blocks_.reserve( static_cast<size_t>(config_.getNumLayers()) );

            encoder_ = this->template getComponentAs<EmbeddingType>( this->getName() + ".lenc" );
            encoder_->build( input_shape );

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

        std::shared_ptr<EmbeddingType> encoder_{ nullptr };
        std::vector<std::shared_ptr<TransformerBlockType>> transformer_blocks_;
        std::shared_ptr<RmsNormType> final_rmsnorm_{ nullptr };
        std::shared_ptr<LinearType> lm_head_{ nullptr };

        TensorType* encoder_out_ptr_{ nullptr };
        std::vector<TensorType*> block_input_ptrs_;
        std::vector<TensorType*> block_output_ptrs_;
        TensorType* normalized_ptr_{ nullptr };
        TensorType* logits_ptr_{ nullptr };

        std::pair<std::string, std::string> parseParameterPath( const std::string& full_name ) const
        {
            auto last_dot = full_name.rfind( '.' );

            if ( last_dot == std::string::npos )
            {
                throw std::runtime_error(
                    std::format( "Invalid parameter path: {}", full_name ) );
            }

            std::string component_path = full_name.substr( 0, last_dot );
            std::string param_name = full_name.substr( last_dot + 1 );

            return { component_path, param_name };
        }

        void validateInputShape( const shape_t& input_shape ) const
        {
            if ( input_shape.size() != 2 )
            {
                throw std::invalid_argument(
                    "LlamaTransformer: input must have rank 2 (batch_size, seq_length)" );
            }

            if ( input_shape[ 1 ] > config_.getMaxSequenceLength() )
            {
                throw std::invalid_argument(
                    std::format( "LlamaTransformer: sequence length {} exceeds maximum {}",
                        input_shape[ 1 ], config_.getMaxSequenceLength() ) );
            }
        }

        void createGraph()
        {
            LlamaConfig enc_cfg;
            enc_cfg.withVocabularyLength( static_cast<size_t>(config_.getVocabSize()) )
                .withMaxSequenceLength( static_cast<size_t>(config_.getMaxSequenceLength()) )
                .withEmbeddingDim( static_cast<size_t>(config_.getEmbeddingSize()) );

            enc_cfg.validate();

            auto encoder = std::make_shared<EmbeddingType>(
                this->getName() + ".rope", enc_cfg );

            this->addComponent( encoder );

            for ( int64_t i = 0; i < config_.getNumLayers(); ++i )
            {
                LlamaConfig block_cfg(
                    static_cast<dim_t>( config_.getEmbeddingSize() ),
                    static_cast<dim_t>( config_.getNumHeads() ) );

                block_cfg.withHiddenSize( static_cast<dim_t>( config_.getHiddenDimension() ) )
                    .withBias( config_.useBias() )
                    .withActivation( ActivationType::Swiglu )
                    .withMaxSequenceLength( static_cast<dim_t>( config_.getMaxSequenceLength() ) )
                    .withResidualScale( 1.0f );

                auto layer = std::make_shared<TransformerBlockType>(
                    this->getName() + ".tf_layer_" + std::to_string( i ), block_cfg, std::nullopt );

                this->addComponent( layer );
            }

            auto rms_config = RmsNormConfig()
                .withNormalizedShape( { config_.getEmbeddingSize() } )
                .withEpsilon( config_.getRMSNormEpsilon() )
                .withBias( false );

            auto final_rmsnorm = std::make_shared<RmsNormType>(
                this->getName() + ".rms_final", rms_config, std::nullopt );

            this->addComponent( final_rmsnorm );

            auto lm_head_config = LinearConfig( config_.getEmbeddingSize(), config_.getVocabSize() )
                .withBias( false )
                .withRowMajor( true );

            auto lm_head = std::make_shared<LinearType>(
                this->getName() + ".lm_head", lm_head_config, std::nullopt );

            this->addComponent( lm_head );
        }

        static auto createConfigFromMetadata( const PretrainedMetadata& metadata ) -> LlamaConfig
        {
            dim_t embedding_size = static_cast<dim_t>(metadata.embedding_dim);
            dim_t num_layers = static_cast<dim_t>(metadata.num_layers);

            LlamaConfig config = LlamaConfig( embedding_size, num_layers );

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