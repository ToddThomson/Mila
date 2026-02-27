/**
 * @file Gpt2Transformer.ixx
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

export module Gpt2.Transformer;

import Mila;
import Dnn.Network;
import Cuda.Error;
import Gpt2.CheckpointReader;

namespace Mila::Gpt2
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief GPT-2 style transformer (decoder-only) for autoregressive token prediction.
     *
     * Construction and serialization patterns mirror CharTransformer:
     *  - Create and own ExecutionContext for specified device
     *  - Build component graph via createGraph() (context-independent)
     *  - Propagate context via setExecutionContext()
     *  - onBuilding() builds children with shapes and allocates outputs
     *
     * Template parameters:
     *  - TDeviceType: device type (Cpu/Cuda)
     *  - TPrecision: tensor precision
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class GptTransformer : public Network<TDeviceType, TPrecision>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using NetworkBase = Network<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using LinearType = Linear<TDeviceType, TPrecision>;
        using LayerNormType = LayerNorm<TDeviceType, TPrecision>;
        using TransformerBlockType = Transformer<TDeviceType, TPrecision>;
        using EncoderType = Encoder<TDeviceType, dtype_t::INT32, TPrecision>;
        using TokenIndexType = Tensor<dtype_t::INT32, MR>;
        using MLPType = MLP<TDeviceType, TPrecision>;
        using ComponentPtr = typename NetworkBase::ComponentPtr;

        /**
         * @brief Construct GptTransformer.
         *
         * @param name Network name
         * @param config GPT transformer configuration
         * @param device_id Device identifier for execution
         *
         * @throws std::invalid_argument on invalid config or device mismatch
         */
        explicit GptTransformer( const std::string& name, const Gpt2Config& config, DeviceId device_id )
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

        ~GptTransformer() override = default;

        /**
         * @brief Load GptTransformer from archive.
         *
         * Reads metadata, constructs network, builds with saved shape and loads weights.
         */
        static std::unique_ptr<GptTransformer> Load( ModelArchive& archive, DeviceId device_id )
        {
            auto scope = ModelArchive::ScopedScope( archive, "network" );

            SerializationMetadata meta = archive.readMetadata( "transformer_meta.json" );

            GptTransformerConfig config;

            config.vocab_size = meta.getInt( "vocab_size" );
            config.max_seq_length = meta.getInt( "max_seq_length" );
            config.embedding_dim = meta.getInt( "embedding_dim" );
            config.num_heads = meta.getInt( "num_heads" );
            config.num_layers = meta.getInt( "num_layers" );
            config.mlp_hidden_dim = meta.getInt( "mlp_hidden_dim" );

            auto transformer = std::make_unique<GptTransformer>( archive.getName(), config, device_id );

            shape_t input_shape = meta.getShape( "input_shape" );
            transformer->build( input_shape );

            loadComponentWeights( archive, transformer.get() );

            return transformer;
        }

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

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << std::endl;
            oss << "GptTransformer: " << this->getName() << std::endl;
            oss << "Device: " << this->getDeviceId().toString() << std::endl;

            oss << "Architecture:" << std::endl;
            oss << "  Vocabulary: " << config_.vocab_size << " tokens" << std::endl;
            oss << "  Max sequence length: " << config_.max_seq_length << std::endl;
            oss << "  Embedding dimension: " << config_.embedding_dim << std::endl;
            oss << "  Number of heads: " << config_.num_heads << std::endl;
            oss << "  Number of layers: " << config_.num_layers << std::endl;
            oss << "  MLP hidden dimension: " << config_.mlp_hidden_dim << std::endl;

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
        void initializeFromCheckpoint( const std::string& checkpoint_path )
        {
            static_assert( std::is_same_v<typename TensorDataTypeTraits<TPrecision>::value_type, float>,
                "initializeFromCheckpoint requires TPrecision == dtype_t::FP32." );

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
        }

    protected:
        void save_( ModelArchive& archive, SerializationMode /*mode*/ ) const override
        {
            SerializationMetadata meta;
            meta.set( "type", "GptTransformer" )
                .set( "version", int64_t( 1 ) )
                .set( "name", this->getName() )
                .set( "vocab_size", config_.vocab_size )
                .set( "max_seq_length", config_.max_seq_length )
                .set( "embedding_dim", config_.embedding_dim )
                .set( "num_heads", config_.num_heads )
                .set( "num_layers", config_.num_layers )
                .set( "mlp_hidden_dim", config_.mlp_hidden_dim );

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

            embedding_shape_ = { batch_size_, seq_length_, config_.embedding_dim };
            output_shape_ = { batch_size_, seq_length_, config_.vocab_size };

            encoder_ = this->template getComponentAs<EncoderType>( this->getName() + ".lenc" );
            encoder_->build( input_shape );

            for ( int64_t i = 0; i < config_.num_layers; ++i )
            {
                std::string block_name = this->getName() + ".tf.layer_" + std::to_string( i );
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

        Gpt2Config config_;

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

        // -------------------------------------------------------------------
        // Helpers extracted from initializeFromCheckpoint to keep it short
        // -------------------------------------------------------------------

        void validateCheckpointCompatibility( const ModelConfig& ckpt_cfg ) const
        {
            if ( static_cast<size_t>( config_.num_layers ) != static_cast<size_t>( ckpt_cfg.num_layers ) )
            {
                throw std::runtime_error( std::format( "Checkpoint has {} layers but transformer config expects {}",
                    ckpt_cfg.num_layers, config_.num_layers ) );
            }

            if ( static_cast<size_t>( config_.embedding_dim ) != static_cast<size_t>( ckpt_cfg.channels ) )
            {
                throw std::runtime_error( std::format( "Checkpoint embedding dim {} != network embedding dim {}",
                    ckpt_cfg.channels, config_.embedding_dim ) );
            }
        }

        void copyHostVectorToComponentTensor( const std::vector<float>& src, ITensor* dst_itensor )
        {
            if ( !dst_itensor )
            {
                throw std::runtime_error( "initializeFromCheckpoint: destination tensor is null" );
            }

            if ( dst_itensor->getDataType() != dtype_t::FP32 )
            {
                throw std::runtime_error( "initializeFromCheckpoint: destination tensor is not FP32" );
            }

            const size_t dst_elems = dst_itensor->size();
            if ( dst_elems != src.size() )
            {
                std::ostringstream oss;
                oss << "initializeFromCheckpoint: size mismatch (dst=" << dst_elems << ", src=" << src.size() << ")";
                throw std::runtime_error( oss.str() );
            }

            using HostTensor = Tensor<dtype_t::FP32, CpuMemoryResource>;
            HostTensor host_tensor( Device::Cpu(), shape_t{ static_cast<int64_t>( dst_elems ) } );

            std::memcpy( const_cast<void*>( host_tensor.rawData() ), src.data(), dst_elems * sizeof( float ) );

            Tensor<dtype_t::FP32, MR>& dst_tensor = *reinterpret_cast<Tensor<dtype_t::FP32, MR>*>( dst_itensor );

            copy( host_tensor, dst_tensor );
            this->getExecutionContext()->synchronize();
        }

        void copySliceToParam( const std::vector<float>& src, size_t offset, size_t count, ITensor* dst )
        {
            if ( count == 0 )
                return;

            std::vector<float> slice;
            slice.assign( src.begin() + offset, src.begin() + offset + count );

            copyHostVectorToComponentTensor( slice, dst );
        }

        void applyEncoderParams( const ParameterTensors& params )
        {
            if ( !encoder_ )
                return;

            auto enc_params = encoder_->getParameters();

            if ( enc_params.size() >= 1 && !params.wte.empty() )
                copyHostVectorToComponentTensor( params.wte, enc_params[ 0 ] );

            if ( enc_params.size() >= 2 && !params.wpe.empty() )
                copyHostVectorToComponentTensor( params.wpe, enc_params[ 1 ] );
        }

        void applyLayerParams( size_t layer, const ParameterTensors& params )
        {
            const size_t C = static_cast<size_t>( config_.embedding_dim );

            const std::string block_name = this->getName() + ".tf_layer_" + std::to_string( layer );

            // ln1
            if ( auto ln1 = this->template getComponentAs<LayerNormType>( block_name + ".lnorm_1" ) )
            {
                auto p = ln1->getParameters();
                const size_t per = C;
                const size_t off = layer * per;

                if ( p.size() >= 1 ) copySliceToParam( params.ln1w, off, per, p[ 0 ] );
                if ( p.size() >= 2 ) copySliceToParam( params.ln1b, off, per, p[ 1 ] );
            }

            // qkv projection
            if ( auto qkv = this->template getComponentAs<LinearType>( block_name + ".fc_qkv_proj" ) )
            {
                auto p = qkv->getParameters();
                const size_t per_w = (3 * C) * C;
                const size_t per_b = 3 * C;
                const size_t off_w = layer * per_w;
                const size_t off_b = layer * per_b;

                if ( p.size() >= 1 ) copySliceToParam( params.qkvw, off_w, per_w, p[ 0 ] );
                if ( p.size() >= 2 && !params.qkvb.empty() ) copySliceToParam( params.qkvb, off_b, per_b, p[ 1 ] );
            }

            // attn proj (optional)
            if ( auto att_proj = this->template getComponentAs<LinearType>( block_name + ".attn_proj" ) )
            {
                auto p = att_proj->getParameters();
                const size_t per_w = C * C;
                const size_t per_b = C;
                const size_t off_w = layer * per_w;
                const size_t off_b = layer * per_b;

                if ( p.size() >= 1 ) copySliceToParam( params.attprojw, off_w, per_w, p[ 0 ] );
                if ( p.size() >= 2 && !params.attprojb.empty() ) copySliceToParam( params.attprojb, off_b, per_b, p[ 1 ] );
            }

            // ln2
            if ( auto ln2 = this->template getComponentAs<LayerNormType>( block_name + ".lnorm_2" ) )
            {
                auto p = ln2->getParameters();
                const size_t per = C;
                const size_t off = layer * per;

                if ( p.size() >= 1 ) copySliceToParam( params.ln2w, off, per, p[ 0 ] );
                if ( p.size() >= 2 ) copySliceToParam( params.ln2b, off, per, p[ 1 ] );
            }

            // ffn (MLP) -> fc1 and fc2
            if ( auto ffn = this->template getComponentAs<MLPType>( block_name + ".mlp" ) )
            {
                // fc1
                try
                {
                    auto fc1 = ffn->template getComponentAs<LinearType>( ffn->getName() + ".fc1" );
                    if ( fc1 )
                    {
                        auto p = fc1->getParameters();
                        const size_t per_w1 = (4 * C) * C;
                        const size_t per_b1 = 4 * C;
                        const size_t off_w1 = layer * per_w1;
                        const size_t off_b1 = layer * per_b1;

                        if ( p.size() >= 1 ) copySliceToParam( params.fcw, off_w1, per_w1, p[ 0 ] );
                        if ( p.size() >= 2 && !params.fcb.empty() ) copySliceToParam( params.fcb, off_b1, per_b1, p[ 1 ] );
                    }
                }
                catch ( ... ) {}

                // fc2
                try
                {
                    auto fc2 = ffn->template getComponentAs<LinearType>( ffn->getName() + ".fc2" );
                    if ( fc2 )
                    {
                        auto p = fc2->getParameters();
                        const size_t per_w2 = C * (4 * C);
                        const size_t per_b2 = C;
                        const size_t off_w2 = layer * per_w2;
                        const size_t off_b2 = layer * per_b2;

                        if ( p.size() >= 1 ) copySliceToParam( params.fcprojw, off_w2, per_w2, p[ 0 ] );
                        if ( p.size() >= 2 && !params.fcprojb.empty() ) copySliceToParam( params.fcprojb, off_b2, per_b2, p[ 1 ] );
                    }
                }
                catch ( ... ) {}
            }
        }

        void applyFinalLayerNorm( const ParameterTensors& params )
        {
            if ( !final_layernorm_ )
                return;

            auto p = final_layernorm_->getParameters();

            if ( p.size() >= 1 && !params.lnfw.empty() )
                copyHostVectorToComponentTensor( params.lnfw, p[ 0 ] );

            if ( p.size() >= 2 && !params.lnfb.empty() )
                copyHostVectorToComponentTensor( params.lnfb, p[ 1 ] );
        }

        // -------------------------------------------------------------------

        void validateInputShape( const shape_t& input_shape ) const
        {
            if ( input_shape.size() != 2 )
            {
                throw std::invalid_argument(
                    "GptTransformer: input must have rank 2 (batch_size, seq_length)" );
            }

            if ( input_shape[ 1 ] > config_.max_seq_length )
            {
                throw std::invalid_argument(
                    std::format( "GptTransformer: sequence length {} exceeds maximum {}",
                        input_shape[ 1 ], config_.max_seq_length ) );
            }
        }

        void createGraph()
        {
            EncoderConfig enc_cfg;
            enc_cfg.withVocabularyLength( static_cast<size_t>(config_.vocab_size) )
                .withMaxSequenceLength( static_cast<size_t>(config_.max_seq_length) )
                .withEmbeddingDim( static_cast<size_t>(config_.embedding_dim) );

            enc_cfg.validate();

            auto encoder = std::make_shared<EncoderType>(
                this->getName() + ".encoder", enc_cfg );

            this->addComponent( encoder );

            for ( int64_t i = 0; i < config_.num_layers; ++i )
            {
                TransformerConfig tf_cfg( static_cast<dim_t>( config_.embedding_dim ),
                    static_cast<dim_t>( config_.num_heads ) );

                tf_cfg.withHiddenDimension( static_cast<dim_t>( config_.mlp_hidden_dim ) )
                    .withBias( false )
                    .withActivation( ActivationType::Gelu )
                    .withResidualScale( 1.0f / sqrtf( static_cast<float>( config_.num_layers ) ) );

                auto layer = std::make_shared<TransformerBlockType>(
                    this->getName() + ".tf_layer_" + std::to_string( i ), tf_cfg, std::nullopt );

                this->addComponent( layer );
            }

            auto ln_config = LayerNormConfig()
                .withNormalizedShape( { config_.embedding_dim } );

            auto final_layernorm = std::make_shared<LayerNormType>(
                this->getName() + ".final_layernorm", ln_config, std::nullopt );

            this->addComponent( final_layernorm );

            auto lm_head_config = LinearConfig( config_.embedding_dim, config_.vocab_size )
                .withBias( false );

            auto lm_head = std::make_shared<LinearType>(
                this->getName() + ".lm_head", lm_head_config, std::nullopt );

            this->addComponent( lm_head );
        }

        static void loadComponentWeights( ModelArchive& /*archive*/,
            GptTransformer* /*transformer*/ )
        {}
    };
}