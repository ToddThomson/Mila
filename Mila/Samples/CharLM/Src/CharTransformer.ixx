/**
 * @file CharTransformer.ixx
 * @brief Character-level transformer for language modeling.
 *
 * Device-templated network implementing a transformer architecture
 * for character-level next-token prediction.
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

export module CharLM.Transformer;

import Mila;
import Dnn.Network;

namespace Mila::CharLM
{
    using namespace Mila::Dnn;
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief Configuration for CharTransformer model.
     */
    export struct CharTransformerConfig
    {
        int64_t vocab_size = 256;
        int64_t max_seq_length = 256;
        int64_t embedding_dim = 256;
        int64_t num_heads = 4;
        int64_t num_layers = 4;
        int64_t mlp_hidden_dim = 1024;

        void validate() const
        {
            if ( vocab_size <= 0 )
                throw std::invalid_argument( "vocab_size must be positive" );

            if ( max_seq_length <= 0 )
                throw std::invalid_argument( "max_seq_length must be positive" );

            if ( embedding_dim <= 0 )
                throw std::invalid_argument( "embedding_dim must be positive" );

            if ( num_heads <= 0 )
                throw std::invalid_argument( "num_heads must be positive" );

            if ( embedding_dim % num_heads != 0 )
                throw std::invalid_argument( "embedding_dim must be divisible by num_heads" );

            if ( num_layers <= 0 )
                throw std::invalid_argument( "num_layers must be positive" );

            if ( mlp_hidden_dim <= 0 )
                throw std::invalid_argument( "mlp_hidden_dim must be positive" );
        }
    };

    /**
     * @brief Character-level transformer for autoregressive language modeling.
     *
     * Transformer decoder architecture for next-character prediction:
     *   Input tokens (B, T) -> Embeddings (B, T, D) -> Transformer Blocks -> Logits (B, T, V)
     *
     * Construction Pattern:
     * 1. Constructor creates and owns ExecutionContext
     * 2. Component graph is built via createGraph() (context-independent)
     * 3. Context is propagated to self and all children via setExecutionContext()
     * 4. onBuilding() hook builds children with shapes and allocates buffers
     *
     * Serialization Contract:
     * - Implements save_() override to write type identifier and configuration
     * - Provides static Load() factory method for type-safe deserialization
     * - Base Network class handles component graph topology serialization
     *
     * @tparam TDeviceType Device type (DeviceType::Cpu or DeviceType::Cuda)
     * @tparam TPrecision Abstract tensor precision (TensorDataType)
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class CharTransformer : public Network<TDeviceType, TPrecision>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using NetworkBase = Network<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using LinearType = Linear<TDeviceType, TPrecision>;
        using LayerNormType = LayerNorm<TDeviceType, TPrecision>;
        using TransformerBlockType = Transformer<TDeviceType, TPrecision>;
        using EncoderType = Encoder<TDeviceType, dtype_t::INT32, TPrecision>;
        using ComponentPtr = typename NetworkBase::ComponentPtr;

        /**
         * @brief Construct CharTransformer network.
         *
         * Follows the concrete network construction pattern:
         * 1. Create and own ExecutionContext for specified device
         * 2. Build component graph (context-independent)
         * 3. Propagate context to self and all children
         *
         * @param config Transformer configuration
         * @param device_id Device identifier for network execution
         *
         * @throws std::invalid_argument if config is invalid
         * @throws std::invalid_argument if device_id.type does not match TDeviceType
         * @throws std::runtime_error if ExecutionContext creation fails
         */
        explicit CharTransformer( const std::string& name, const CharTransformerConfig& config, DeviceId device_id )
            : NetworkBase( name ), owned_context_( createExecutionContext( device_id ) ), config_( config )
        {
            config_.validate();

            if ( device_id.type != TDeviceType )
            {
                throw std::invalid_argument(
                    std::format( "CharTransformer: device type mismatch: expected {}, got {}",
                        deviceTypeToString( TDeviceType ),
                        deviceTypeToString( device_id.type ) ) );
            }

            createGraph();

            this->setExecutionContext( owned_context_.get() );
        }

        ~CharTransformer() override = default;

        /**
         * @brief Load CharTransformer from archive.
         *
         * Static factory method for type-safe deserialization. Reconstructs
         * the transformer by:
         * 1. Reading configuration from archive metadata
         * 2. Constructing via normal constructor (creates graph + context)
         * 3. Building with saved input shape
         * 4. Loading component weights into built components
         *
         * @param archive Archive containing serialized transformer
         * @param device_id Device for execution (may differ from saved device)
         * @return Unique pointer to reconstructed CharTransformer
         *
         * @throws std::runtime_error if archive is malformed
         * @throws std::runtime_error if configuration is invalid
         */
        static std::unique_ptr<CharTransformer> Load( ModelArchive& archive, DeviceId device_id )
        {
            auto scope = ModelArchive::ScopedScope( archive, "network" );

            SerializationMetadata meta = archive.readMetadata( "transformer_meta.json" );

            CharTransformerConfig config;

            config.vocab_size = meta.getInt( "vocab_size" );
            config.max_seq_length = meta.getInt( "max_seq_length" );
            config.embedding_dim = meta.getInt( "embedding_dim" );
            config.num_heads = meta.getInt( "num_heads" );
            config.num_layers = meta.getInt( "num_layers" );
            config.mlp_hidden_dim = meta.getInt( "mlp_hidden_dim" );

            auto transformer = std::make_unique<CharTransformer>( config, device_id );

            shape_t input_shape = meta.getShape( "input_shape" );
            transformer->build( input_shape );

            loadComponentWeights( archive, transformer.get() );

            return transformer;
        }

        // ====================================================================
        // Compute operation dispatch
        // ====================================================================

        /**
         * @brief Forward pass - HOT PATH, pure dispatch to child components.
         *
         * All setup and validation was done in onBuilding(). This method chains
         * forward calls through the transformer structure using pre-allocated buffers.
         *
         * @param input Input tensor containing token indices (batch_size, seq_length)
         * @param output Output tensor for next-token logits (batch_size, seq_length, vocab_size)
         *
         * @throws std::runtime_error if transformer has not been built
         */
        void forward( const ITensor& input, ITensor& output )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "CharTransformer must be built before calling forward." );
            }

            encoder_->forward( input, *embedded_ );

            if ( transformer_blocks_.empty() )
            {
                copy( *embedded_, *transformer_output_ );
            }
            else
            {
                ITensor* read = embedded_.get();
                ITensor* write = transformer_output_.get();

                for ( size_t i = 0; i < transformer_blocks_.size(); ++i )
                {
                    transformer_blocks_[ i ]->forward( *read, *write );
                    transformer_blocks_[ i ]->synchronize();

                    std::swap( read, write );
                }
            }

            final_layernorm_->forward( *transformer_output_, *normalized_ );
            lm_head_->forward( *normalized_, output );
        }

        /**
         * @brief Backward pass - HOT PATH, pure dispatch to child components.
         *
         * Chains backward calls through the transformer structure in reverse order.
         *
         * @param input Original forward input tensor
         * @param output_grad Gradient w.r.t. transformer output
         * @param input_grad Gradient w.r.t. transformer input (output)
         *
         * @throws std::runtime_error if transformer has not been built
         */
        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "CharTransformer must be built before calling backward." );
            }

            auto device = this->getDeviceId();

            TensorType normalized_grad( device, cached_embedding_shape_ );
            zeros( normalized_grad );
            lm_head_->backward( *normalized_, output_grad, normalized_grad );

            TensorType transformer_output_grad( device, cached_embedding_shape_ );
            zeros( transformer_output_grad );
            final_layernorm_->backward( *transformer_output_, normalized_grad, transformer_output_grad );

            TensorType& embedding_grad = transformer_output_grad;

            encoder_->backward( input, embedding_grad );

            if ( auto* in_t = dynamic_cast<TensorType*>(&input_grad) )
            {
                zeros( *in_t );
            }
        }

        // ====================================================================
        // Component interface
        // ====================================================================

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << std::endl;
            oss << "CharTransformer: " << this->getName() << std::endl;
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

                oss << "  Input shape: (";
                for ( size_t i = 0; i < cached_input_shape_.size(); ++i )
                {
                    oss << cached_input_shape_[ i ];
                    if ( i != cached_input_shape_.size() - 1 )
                        oss << ", ";
                }
                oss << ")" << std::endl;

                oss << "  Output shape: (";
                for ( size_t i = 0; i < cached_output_shape_.size(); ++i )
                {
                    oss << cached_output_shape_[ i ];
                    if ( i != cached_output_shape_.size() - 1 )
                        oss << ", ";
                }
                oss << ")" << std::endl;
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

        /*const CharTransformerConfig& getConfig() const
        {
            return config_;
        }*/

        /*IExecutionContext* getExecutionContext() const
        {
            return NetworkBase::getExecutionContext();
        }*/

        // ====================================================================
        // Child module accessors
        // ====================================================================

        /*std::shared_ptr<EncoderType> getEncoder() const noexcept
        {
            return encoder_;
        }

        std::shared_ptr<LayerNormType> getFinalLayerNorm() const noexcept
        {
            return final_layernorm_;
        }

        std::shared_ptr<LinearType> getLMHead() const noexcept
        {
            return lm_head_;
        }

        std::vector<std::shared_ptr<TransformerBlockType>> getTransformerBlocks() const noexcept
        {
            return transformer_blocks_;
        }*/

    protected:

        /**
         * @brief Save transformer-specific configuration (required by Network base).
         *
         * Implements the serialization contract by writing type identifier
         * and configuration metadata to enable reconstruction via Load().
         *
         * @param archive Archive to write to
         * @param mode Serialization mode (passed from Network::save())
         */
        void save_( ModelArchive& archive, SerializationMode /*mode*/ ) const override
        {
            SerializationMetadata meta;
            meta.set( "type", "CharTransformer" )
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
                meta.set( "input_shape", cached_input_shape_ )
                    .set( "embedding_shape", cached_embedding_shape_ )
                    .set( "output_shape", cached_output_shape_ )
                    .set( "batch_size", batch_size_ )
                    .set( "seq_length", seq_length_ );
            }

            archive.writeMetadata( "transformer_meta.json", meta );
        }

        /**
         * @brief Hook invoked during build() to initialize transformer with input shape.
         *
         * Validates input shape, computes per-layer shapes, caches typed pointers to children,
         * builds all child components with appropriate shapes, and allocates intermediate buffers.
         *
         * All children have ExecutionContext at this point (propagated in constructor).
         *
         * @param input_shape Expected input tensor shape
         *
         * @throws std::invalid_argument if input_shape is invalid for transformer
         */
        void onBuilding( const shape_t& input_shape ) override
        {
            validateInputShape( input_shape );

            cached_input_shape_ = input_shape;
            batch_size_ = input_shape[ 0 ];
            seq_length_ = input_shape[ 1 ];

            cached_embedding_shape_ = { batch_size_, seq_length_, config_.embedding_dim };
            cached_output_shape_ = { batch_size_, seq_length_, config_.vocab_size };

            encoder_ = this->template getComponentAs<EncoderType>( this->getName() + ".encoder" );
            encoder_->build( input_shape );

            for ( int64_t i = 0; i < config_.num_layers; ++i )
            {
                std::string block_name = this->getName() + ".layer" + std::to_string( i );
                auto block = this->template getComponentAs<TransformerBlockType>( block_name );
                block->build( cached_embedding_shape_ );
                transformer_blocks_.push_back( block );
            }

            final_layernorm_ = this->template getComponentAs<LayerNormType>( this->getName() + ".final_layernorm" );
            final_layernorm_->build( cached_embedding_shape_ );

            lm_head_ = this->template getComponentAs<LinearType>( this->getName() + ".lm_head" );
            lm_head_->build( cached_embedding_shape_ );

            auto device = this->getDeviceId();

            embedded_ = std::make_shared<TensorType>( device, cached_embedding_shape_ );
            embedded_->setName( this->getName() + ".embedded" );

            transformer_output_ = std::make_shared<TensorType>( device, cached_embedding_shape_ );
            transformer_output_->setName( this->getName() + ".transformer_output" );

            normalized_ = std::make_shared<TensorType>( device, cached_embedding_shape_ );
            normalized_->setName( this->getName() + ".normalized" );
        }

    private:

        std::unique_ptr<IExecutionContext> owned_context_{ nullptr };

        CharTransformerConfig config_;

        shape_t cached_input_shape_;
        shape_t cached_embedding_shape_;
        shape_t cached_output_shape_;
        int64_t batch_size_{ 0 };
        int64_t seq_length_{ 0 };

        std::shared_ptr<EncoderType> encoder_{ nullptr };
        std::vector<std::shared_ptr<TransformerBlockType>> transformer_blocks_;
        std::shared_ptr<LayerNormType> final_layernorm_{ nullptr };
        std::shared_ptr<LinearType> lm_head_{ nullptr };

        std::shared_ptr<TensorType> embedded_{ nullptr };
        std::shared_ptr<TensorType> transformer_output_{ nullptr };
        std::shared_ptr<TensorType> normalized_{ nullptr };

        /**
         * @brief Validate input shape for CharTransformer.
         *
         * @throws std::invalid_argument if shape is invalid
         */
        void validateInputShape( const shape_t& input_shape ) const
        {
            if ( input_shape.size() != 2 )
            {
                throw std::invalid_argument(
                    "CharTransformer: input must have rank 2 (batch_size, seq_length)" );
            }

            if ( input_shape[ 1 ] > config_.max_seq_length )
            {
                throw std::invalid_argument(
                    std::format( "CharTransformer: sequence length {} exceeds maximum {}",
                        input_shape[ 1 ], config_.max_seq_length ) );
            }
        }

        /**
         * @brief Create the CharTransformer network graph (context-independent).
         *
         * Defines the computational graph:
         *   encoder -> transformer_blocks -> final_layernorm -> lm_head
         *
         * Components are created without ExecutionContext (shared mode).
         * Context will be propagated after this method returns via setExecutionContext().
         */
        void createGraph()
        {
            EncoderConfig enc_cfg;
            enc_cfg.withVocabularyLength( static_cast<size_t>(config_.vocab_size) )
                .withMaxSequenceLength( static_cast<size_t>(config_.max_seq_length) )
                .withChannels( static_cast<size_t>(config_.embedding_dim) );
            
            enc_cfg.validate();

            auto encoder = std::make_shared<EncoderType>(
                this->getName() + ".encoder", enc_cfg, std::nullopt );

            this->addComponent( encoder );

            for ( int64_t i = 0; i < config_.num_layers; ++i )
            {
                TransformerConfig tcfg( static_cast<dim_t>( config_.embedding_dim ),
                    static_cast<dim_t>( config_.num_heads ) );

                tcfg.withHiddenDimension( static_cast<dim_t>( config_.mlp_hidden_dim ) );
                tcfg.withBias( false );
                tcfg.withActivation( ActivationType::Gelu );

                auto block = std::make_shared<TransformerBlockType>(
                    this->getName() + ".layer" + std::to_string( i ), tcfg, std::nullopt );

                this->addComponent( block );
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

        /**
         * @brief Load component weights from archive.
         *
         * Helper method for Load() to populate weights into already-built components.
         * Base Network class handles component graph traversal; this method loads
         * weights into each component.
         *
         * @param archive Archive containing serialized weights
         * @param transformer Transformer instance with built components
         */
        static void loadComponentWeights( ModelArchive& /*archive*/,
            CharTransformer* /*transformer*/ )
        {}
    };

    /*export template<TensorDataType TPrecision>
        using CpuCharTransformer = CharTransformer<DeviceType::Cpu, TPrecision>;

    export template<TensorDataType TPrecision>
        using CudaCharTransformer = CharTransformer<DeviceType::Cuda, TPrecision>;*/
}