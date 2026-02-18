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
#include <cassert>

export module Bard.Transformer;

//export import :Config;

import Mila;
import Dnn.Network;
import Cuda.Error; // Debugging

// DEPRECATED: This transformer in the Bard sample has been moved to general GPT open source compatible transformer

//namespace Bard
//{
//    using namespace Mila::Dnn;
//    using namespace Mila::Dnn::Compute;
//    using namespace Mila::Dnn::Serialization;
//
//    /**
//     * @brief Character-level transformer for autoregressive language modeling.
//     *
//     * Transformer decoder architecture for next-character prediction:
//     *   Input tokens (B, T) -> Embeddings (B, T, D) -> Transformer Blocks -> Logits (B, T, V)
//     * Where:
//     *   B = batch size
//     *   T = sequence length
//     *   D = embedding dimension
//     *   V = vocabulary size
//     *
//     * Construction Pattern:
//     * 1. Constructor creates and owns ExecutionContext for specified device
//     * 2. Component graph is built via createGraph() (context-independent)
//     * 3. Context is propagated to self and all children via setExecutionContext()
//     * 4. onBuilding() hook builds children with shapes and allocates buffers
//     *
//     * Serialization Contract:
//     * - Implements save_() override to write type identifier and configuration
//     * - Provides static Load() factory method for type-safe deserialization
//     * - Base Network class handles component graph topology serialization
//     *
//     * New compute API:
//     * - Child components return component-owned ITensor* from forward()
//     * - Child components return component-owned ITensor* from backward()
//     * - Network chains those pointers and caches component-owned activation
//     *   pointers required for backward; no longer allocates per-stage activation
//     *   buffers itself.
//     *
//     * @tparam TDeviceType Device type (DeviceType::Cpu or DeviceType::Cuda)
//     * @tparam TPrecision Abstract tensor precision (TensorDataType)
//     */
//    export template<DeviceType TDeviceType, TensorDataType TPrecision>
//        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
//    class CharTransformer : public Network<TDeviceType, TPrecision>
//    {
//    public:
//        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
//        using NetworkBase = Network<TDeviceType, TPrecision>;
//        using TensorType = Tensor<TPrecision, MR>;
//        using LinearType = Linear<TDeviceType, TPrecision>;
//        using LayerNormType = LayerNorm<TDeviceType, TPrecision>;
//        using TransformerBlockType = GptBlock<TDeviceType, TPrecision>;
//        using EncoderType = LearnedEncoder<TDeviceType, dtype_t::INT32, TPrecision>;
//        using TokenIndexType = Tensor<dtype_t::INT32, MR>;
//        using ComponentPtr = typename NetworkBase::ComponentPtr;
//
//        /**
//         * @brief Construct CharTransformer network.
//         *
//         * Follows the concrete network construction pattern:
//         * 1. Create and own ExecutionContext for specified device
//         * 2. Build component graph (context-independent)
//         * 3. Propagate context to self and all children
//         *
//         * @param config Transformer configuration
//         * @param device_id Device identifier for network execution
//         *
//         * @throws std::invalid_argument if config is invalid
//         * @throws std::invalid_argument if device_id.type does not match TDeviceType
//         * @throws std::runtime_error if ExecutionContext creation fails
//         */
//        explicit CharTransformer( const std::string& name, const CharTransformerConfig& config, DeviceId device_id )
//            : NetworkBase( name ), owned_context_( createExecutionContext( device_id ) ), config_( config )
//        {
//            config_.validate();
//
//            if ( device_id.type != TDeviceType )
//            {
//                throw std::invalid_argument(
//                    std::format( "CharTransformer: device type mismatch: expected {}, got {}",
//                        deviceTypeToString( TDeviceType ),
//                        deviceTypeToString( device_id.type ) ) );
//            }
//
//            createGraph();
//
//            this->setExecutionContext( owned_context_.get() );
//        }
//
//        ~CharTransformer() override = default;
//
//        /**
//         * @brief Load CharTransformer from archive.
//         *
//         * Static factory method for type-safe deserialization. Reconstructs
//         * the transformer by:
//         * 1. Reading configuration from archive metadata
//         * 2. Constructing via normal constructor (creates graph + context)
//         * 3. Building with saved input shape
//         * 4. Loading component weights into built components
//         *
//         * @param archive Archive containing serialized transformer
//         * @param device_id Device for execution (may differ from saved device)
//         * @return Unique pointer to reconstructed CharTransformer
//         *
//         * @throws std::runtime_error if archive is malformed
//         * @throws std::runtime_error if configuration is invalid
//         */
//        static std::unique_ptr<CharTransformer> Load( ModelArchive& archive, DeviceId device_id )
//        {
//            auto scope = ModelArchive::ScopedScope( archive, "network" );
//
//            SerializationMetadata meta = archive.readMetadata( "transformer_meta.json" );
//
//            CharTransformerConfig config;
//
//            config.vocab_size = meta.getInt( "vocab_size" );
//            config.max_seq_length = meta.getInt( "max_seq_length" );
//            config.embedding_dim = meta.getInt( "embedding_dim" );
//            config.num_heads = meta.getInt( "num_heads" );
//            config.num_layers = meta.getInt( "num_layers" );
//            config.mlp_hidden_dim = meta.getInt( "mlp_hidden_dim" );
//
//            auto transformer = std::make_unique<CharTransformer>( config, device_id );
//
//            shape_t input_shape = meta.getShape( "input_shape" );
//            transformer->build( input_shape );
//
//            loadComponentWeights( archive, transformer.get() );
//
//            return transformer;
//        }
//
//        // ====================================================================
//        // Compute operation dispatch (new API: component-owned outputs)
//        // ====================================================================
//
//        /**
//         * @brief Forward pass using child components' new forward() API.
//         *
//         * Chains component-owned outputs without allocating intermediate network
//         * activation buffers. Final logits are copied into network-owned output
//         * buffer to preserve the external return semantics.
//         *
//         * @param input Forward input tensor
//         * @return Pointer to network-owned output tensor (logits)
//         */
//        TensorType& forward( const TokenIndexType& input )
//        {
//            if ( !this->isBuilt() )
//            {
//                throw std::runtime_error( "CharTransformer must be built before calling forward." );
//            }
//
//            encoder_out_ptr_ = &encoder_->forward( input );
//            this->getExecutionContext()->synchronize();
//
//            // Set first block input
//            if ( block_input_ptrs_.empty() || block_input_ptrs_.size() != transformer_blocks_.size() )
//            {
//                throw std::runtime_error( "CharTransformer: forward internal state not initialized" );
//            }
//
//            block_input_ptrs_[ 0 ] = encoder_out_ptr_;
//
//            // Transformer blocks (chain component-owned outputs)
//            for ( size_t i = 0; i < transformer_blocks_.size(); ++i )
//            {
//                auto& block_out = transformer_blocks_[ i ]->forward( *block_input_ptrs_[ i ] );
//                this->getExecutionContext()->synchronize();
//
//                block_output_ptrs_[ i ] = &block_out;
//
//                if ( i + 1 < transformer_blocks_.size() )
//                {
//                    block_input_ptrs_[ i + 1 ] = &block_out;
//                }
//            }
//
//            // Final layernorm forward -> normalized (component-owned)
//            normalized_ptr_ = &final_layernorm_->forward( *block_output_ptrs_.back() );
//            this->getExecutionContext()->synchronize();
//
//            // LM head forward -> logits (component-owned)
//            logits_ptr_ = &lm_head_->forward( *normalized_ptr_ );
//            this->getExecutionContext()->synchronize();
//
//            // REVIEW: This doesn't seem right. Why not just pass the logits_ptr_ back to the caller?
//            copy( *dynamic_cast<const TensorType*>( logits_ptr_ ), *owned_output_ );
//            this->getExecutionContext()->synchronize();
//
//            return *owned_output_;
//        }
//
//        /**
//         * @brief Backward pass using child components' new backward() API.
//         *
//         * Chains component backward calls using the forward inputs/outputs that
//         * were cached during the forward() call (component-owned pointers).
//         * Returns the component-owned input-gradient produced by the encoder.
//         *
//         * @param input Original forward input tensor
//         * @param output_grad Gradient w.r.t. transformer output (logits)
//         * @return Reference to component-owned input gradient tensor (token-index typed)
//         */
//        TokenIndexType& backward( const TokenIndexType& input, const TensorType& output_grad )
//        {
//            if ( !this->isBuilt() )
//            {
//                throw std::runtime_error( "CharTransformer must be built before calling backward." );
//            }
//
//            if ( !this->isTraining() )
//            {
//                throw std::runtime_error( "CharTransformer: backward requires training mode (setTraining(true))." );
//            }
//
//            // Validate we have cached forward activation pointers
//            if ( !encoder_out_ptr_ )
//            {
//                throw std::runtime_error( "CharTransformer: forward activations not present for backward. Call forward() before backward()." );
//            }
//
//            for ( size_t i = 0; i < transformer_blocks_.size(); ++i )
//            {
//                if ( !block_input_ptrs_[ i ] || !block_output_ptrs_[ i ] )
//                {
//                    throw std::runtime_error(
//                        std::format( "CharTransformer: missing cached activation for block {}", i ) );
//                }
//            }
//
//            if ( !normalized_ptr_ || !logits_ptr_ )
//            {
//                throw std::runtime_error( "CharTransformer: missing final activations for backward" );
//            }
//
//            // LM head backward -> gradient w.r.t. normalized (component-owned)
//            auto& normalized_grad_ptr = lm_head_->backward( *normalized_ptr_, output_grad );
//            this->getExecutionContext()->synchronize();
//
//            // Final layernorm backward -> gradient w.r.t. last block output (component-owned)
//            auto& last_block_grad_ptr = final_layernorm_->backward( *block_output_ptrs_.back(), normalized_grad_ptr );
//            this->getExecutionContext()->synchronize();
//
//            // Backprop through transformer blocks in reverse order
//            TensorType* curr_grad = &last_block_grad_ptr;
//
//            // Then update the assignment inside the loop:
//            for ( int64_t i = static_cast<int64_t>(transformer_blocks_.size()) - 1; i >= 0; --i )
//            {
//                auto& block_grad_ptr = transformer_blocks_[ static_cast<size_t>(i) ]->backward(
//                    *block_input_ptrs_[ static_cast<size_t>(i) ], *curr_grad );
//
//                curr_grad = &block_grad_ptr;
//
//                this->getExecutionContext()->synchronize();
//            }
//
//            // Encoder backward -> gradient w.r.t. input (component-owned)
//            auto& input_grad_ptr = encoder_->backward( input, *curr_grad );
//            this->getExecutionContext()->synchronize();
//
//            return input_grad_ptr;
//        }
//
//        void zeroGradients() override
//        {
//            // If not built, nothing to clear
//            if ( !this->isBuilt() )
//            {
//                return;
//            }
//
//            // Zero gradients in child components only; intermediate activation
//            // buffers and grads are owned and managed by components now.
//            encoder_->zeroGradients();
//
//            for ( auto& block : transformer_blocks_ )
//            {
//                block->zeroGradients();
//            }
//
//            final_layernorm_->zeroGradients();
//            lm_head_->zeroGradients();
//        }
//
//        std::vector<ITensor*> getGradientsForDebug() {
//            return this->getGradients();
//        }
//
//        std::vector<ITensor*> getParametersForDebug() {
//            return this->getParameters();
//        }
//
//        // ====================================================================
//        // Component interface
//        // ====================================================================
//
//        std::string toString() const override
//        {
//            std::ostringstream oss;
//            oss << std::endl;
//            oss << "CharTransformer: " << this->getName() << std::endl;
//            oss << "Device: " << this->getDeviceId().toString() << std::endl;
//
//            oss << "Architecture:" << std::endl;
//            oss << "  Vocabulary: " << config_.vocab_size << " tokens" << std::endl;
//            oss << "  Max sequence length: " << config_.max_seq_length << std::endl;
//            oss << "  Embedding dimension: " << config_.embedding_dim << std::endl;
//            oss << "  Number of heads: " << config_.num_heads << std::endl;
//            oss << "  Number of layers: " << config_.num_layers << std::endl;
//            oss << "  MLP hidden dimension: " << config_.mlp_hidden_dim << std::endl;
//
//            if ( this->isBuilt() )
//            {
//                oss << "  Parameters: " << this->parameterCount() << std::endl;
//                oss << "  Batch size: " << batch_size_ << std::endl;
//                oss << "  Sequence length: " << seq_length_ << std::endl;
//
//                oss << "  Input shape: ("; for ( size_t i = 0; i < input_shape_.size(); ++i ) {
//                    oss << input_shape_[ i ]; if ( i != input_shape_.size() - 1 ) oss << ", ";
//                } oss << ")" << std::endl;
//
//                oss << "  Output shape: ("; for ( size_t i = 0; i < output_shape_.size(); ++i ) {
//                    oss << output_shape_[ i ]; if ( i != output_shape_.size() - 1 ) oss << ", ";
//                } oss << ")" << std::endl;
//            }
//
//            oss << "  Sub-Modules:" << std::endl;
//
//            if ( encoder_ )
//            {
//                oss << "    - encoder: " << encoder_->getName() << std::endl;
//            }
//
//            oss << "    - transformer_blocks: " << transformer_blocks_.size() << " layers" << std::endl;
//
//            if ( final_layernorm_ )
//            {
//                oss << "    - final_layernorm: " << final_layernorm_->getName() << std::endl;
//            }
//
//            if ( lm_head_ )
//            {
//                oss << "    - lm_head: " << lm_head_->getName() << std::endl;
//            }
//
//            oss << std::endl;
//
//            return oss.str();
//        }
//
//        IExecutionContext* getExecutionContext() const
//        {
//            return NetworkBase::getExecutionContext();
//        }
//
//    protected:
//
//        /**
//         * @brief Save transformer-specific configuration (required by Network base).
//         *
//         * Implements the serialization contract by writing type identifier
//         * and configuration metadata to enable reconstruction via Load().
//         *
//         * @param archive Archive to write to
//         * @param mode Serialization mode (passed from Network::save())
//         */
//        void save_( ModelArchive& archive, SerializationMode /*mode*/ ) const override
//        {
//            SerializationMetadata meta;
//            meta.set( "type", "CharTransformer" )
//                .set( "version", int64_t( 1 ) )
//                .set( "name", this->getName() )
//                .set( "vocab_size", config_.vocab_size )
//                .set( "max_seq_length", config_.max_seq_length )
//                .set( "embedding_dim", config_.embedding_dim )
//                .set( "num_heads", config_.num_heads )
//                .set( "num_layers", config_.num_layers )
//                .set( "mlp_hidden_dim", config_.mlp_hidden_dim );
//
//            if ( this->isBuilt() )
//            {
//                meta.set( "input_shape", input_shape_ )
//                    .set( "embedding_shape", embedding_shape_ )
//                    .set( "output_shape", output_shape_ )
//                    .set( "batch_size", batch_size_ )
//                    .set( "seq_length", seq_length_ );
//            }
//
//            archive.writeMetadata( "transformer_meta.json", meta );
//        }
//
//        /**
//         * @brief Called when training mode changes.
//         *
//         * Previously this class allocated activation buffers for backward.
//         * With the new component-owned forward/backward API the network no
//         * longer needs to allocate per-stage activations; simply propagate the
//         * change to the base class.
//         */
//        void onTrainingChanging( bool is_training ) override
//        {
//            NetworkBase::onTrainingChanging( is_training );
//        }
//
//        /**
//         * @brief Hook invoked during build() to initialize transformer with input shape.
//         *
//         * Validates input shape, computes per-layer shapes, caches typed pointers to children,
//         * builds all child components with appropriate shapes, and allocates network-owned
//         * output buffer. Components now own intermediate activations.
//         *
//         * All children have ExecutionContext at this point (propagated in constructor).
//         *
//         * @param input_shape Expected input tensor shape
//         *
//         * @throws std::invalid_argument if input_shape is invalid for transformer
//         */
//        void onBuilding( const shape_t& input_shape ) override
//        {
//            validateInputShape( input_shape );
//
//            input_shape_ = input_shape;
//            batch_size_ = input_shape[ 0 ];
//            seq_length_ = input_shape[ 1 ];
//
//            embedding_shape_ = { batch_size_, seq_length_, config_.embedding_dim };
//            output_shape_ = { batch_size_, seq_length_, config_.vocab_size };
//
//            encoder_ = this->template getComponentAs<EncoderType>( this->getName() + ".lenc" );
//            encoder_->build( input_shape );
//
//            for ( int64_t i = 0; i < config_.num_layers; ++i )
//            {
//                std::string block_name = this->getName() + ".tf.layer_" + std::to_string( i );
//                auto block = this->template getComponentAs<TransformerBlockType>( block_name );
//                block->build( embedding_shape_ );
//                transformer_blocks_.push_back( block );
//            }
//
//            final_layernorm_ = this->template getComponentAs<LayerNormType>( this->getName() + ".ln_final" );
//            final_layernorm_->build( embedding_shape_ );
//
//            lm_head_ = this->template getComponentAs<LinearType>( this->getName() + ".lm_head" );
//            lm_head_->build( embedding_shape_ );
//
//            auto device_id = this->getDeviceId();
//
//            // Allocate network-owned output buffer (logits)
//            owned_output_ = std::make_shared<TensorType>( device_id, output_shape_ );
//            owned_output_->setName( this->getName() + ".output" );
//
//            // Prepare vectors to cache component-owned activation pointers during forward
//            block_input_ptrs_.assign( transformer_blocks_.size(), nullptr );
//            block_output_ptrs_.assign( transformer_blocks_.size(), nullptr );
//
//            // Clear cached pointers
//            encoder_out_ptr_ = nullptr;
//            normalized_ptr_ = nullptr;
//            logits_ptr_ = nullptr;
//        }
//
//    private:
//
//        std::unique_ptr<IExecutionContext> owned_context_{ nullptr };
//
//        CharTransformerConfig config_;
//
//        shape_t input_shape_;
//        shape_t embedding_shape_;
//        shape_t output_shape_;
//        int64_t batch_size_{ 0 };
//        int64_t seq_length_{ 0 };
//
//        std::shared_ptr<EncoderType> encoder_{ nullptr };
//        std::vector<std::shared_ptr<TransformerBlockType>> transformer_blocks_;
//        std::shared_ptr<LayerNormType> final_layernorm_{ nullptr };
//        std::shared_ptr<LinearType> lm_head_{ nullptr };
//
//        // Network-owned forward output buffer (logits)
//        std::shared_ptr<TensorType> owned_output_{ nullptr };
//
//        // Component-owned activation pointers cached during forward (no network allocation)
//        TensorType* encoder_out_ptr_{ nullptr };
//        std::vector<TensorType*> block_input_ptrs_;
//        std::vector<TensorType*> block_output_ptrs_;
//        
//        TensorType* normalized_ptr_{ nullptr };
//        TensorType* logits_ptr_{ nullptr };
//
//        /**
//         * @brief Validate input shape for CharTransformer.
//         *
//         * @throws std::invalid_argument if shape is invalid
//         */
//        void validateInputShape( const shape_t& input_shape ) const
//        {
//            if ( input_shape.size() != 2 )
//            {
//                throw std::invalid_argument(
//                    "CharTransformer: input must have rank 2 (batch_size, seq_length)" );
//            }
//
//            if ( input_shape[ 1 ] > config_.max_seq_length )
//            {
//                throw std::invalid_argument(
//                    std::format( "CharTransformer: sequence length {} exceeds maximum {}",
//                        input_shape[ 1 ], config_.max_seq_length ) );
//            }
//        }
//
//        /**
//         * @brief Create the CharTransformer network graph (context-independent).
//         *
//         * Defines the computational graph:
//         *   encoder -> transformer_blocks -> final_layernorm -> lm_head
//         *
//         * Components are created without ExecutionContext (shared mode).
//         * Context will be propagated after this method returns via setExecutionContext().
//         */
//        void createGraph()
//        {
//            LearnedEncoderConfig enc_cfg;
//            enc_cfg.withVocabularyLength( static_cast<size_t>(config_.vocab_size) )
//                .withMaxSequenceLength( static_cast<size_t>(config_.max_seq_length) )
//                .withEmbeddingDim( static_cast<size_t>(config_.embedding_dim) );
//
//            enc_cfg.validate();
//
//            auto encoder = std::make_shared<EncoderType>(
//                this->getName() + ".lenc", enc_cfg );
//
//            this->addComponent( encoder );
//
//            for ( int64_t i = 0; i < config_.num_layers; ++i )
//            {
//                GptBlockConfig tf_cfg( static_cast<dim_t>( config_.embedding_dim ),
//                    static_cast<dim_t>( config_.num_heads ) );
//
//                tf_cfg.withHiddenSize( static_cast<dim_t>( config_.mlp_hidden_dim ) )
//                    .withBias( false )
//                    .withActivation( ActivationType::Gelu )
//                    .withResidualScale( 1.0f / sqrtf( static_cast<float>( config_.num_layers ) ) );
//
//                // REVIEW: Is the name here ".tf_layer_"
//                auto layer = std::make_shared<TransformerBlockType>(
//                    this->getName() + ".tf.layer_" + std::to_string( i ), tf_cfg, std::nullopt );
//
//                this->addComponent( layer );
//            }
//
//            auto ln_config = LayerNormConfig()
//                .withNormalizedShape( { config_.embedding_dim } );
//
//            auto final_layernorm = std::make_shared<LayerNormType>(
//                this->getName() + ".ln_final", ln_config, std::nullopt );
//
//            this->addComponent( final_layernorm );
//
//            auto lm_head_config = LinearConfig( config_.embedding_dim, config_.vocab_size )
//                .withBias( false );
//
//            auto lm_head = std::make_shared<LinearType>(
//                this->getName() + ".lm_head", lm_head_config, std::nullopt );
//
//            this->addComponent( lm_head );
//        }
//
//        /**
//         * @brief Load component weights from archive.
//         *
//         * Helper method for Load() to populate weights into already-built components.
//         * Base Network class handles component graph traversal; this method loads
//         * weights into each component.
//         *
//         * @param archive Archive containing serialized weights
//         * @param transformer Transformer instance with built components
//         */
//        static void loadComponentWeights( ModelArchive& /*archive*/,
//            CharTransformer* /*transformer*/ )
//        {}
//    };
//}