/**
 * @file CharTransformer.ixx
 * @brief Character-level transformer for language modeling.
 *
 * Device-templated composite module implementing a transformer architecture
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


export module CharLM.Transformer;

import Mila;

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
        int64_t vocab_size = 256;          // Number of unique characters (default: ASCII extended)
        int64_t max_seq_length = 256;      // Maximum sequence length for positional encoding
        int64_t embedding_dim = 256;       // Dimension of token embeddings (d_model)
        int64_t num_heads = 4;             // Number of attention heads
        int64_t num_layers = 4;            // Number of transformer blocks
        int64_t mlp_hidden_dim = 1024;     // Hidden dimension in MLP (typically 4 * embedding_dim)

        std::string name = "CharTransformer";

        void validate() const
        {
            if (vocab_size <= 0)
                throw std::invalid_argument( "vocab_size must be positive" );
            if (max_seq_length <= 0)
                throw std::invalid_argument( "max_seq_length must be positive" );
            if (embedding_dim <= 0)
                throw std::invalid_argument( "embedding_dim must be positive" );
            if (num_heads <= 0)
                throw std::invalid_argument( "num_heads must be positive" );
            if (embedding_dim % num_heads != 0)
                throw std::invalid_argument( "embedding_dim must be divisible by num_heads" );
            if (num_layers <= 0)
                throw std::invalid_argument( "num_layers must be positive" );
            if (mlp_hidden_dim <= 0)
                throw std::invalid_argument( "mlp_hidden_dim must be positive" );
        }
    };

    /**
     * @brief Character-level transformer for autoregressive language modeling.
     *
     * Transformer decoder architecture for next-character prediction:
     *   Input tokens (B, T) -> Embeddings (B, T, D) -> Transformer Blocks -> Logits (B, T, V)
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class CharTransformer : public CompositeComponent<TDeviceType, TPrecision>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using CompositeComponentBase = CompositeComponent<TDeviceType, TPrecision>;
        using ExecutionContextType = ExecutionContext<TDeviceType>;
        using TensorType = Tensor<TPrecision, MR>;
        using LinearType = Linear<TDeviceType, TPrecision>;
        using LayerNormType = LayerNorm<TDeviceType, TPrecision>;
        using TransformerBlockType = Transformer<TDeviceType, TPrecision>;
        using EncoderType = Encoder<TDeviceType, dtype_t::INT32, TPrecision>;

        explicit CharTransformer(
            std::shared_ptr<ExecutionContextType> exec_context,
            const CharTransformerConfig& config )
            : exec_context_( exec_context ), config_( config )
        {
            if (!exec_context_)
            {
                throw std::invalid_argument( "ExecutionContext cannot be null." );
            }

            config_.validate();

            createComponents();
        }

        ~CharTransformer() override = default;

        // ====================================================================
        // Lifecycle
        // ====================================================================

        /*bool isBuilt() const override
        {
            bool blocks_ok = true;

            if (!transformer_blocks_.empty())
            {
                for (const auto &b : transformer_blocks_)
                {
                    if (!b) { blocks_ok = false; break; }
                }
            }

            return CompositeComponentBase::isBuilt() && encoder_ && final_layernorm_ && lm_head_ && blocks_ok;
        }*/

        // Architecture-specific build hook called by CompositeComponent::build()
        void onBuilding( const shape_t& input_shape ) override
        {
            validateInputShape( input_shape );

            cached_input_shape_ = input_shape;
            batch_size_ = input_shape[0];
            seq_length_ = input_shape[1];

            cached_embedding_shape_ = { batch_size_, seq_length_, config_.embedding_dim };
            cached_output_shape_ = { batch_size_, seq_length_, config_.vocab_size };

            // Build encoder (token + positional embeddings)
            encoder_->build( input_shape );

            // Build transformer blocks (use embedding-shaped inputs)
            for (auto& block : transformer_blocks_)
            {
                block->build( cached_embedding_shape_ );
            }

            final_layernorm_->build( cached_embedding_shape_ );
            lm_head_->build( cached_embedding_shape_ );

            auto device = exec_context_->getDevice();

            embedded_ = std::make_shared<TensorType>( device, cached_embedding_shape_ );
            embedded_->setName( config_.name + ".embedded" );

            // allocate a second buffer to ping-pong between transformer layers
            transformer_output_ = std::make_shared<TensorType>( device, cached_embedding_shape_ );
            transformer_output_->setName( config_.name + ".transformer_output" );

            normalized_ = std::make_shared<TensorType>( device, cached_embedding_shape_ );
            normalized_->setName( config_.name + ".normalized" );
        }

        // ====================================================================
        // Compute operation dispatch
        // ====================================================================

        void forward( const ITensor& input, ITensor& output )
        {
            if (!this->isBuilt())
            {
                throw std::runtime_error( "CharTransformer must be built before calling forward." );
            }

            encoder_->forward( input, *embedded_ );

            if (transformer_blocks_.empty())
            {
                copy( *embedded_, *transformer_output_ );
            }

            else
            {
                ITensor* read = embedded_.get();
                ITensor* write = transformer_output_.get();


                std::cout << "# of blocks to forward through: " << transformer_blocks_.size();

                for (size_t i = 0; i < transformer_blocks_.size(); ++i)
                {
                    transformer_blocks_[i]->forward( *read, *write );

                    transformer_blocks_[i]->synchronize();

                    std::swap( read, write );
                }

                // final buffer already in place; avoid unnecessary copies
            }

            final_layernorm_->forward( *transformer_output_, *normalized_ );
            lm_head_->forward( *normalized_, output );
        }

        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad )
        {
            if (!this->isBuilt())
            {
                throw std::runtime_error( "CharTransformer must be built before calling backward." );
            }

            auto device = exec_context_->getDevice();

            TensorType normalized_grad( device, cached_embedding_shape_ );
            zeros( normalized_grad );
            lm_head_->backward( *normalized_, output_grad, normalized_grad );

            TensorType transformer_output_grad( device, cached_embedding_shape_ );
            zeros( transformer_output_grad );
            final_layernorm_->backward( *transformer_output_, normalized_grad, transformer_output_grad );

            // TODO: Backprop through transformer blocks in reverse order.
            TensorType& embedding_grad = transformer_output_grad;

            encoder_->backward( input, embedding_grad );

            if (auto* in_t = dynamic_cast<TensorType*>(&input_grad))
            {
                zeros( *in_t );
            }
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        void save( ModelArchive& archive ) const
        {
            encoder_->save( archive );

            for (const auto& block : transformer_blocks_)
            {
                block->save( archive );
            }

            final_layernorm_->save( archive );
            lm_head_->save( archive );
        }

        void load( ModelArchive& archive )
        {
            encoder_->load( archive );

            for (auto& block : transformer_blocks_)
            {
                block->load( archive );
            }

            final_layernorm_->load( archive );
            lm_head_->load( archive );
        }

        // ====================================================================
        // Component interface
        // ====================================================================

        std::string getName() const override
        {
            return config_.name;
        }

        std::shared_ptr<ComputeDevice> getDevice() const override
        {
            return exec_context_->getDevice();
        }

        void synchronize() override
        {
            if (exec_context_)
            {
                exec_context_->synchronize();
            }

            encoder_->synchronize();

            for (auto& block : transformer_blocks_)
            {
                block->synchronize();
            }

            final_layernorm_->synchronize();
            lm_head_->synchronize();
        }

        size_t parameterCount() const override
        {
            size_t total = 0;

            total += encoder_->parameterCount();

            for (const auto& block : transformer_blocks_)
            {
                total += block->parameterCount();
            }

            total += final_layernorm_->parameterCount();
            total += lm_head_->parameterCount();

            return total;
        }

        std::vector<ITensor*> getParameters() const override
        {
            std::vector<ITensor*> params;

            auto enc_params = encoder_->getParameters();
            params.insert( params.end(), enc_params.begin(), enc_params.end() );

            for (const auto& block : transformer_blocks_)
            {
                auto block_params = block->getParameters();
                params.insert( params.end(), block_params.begin(), block_params.end() );
            }

            auto ln_params = final_layernorm_->getParameters();
            params.insert( params.end(), ln_params.begin(), ln_params.end() );

            auto lm_params = lm_head_->getParameters();
            params.insert( params.end(), lm_params.begin(), lm_params.end() );

            return params;
        }

        std::vector<ITensor*> getGradients() const override
        {
            std::vector<ITensor*> grads;

            if (this->isTraining())
            {
                auto enc_grads = encoder_->getGradients();
                grads.insert( grads.end(), enc_grads.begin(), enc_grads.end() );
            }

            for (const auto& block : transformer_blocks_)
            {
                auto block_grads = block->getGradients();
                grads.insert( grads.end(), block_grads.begin(), block_grads.end() );
            }

            auto ln_grads = final_layernorm_->getGradients();
            grads.insert( grads.end(), ln_grads.begin(), ln_grads.end() );

            auto lm_grads = lm_head_->getGradients();
            grads.insert( grads.end(), lm_grads.begin(), lm_grads.end() );

            return grads;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << std::endl;
            oss << "CharTransformer: " << config_.name << std::endl;
            oss << "Architecture:" << std::endl;
            oss << "  Vocabulary: " << config_.vocab_size << " tokens" << std::endl;
            oss << "  Max sequence length: " << config_.max_seq_length << std::endl;
            oss << "  Embedding dimension: " << config_.embedding_dim << std::endl;
            oss << "  Number of heads: " << config_.num_heads << std::endl;
            oss << "  Number of layers: " << config_.num_layers << std::endl;
            oss << "  MLP hidden dimension: " << config_.mlp_hidden_dim << std::endl;
            oss << "  Parameters: " << parameterCount() << std::endl;

            if (exec_context_ && exec_context_->getDevice())
            {
                oss << "  Device: " << deviceTypeToString( exec_context_->getDevice()->getDeviceType() ) << std::endl;
            }

            if (this->isBuilt())
            {
                oss << "  Built with:" << std::endl;
                oss << "    Batch size: " << batch_size_ << std::endl;
                oss << "    Sequence length: " << seq_length_ << std::endl;
                oss << "    Input shape: (";

                for (size_t i = 0; i < cached_input_shape_.size(); ++i)
                {
                    oss << cached_input_shape_[i];
                    if (i != cached_input_shape_.size() - 1)
                        oss << ", ";
                }

                oss << ")" << std::endl;

                oss << "    Output shape: (";
                for (size_t i = 0; i < cached_output_shape_.size(); ++i)
                {
                    oss << cached_output_shape_[i];
                    if (i != cached_output_shape_.size() - 1)
                        oss << ", ";
                }

                oss << ")" << std::endl;
            }

            oss << "  Sub-Modules:" << std::endl;
            oss << "    - encoder: " << encoder_->getName() << std::endl;
            oss << "    - transformer_blocks: " << transformer_blocks_.size() << " layers" << std::endl;
            oss << "    - final_layernorm: " << final_layernorm_->getName() << std::endl;
            oss << "    - lm_head: " << lm_head_->getName() << std::endl;

            oss << std::endl;

            return oss.str();
        }

        const CharTransformerConfig& getConfig() const
        {
            return config_;
        }

        // ====================================================================
        // Child module accessors
        // ====================================================================

        std::shared_ptr<EncoderType> getEncoder() const noexcept
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
        }

    protected:

        /**
         * @brief Hook invoked when training mode is about to change.
         *
         * Propagate the new mode to child modules and invalidate any cached
         * forward-executed state. Called with the CompositeComponent's mutex held;
         * do not call setTraining() within this hook.
         */
        void onTrainingChanging( bool newMode ) override
        {
            if (encoder_) encoder_->setTraining( newMode );

            for (auto& block : transformer_blocks_)
            {
                if (block) block->setTraining( newMode );
            }

            if (final_layernorm_) final_layernorm_->setTraining( newMode );
            if (lm_head_) lm_head_->setTraining( newMode );

            // TODO: forward_executed_ = false;
        }

    private:
        CharTransformerConfig config_;
        std::shared_ptr<ExecutionContextType> exec_context_;

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

        void validateInputShape( const shape_t& input_shape ) const
        {
            if (input_shape.size() != 2)
            {
                throw std::invalid_argument(
                    "CharTransformer: input must have rank 2 (batch_size, seq_length)" );
            }

            if (input_shape[1] > config_.max_seq_length)
            {
                std::ostringstream oss;
                oss << "CharTransformer: sequence length " << input_shape[1]
                    << " exceeds maximum " << config_.max_seq_length;
                throw std::invalid_argument( oss.str() );
            }
        }

        void createComponents()
        {
            EncoderConfig enc_cfg;
            enc_cfg.withVocabularyLength( static_cast<size_t>( config_.vocab_size ) )
                   .withMaxSequenceLength( static_cast<size_t>( config_.max_seq_length ) )
                   .withChannels( static_cast<size_t>( config_.embedding_dim ) );
            enc_cfg.validate();

            encoder_ = std::make_shared<EncoderType>( exec_context_, enc_cfg );
            this->addComponent( encoder_ );

            transformer_blocks_.clear();

            for (int64_t i = 0; i < config_.num_layers; ++i)
            {
                TransformerConfig tcfg( static_cast<dim_t>(config_.embedding_dim),
                                        static_cast<dim_t>(config_.num_heads) );

                tcfg.withHiddenDimension( static_cast<dim_t>(config_.mlp_hidden_dim) );
                tcfg.withBias( false );
                tcfg.withActivation( ActivationType::Gelu );
                tcfg.withName( config_.name + ".layer" + std::to_string(i) );

                auto block = std::make_shared<TransformerBlockType>( exec_context_, tcfg );
                transformer_blocks_.push_back( std::move(block) );

                // Register block with composite so name-based lookup/serialization works.
                this->addComponent( transformer_blocks_.back() );
            }

            auto ln_config = LayerNormConfig()
                .withName( config_.name + ".final_layernorm" )
                .withNormalizedShape( { config_.embedding_dim } );

            final_layernorm_ = std::make_shared<LayerNormType>( exec_context_, ln_config );
            this->addComponent( final_layernorm_ );

            auto lm_head_config = LinearConfig( config_.embedding_dim, config_.vocab_size )
                .withName( config_.name + ".lm_head" )
                .withBias( false );

            lm_head_ = std::make_shared<LinearType>( exec_context_, lm_head_config );
            this->addComponent( lm_head_ );
        }
    };

    export template<TensorDataType TPrecision>
        using CpuCharTransformer = CharTransformer<DeviceType::Cpu, TPrecision>;

    export template<TensorDataType TPrecision>
        using CudaCharTransformer = CharTransformer<DeviceType::Cuda, TPrecision>;
}