/**
 * @file CharTransformer.ixx
 * @brief Character-level transformer for language modeling.
 *
 * Device-templated composite module implementing a transformer architecture
 * for character-level next-token prediction. Architecture follows GPT-style
 * decoder-only design with causal masking.
 *
 * Architecture:
 *   Token Embedding -> Positional Encoding -> N x Transformer Blocks -> LayerNorm -> LM Head
 *
 * Each Transformer Block contains:
 *   - LayerNorm -> Multi-Head Self-Attention -> Residual
 *   - LayerNorm -> MLP (Linear -> GELU -> Linear) -> Residual
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
     *
     * Design philosophy:
     * - Two-phase initialization: build() performs shape validation and buffer allocation
     * - Composite module pattern: manages embeddings, transformer blocks, and LM head
     * - Causal attention masking for autoregressive generation
     * - Positional encoding for sequence position awareness
     * - Child modules stored as concrete types for type safety and direct access
     *
     * @tparam TDeviceType Device type (DeviceType::Cpu or DeviceType::Cuda)
     * @tparam TPrecision Abstract tensor precision (TensorDataType)
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class CharTransformer : public CompositeModule<TDeviceType>
    {
    public:
        using MR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaDeviceMemoryResource, CpuMemoryResource>;
        using CompositeModuleBase = CompositeModule<TDeviceType>;
        using ExecutionContextType = ExecutionContext<TDeviceType>;
        using TensorType = Tensor<TPrecision, MR>;
        using LinearType = Linear<TDeviceType, TPrecision>;
        using LayerNormType = LayerNorm<TDeviceType, TPrecision>;

        /**
         * @brief Construct transformer with execution context and configuration.
         *
         * @param exec_context Shared execution context for device resources.
         * @param config Transformer configuration (vocab size, dimensions, etc.)
         */
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

            createModules();
        }

        ~CharTransformer() override = default;

        // ====================================================================
        // Lifecycle
        // ====================================================================

        bool isBuilt() const override
        {
            return built_ && token_embedding_ && final_layernorm_ && lm_head_;
        }

        /**
         * @brief Build the transformer for a concrete input shape.
         *
         * COLD PATH - performs one-time setup:
         * - Validates input shape (batch_size, seq_length)
         * - Builds all child modules with appropriate shapes
         * - Allocates intermediate buffer tensors for forward/backward passes
         *
         * @param input_shape Expected shape: (batch_size, seq_length)
         */
        void build( const shape_t& input_shape ) override
        {
            if (built_)
                return;

            validateInputShape( input_shape );

            cached_input_shape_ = input_shape;
            batch_size_ = input_shape[0];
            seq_length_ = input_shape[1];

            cached_embedding_shape_ = { batch_size_, seq_length_, config_.embedding_dim };
            cached_output_shape_ = { batch_size_, seq_length_, config_.vocab_size };

            token_embedding_->build( input_shape );

            // TODO: Build transformer blocks when implemented
            // for (auto& block : transformer_blocks_)
            // {
            //     block->build( cached_embedding_shape_ );
            // }

            final_layernorm_->build( cached_embedding_shape_ );
            lm_head_->build( cached_embedding_shape_ );

            auto device = exec_context_->getDevice();

            embedded_ = std::make_shared<TensorType>( device, cached_embedding_shape_ );
            embedded_->setName( config_.name + ".embedded" );

            transformer_output_ = std::make_shared<TensorType>( device, cached_embedding_shape_ );
            transformer_output_->setName( config_.name + ".transformer_output" );

            normalized_ = std::make_shared<TensorType>( device, cached_embedding_shape_ );
            normalized_->setName( config_.name + ".normalized" );

            built_ = true;
        }

        // ====================================================================
        // Compute operation dispatch
        // ====================================================================

        /**
         * @brief Forward pass - HOT PATH, pure dispatch to child modules.
         *
         * Processes character tokens through the transformer to produce next-token logits.
         *
         * @param input Input tensor containing token indices (batch_size, seq_length)
         * @param output Output tensor for logits (batch_size, seq_length, vocab_size)
         */
        void forward( const ITensor& input, ITensor& output )
        {
            if (!isBuilt())
            {
                throw std::runtime_error( "CharTransformer must be built before calling forward." );
            }

            token_embedding_->forward( input, *embedded_ );

            // TODO: Add positional encoding
            // positional_encoding_->forward( *embedded_, *embedded_ );

            // TODO: Transformer blocks with causal masking
            // ITensor* block_input = embedded_.get();
            // for (auto& block : transformer_blocks_)
            // {
            //     block->forward( *block_input, *transformer_output_ );
            //     block_input = transformer_output_.get();
            // }

            copy( *embedded_, *transformer_output_ );

            final_layernorm_->forward( *transformer_output_, *normalized_ );
            lm_head_->forward( *normalized_, output );
        }

        /**
         * @brief Backward pass - HOT PATH, pure dispatch to child modules.
         *
         * Chains backward calls through the transformer in reverse order.
         */
        void backward( const ITensor& input, const ITensor& output_grad, ITensor& input_grad )
        {
            if (!isBuilt())
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

            // TODO: Backprop through transformer blocks in reverse
            // for (auto it = transformer_blocks_.rbegin(); it != transformer_blocks_.rend(); ++it)
            // {
            //     TensorType block_input_grad( device, cached_embedding_shape_ );
            //     zeros( block_input_grad );
            //     (*it)->backward( block_input, transformer_output_grad, block_input_grad );
            //     transformer_output_grad = std::move( block_input_grad );
            // }

            TensorType& embedding_grad = transformer_output_grad;

            // TODO: Backprop through positional encoding if implemented

            token_embedding_->backward( input, embedding_grad, input_grad );
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        void save( ModelArchive& archive ) const override
        {
            token_embedding_->save( archive );

            // TODO: Save transformer blocks
            // for (const auto& block : transformer_blocks_)
            // {
            //     block->save( archive );
            // }

            final_layernorm_->save( archive );
            lm_head_->save( archive );
        }

        void load( ModelArchive& archive ) override
        {
            token_embedding_->load( archive );

            // TODO: Load transformer blocks
            // for (auto& block : transformer_blocks_)
            // {
            //     block->load( archive );
            // }

            final_layernorm_->load( archive );
            lm_head_->load( archive );
        }

        // ====================================================================
        // Module interface
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

            token_embedding_->synchronize();

            // TODO: Sync transformer blocks
            // for (auto& block : transformer_blocks_)
            // {
            //     block->synchronize();
            // }

            final_layernorm_->synchronize();
            lm_head_->synchronize();
        }

        void setTraining( bool is_training ) override
        {
            CompositeModuleBase::setTraining( is_training );

            token_embedding_->setTraining( is_training );

            // TODO: Set training mode for transformer blocks
            // for (auto& block : transformer_blocks_)
            // {
            //     block->setTraining( is_training );
            // }

            final_layernorm_->setTraining( is_training );
            lm_head_->setTraining( is_training );
        }

        bool isTraining() const override
        {
            return CompositeModuleBase::isTraining();
        }

        size_t parameterCount() const override
        {
            size_t total = 0;

            total += token_embedding_->parameterCount();

            // TODO: Count transformer block parameters
            // for (const auto& block : transformer_blocks_)
            // {
            //     total += block->parameterCount();
            // }

            total += final_layernorm_->parameterCount();
            total += lm_head_->parameterCount();

            return total;
        }

        std::vector<ITensor*> getParameters() const override
        {
            std::vector<ITensor*> params;

            auto emb_params = token_embedding_->getParameters();
            params.insert( params.end(), emb_params.begin(), emb_params.end() );

            // TODO: Collect transformer block parameters
            // for (const auto& block : transformer_blocks_)
            // {
            //     auto block_params = block->getParameters();
            //     params.insert( params.end(), block_params.begin(), block_params.end() );
            // }

            auto ln_params = final_layernorm_->getParameters();
            params.insert( params.end(), ln_params.begin(), ln_params.end() );

            auto lm_params = lm_head_->getParameters();
            params.insert( params.end(), lm_params.begin(), lm_params.end() );

            return params;
        }

        std::vector<ITensor*> getParameterGradients() const override
        {
            std::vector<ITensor*> grads;

            auto emb_grads = token_embedding_->getParameterGradients();
            grads.insert( grads.end(), emb_grads.begin(), emb_grads.end() );

            // TODO: Collect transformer block gradients
            // for (const auto& block : transformer_blocks_)
            // {
            //     auto block_grads = block->getParameterGradients();
            //     grads.insert( grads.end(), block_grads.begin(), block_grads.end() );
            // }

            auto ln_grads = final_layernorm_->getParameterGradients();
            grads.insert( grads.end(), ln_grads.begin(), ln_grads.end() );

            auto lm_grads = lm_head_->getParameterGradients();
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

            if (isBuilt())
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
            oss << "    - token_embedding: " << token_embedding_->getName() << std::endl;
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

        std::shared_ptr<LinearType> getTokenEmbedding() const noexcept
        {
            return token_embedding_;
        }

        std::shared_ptr<LayerNormType> getFinalLayerNorm() const noexcept
        {
            return final_layernorm_;
        }

        std::shared_ptr<LinearType> getLMHead() const noexcept
        {
            return lm_head_;
        }

    protected:

        void buildImpl( const shape_t& input_shape ) override
        {
            // Build is already handled by public build() method
        }

    private:
        CharTransformerConfig config_;
        bool built_{ false };
        std::shared_ptr<ExecutionContextType> exec_context_;

        shape_t cached_input_shape_;
        shape_t cached_embedding_shape_;
        shape_t cached_output_shape_;
        int64_t batch_size_{ 0 };
        int64_t seq_length_{ 0 };

        std::shared_ptr<LinearType> token_embedding_{ nullptr };
        // TODO: Add positional encoding module
        // std::shared_ptr<PositionalEncodingType> positional_encoding_{ nullptr };
        // TODO: Add transformer blocks
        // std::vector<std::shared_ptr<TransformerType>> transformer_blocks_;
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

        void createModules()
        {
            auto embedding_config = LinearConfig( config_.vocab_size, config_.embedding_dim );
            embedding_config.withName( config_.name + ".token_embedding" )
                .withBias( false );

            token_embedding_ = std::make_shared<LinearType>(
                exec_context_, embedding_config );

            // TODO: Add positional encoding
            // TODO: Add transformer blocks

            auto ln_config = LayerNormConfig()
                .withName( config_.name + ".final_layernorm" )
                .withNormalizedShape( { config_.embedding_dim } );

            final_layernorm_ = std::make_shared<LayerNormType>(
                exec_context_, ln_config );

            auto lm_head_config = LinearConfig( config_.embedding_dim, config_.vocab_size )
                .withName( config_.name + ".lm_head" )
                .withBias( false );

            lm_head_ = std::make_shared<LinearType>(
                exec_context_, lm_head_config );
        }
    };

    export template<TensorDataType TPrecision>
        using CpuCharTransformer = CharTransformer<DeviceType::Cpu, TPrecision>;

    export template<TensorDataType TPrecision>
        using CudaCharTransformer = CharTransformer<DeviceType::Cuda, TPrecision>;
}