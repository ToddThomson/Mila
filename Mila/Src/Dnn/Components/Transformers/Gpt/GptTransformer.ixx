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
#include <cstring>
#include <ostream>
#include <iostream>
#include <filesystem>
#include <format>
#include <optional>
#include <cassert>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>

export module Dnn.Components.GptTransformer;
export import :Config;
export import :Presets;

import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.LanguageNetwork;
import Dnn.Components.Linear;
import Dnn.Components.LayerNorm;
import Dnn.Components.Lpe;
import Dnn.Components.GptBlock;
import Dnn.Component;
import Dnn.ComponentType;
import Dnn.RuntimeMode;
import Dnn.ActivationType;
import Compute.Device;
import Compute.DeviceType;
import Compute.DeviceId;
import Compute.DeviceTypeTraits;
import Compute.CpuMemoryResource;
#ifdef MILA_HAS_CUDA
import Compute.CudaPinnedMemoryResource;
#endif
import Compute.ExecutionContext;
import Compute.ExecutionContextFactory;
import Serialization.ModelArchive;
import Serialization.Mode;
import Serialization.PretrainedReader;
import Serialization.Tensor;
import Logging.Logger;

namespace Mila::Dnn
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
    class GptTransformer : public LanguageNetwork<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using NetworkBase = LanguageNetwork<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using LinearType = Linear<TDeviceType, TPrecision>;
        using LayerNormType = LayerNorm<TDeviceType, TPrecision>;
        using TransformerBlockType = GptBlock<TDeviceType, TPrecision>;
        using EncoderType = Lpe<TDeviceType, dtype_t::INT32, TPrecision>;
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
        explicit GptTransformer( const std::string& name, const GptConfig& config, DeviceId device_id )
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

        // --------------------------------------------------------------------
        // Factory methods
        // --------------------------------------------------------------------

        // REVIEW: Is this factory method necessary given GptModel::fromPretrained()?

        static std::unique_ptr<GptTransformer<TDeviceType, TPrecision>> fromPretrained(
            const std::filesystem::path& model_path,
            std::size_t batch_size,      // User specifies runtime dimensions
            std::size_t seq_length,      // Must be ? max_seq_length from weights
            DeviceId device_id = DeviceId{ TDeviceType, 0 },
            bool strict = true )
        {
            PretrainedModelReader reader( model_path );
            const auto& metadata = reader.getPretrainedMetadata();

            // REVIEW: this->verifyArchitectureCompatibility( metadata );

            GptConfig config = createConfigFromMetadata( metadata );

            std::unique_ptr<GptTransformer<TDeviceType, TPrecision>> gpt =
                std::make_unique<GptTransformer<TDeviceType, TPrecision>>(
                    metadata.model_name,
                    config,
                    device_id
                );

            // Build with max sequence length (position embeddings support full range)
            auto build_config = BuildContext( { 1, config.getMaxSequenceLength() }, RuntimeMode::Inference );
            
            gpt->build( build_config );

            gpt->loadParameters( reader, strict );

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

        TensorType& forward( const TokenIndexType& input ) override
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

            return *logits_ptr_;
        }

        TokenIndexType& backward( const TokenIndexType& input, const TensorType& output_grad ) override
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "GptTransformer must be built before calling backward." );
            }

            if ( !this->isTrainingMode() )
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

        /**
         * @brief Inference prefill — process full prompt and return last-token logits.
         *
         * Populates the KV cache across all transformer blocks by running the
         * full prompt through encoder + blocks via forward(). Then extracts only
         * the last token's representation for the final LayerNorm + LM head,
         * avoiding the T=1 output buffer overflow that forward() would cause
         * on those components.
         *
         * Unlike LlamaTransformer::prefill(), GPT does not need chunked prefill
         * or explicit position offsets (no RoPE). The full sequence is processed
         * in a single pass.
         *
         * @param input  Full prompt token indices [B, T].
         * @return        Logits for the last token [B, 1, vocab_size].
         */
        TensorType& prefill( const TokenIndexType& input ) override
        {
            if ( !this->isBuilt() )
                throw std::runtime_error(
                    "GptTransformer must be built before calling prefill()." );

            int64_t T_prompt = input.shape()[ 1 ];

            // 1. Encoder — full sequence embedding (Lpe output buffer is full-sized)
            encoder_out_ptr_ = &encoder_->forward( input );
            this->getExecutionContext()->synchronize();

            if ( block_input_ptrs_.empty() || block_input_ptrs_.size() != transformer_blocks_.size() )
                throw std::runtime_error(
                    "GptTransformer: prefill internal state not initialized" );

            // 2. Blocks — full sequence forward populates KV cache
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

            // 3. Extract only the last token's representation for LN + LM head.
            //    This avoids writing T=seq_len into the T=1 inference output buffers
            //    allocated by LayerNorm and lm_head.
            size_t last_pos_offset = static_cast<size_t>(
                (T_prompt - 1) * config_.getEmbeddingSize());

            auto last_pos = block_output_ptrs_.back()->view(
                shape_t{ input.shape()[ 0 ], 1, config_.getEmbeddingSize() },
                last_pos_offset );

            normalized_ptr_ = &final_layernorm_->forward( last_pos );
            this->getExecutionContext()->synchronize();

            logits_ptr_ = &lm_head_->forward( *normalized_ptr_ );
            this->getExecutionContext()->synchronize();

            return *logits_ptr_;
        }

        /**
         * @brief Inference-only single-token decode pass.
         *
         * Mirrors forward() exactly except each transformer block is driven
         * via decode() rather than forward(). Each block's decode() delegates
         * to attn_->decode() for the attention step — Attention decides
         * internally whether to use the fast KV cache path or fall back to
         * forward(). All other components in each block use forward() unchanged.
         *
         * The encoder (token + position embeddings) and final LayerNorm + LM head
         * are identical to forward() — only the block traversal differs.
         *
         * Precondition: forward() must have been called at least once (prefill)
         * before decode() is called. Attention internally manages cache state —
         * no explicit initializeKVCache / resetKVCache needed here.
         *
         * Calling forward() again after decode() steps automatically resets
         * the KV cache and begins a new prefill session.
         *
         * @param input    Single-token input [B, 1] token indices.
         * @param position Current sequence position (0-based).
         * @return         Reference to logits tensor [B, 1, vocab_size].
         */
        TensorType& decode( const TokenIndexType& input, int position ) override
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "GptTransformer must be built before calling decode()." );
            }

            // Encoder — same as forward(), single token embedding
            encoder_out_ptr_ = &encoder_->decode( input, position );
            this->getExecutionContext()->synchronize();

            if ( block_input_ptrs_.empty() || block_input_ptrs_.size() != transformer_blocks_.size() )
            {
                throw std::runtime_error( "GptTransformer: decode internal state not initialized" );
            }

            // Block traversal — decode() instead of forward() on each block.
            // Attention inside each block decides KV cache vs fallback transparently.
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

            // Final LayerNorm + LM head — identical to forward()
            normalized_ptr_ = &final_layernorm_->forward( *block_output_ptrs_.back() );
            this->getExecutionContext()->synchronize();

            logits_ptr_ = &lm_head_->forward( *normalized_ptr_ );
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

            final_layernorm_->zeroGradients();
            lm_head_->zeroGradients();
        }

        const ComponentType getType() const override
        {
            return ComponentType::Gpt2;
        }

        MemoryStats getMemoryStats() const override
        {
            MemoryStats stats;

            for ( const auto& child : this->getComponents() )
            {
                stats += child->getMemoryStats();
            }

            return stats;
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

                oss << "  Input shape: ("; for ( size_t i = 0; i < leading_shape_.size(); ++i ) {
                    oss << leading_shape_[ i ]; if ( i != leading_shape_.size() - 1 ) oss << ", ";
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

        /**
         * @brief Load parameters (weights and biases) from an already-opened PretrainedModelReader
         *
         * Separated from fromPretrained to allow flexibility in weight loading
         */
        void loadParameters( PretrainedModelReader& reader, bool strict )
        {
            const int device_index = this->getExecutionContext()->getDeviceId().index;

            auto consume = [&]( const std::string& full_name, const Serialization::ITensorBlob& blob )
            {
                auto [component_path, param_name] = parseParameterPath( full_name );

                ComponentPtr target = nullptr;

                try
                {
                    target = this->findComponent( component_path );
                }
                catch ( const std::out_of_range& )
                {
                    if ( strict )
                        throw std::runtime_error( "Component not found: " + component_path );

                    return;
                }

                target->loadParameter( param_name, blob );

                // The reader reuses its pinned staging slot as soon as this returns, so the
                // device read of blob must be complete. The quantize-on-load H2D is async on
                // the op stream and does not self-synchronize; force completion here.
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
        }

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
                meta.set( "input_shape", leading_shape_ )
                    .set( "embedding_shape", embedding_shape_ )
                    .set( "output_shape", output_shape_ )
                    .set( "batch_size", batch_size_ )
                    .set( "seq_length", seq_length_ );
            }

            archive.writeMetadata( "transformer_meta.json", meta );
        }

        void onTrainingModeChanging( TrainingMode training_mode ) override
        {
            NetworkBase::onTrainingModeChanging( training_mode );
        }

        void onBuilding( const BuildContext& context ) override
        {
            const auto& input_shape = context.inputShape();
            validateBuildContext( context );

            // encoder receives token ids — same shape as incoming context [B, T]
            encoder_ = this->template getComponentAs<EncoderType>(
                this->getName() + ".lenc" );
            encoder_->build( context );

            // All downstream components receive embeddings [B, T, embedding_dim]
            shape_t embedding_shape = {
                input_shape[ 0 ],
                input_shape[ 1 ],
                config_.getEmbeddingSize()
            };
            BuildContext embedding_context( embedding_shape, context.getRuntimeMode() );

            transformer_blocks_.clear();
            transformer_blocks_.reserve( static_cast<size_t>(config_.getNumLayers()) );

            for ( int64_t i = 0; i < config_.getNumLayers(); ++i )
            {
                std::string block_name = this->getName() + ".tf_layer_" + std::to_string( i );
                auto block = this->template getComponentAs<TransformerBlockType>( block_name );
                block->build( embedding_context );
                transformer_blocks_.push_back( block );
            }

            final_layernorm_ = this->template getComponentAs<LayerNormType>(
                this->getName() + ".ln_final" );
            final_layernorm_->build( embedding_context );

            lm_head_ = this->template getComponentAs<LinearType>(
                this->getName() + ".lm_head" );
            lm_head_->build( embedding_context );

            block_input_ptrs_.assign( transformer_blocks_.size(), nullptr );
            block_output_ptrs_.assign( transformer_blocks_.size(), nullptr );
            encoder_out_ptr_ = nullptr;
            normalized_ptr_ = nullptr;
            logits_ptr_ = nullptr;
        }

    private:
        std::unique_ptr<IExecutionContext> owned_context_{ nullptr };

        GptConfig config_;

        shape_t leading_shape_;
        shape_t embedding_shape_;
        shape_t output_shape_;
        int64_t batch_size_{ 0 };
        int64_t seq_length_{ 0 };

        std::shared_ptr<EncoderType> encoder_{ nullptr };
        std::vector<std::shared_ptr<TransformerBlockType>> transformer_blocks_;
        std::shared_ptr<LayerNormType> final_layernorm_{ nullptr };
        std::shared_ptr<LinearType> lm_head_{ nullptr };

        //std::shared_ptr<TensorType> output_{ nullptr };
        //std::unique_ptr<TensorType> output_view_{ nullptr };

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

        void validateBuildContext( const BuildContext& context ) const
        {
            const auto& input_shape = context.inputShape();

            if ( input_shape.size() != 2 )
            {
                throw std::invalid_argument( std::format(
                    "GptTransformer: input must be rank 2 [B, T], got rank {}",
                    input_shape.size() ) );
            }

            if ( input_shape[ 0 ] < 1 || input_shape[ 1 ] < 1 )
            {
                throw std::invalid_argument( std::format(
                    "GptTransformer: B and T must be >= 1, got [{}, {}]",
                    input_shape[ 0 ], input_shape[ 1 ] ) );
            }
        }

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
            LpeConfig enc_cfg;
            enc_cfg.withVocabularyLength( static_cast<size_t>(config_.getVocabSize() ))
                .withMaxSequenceLength( static_cast<size_t>(config_.getMaxSequenceLength() ))
                .withEmbeddingDim( static_cast<size_t>(config_.getEmbeddingSize()));

            enc_cfg.validate();

            auto encoder = std::make_shared<EncoderType>( this->getName() + ".lenc", enc_cfg );
            this->addComponent( encoder );

            for ( int64_t i = 0; i < config_.getNumLayers(); ++i )
            {
                GptBlockConfig block_cfg( 
                    static_cast<dim_t>( config_.getEmbeddingSize() ),
                    static_cast<dim_t>( config_.getNumHeads() ));

                block_cfg.withHiddenSize( static_cast<dim_t>( config_.getHiddenSize() ))
                    .withBias( config_.getUseBias() )
                    .withActivation( ActivationType::Gelu )
                    .withResidualScale( 1.0f );

                auto layer = std::make_shared<TransformerBlockType>( this->getName() + ".tf_layer_" + std::to_string( i ), block_cfg, std::nullopt );
                this->addComponent( layer );
            }

            auto ln_config = LayerNormConfig( shape_t{ config_.getEmbeddingSize() } );

            auto final_layernorm = std::make_shared<LayerNormType>( this->getName() + ".ln_final", ln_config, std::nullopt );
            this->addComponent( final_layernorm );

            auto lm_head_config = LinearConfig( config_.getEmbeddingSize(), config_.getVocabSize() )
                .withBias( false );

            auto lm_head = std::make_shared<LinearType>( this->getName() + ".lm_head", lm_head_config, std::nullopt );
            this->addComponent( lm_head );
        }

        /**
         * @brief Create GptConfig from Mila metadata.
         */
        static auto createConfigFromMetadata( const PretrainedMetadata& metadata ) -> GptConfig
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