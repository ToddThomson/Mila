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
import Dnn.Network;
import Dnn.Components.Linear;
import Dnn.Components.LayerNorm;
import Dnn.Components.LearnedEncoder;
import Dnn.Components.GptBlock;
import Dnn.Component;
import Dnn.ComponentType;
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
import Utils.Logger;

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
    class GptTransformer : public Network<TDeviceType, TPrecision>
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
            shape_t build_shape = { 1, config.getMaxSequenceLength() };
            gpt->build( build_shape );

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

            return *logits_ptr_;
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

        /**
         * @brief Autoregressive text generation from prompt tokens.
         *
         * @param prompt_tokens Initial token IDs to condition generation
         * @param max_new_tokens Maximum number of tokens to generate
         * @param temperature Sampling temperature (lower = more deterministic, higher = more random)
         * @param top_k If > 0, only sample from top k most probable tokens
         * @return Vector of all token IDs (prompt + generated)
         *
         * @note Uses dynamic sequence lengths for efficiency. Each forward pass processes
         *       only the actual number of tokens, avoiding padding overhead.
         */
        std::vector<int32_t> generate_naive(
            const std::vector<int32_t>& prompt_tokens,
            size_t max_new_tokens = 64,
            float temperature = 1.0f,
            int top_k = 0 )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "GptTransformer must be built before calling generate()" );
            }

            // Ensure we're in inference mode
            bool was_training = this->isTraining();
            this->setTraining( false );

            std::vector<int32_t> tokens = prompt_tokens;
            std::mt19937 rng( std::chrono::high_resolution_clock::now().time_since_epoch().count() );

            try
            {
                for ( size_t step = 0; step < max_new_tokens; ++step )
                {
                    // Create input tensor with actual sequence length (no padding!)
                    int64_t seq_len = static_cast<int64_t>( tokens.size() );

                    if ( seq_len > config_.getMaxSequenceLength() )
                    {
                        Utils::Logger::warning( std::format(
                            "Sequence length {} exceeds max {}, truncating from start",
                            seq_len, config_.getMaxSequenceLength() ) );

                        // Keep only the last max_seq_length tokens
                        int64_t excess = seq_len - config_.getMaxSequenceLength();
                        tokens.erase( tokens.begin(), tokens.begin() + excess );
                        seq_len = config_.getMaxSequenceLength();
                    }

                    shape_t input_shape = { 1, seq_len };

                    // Create tensors for this iteration (use explicit CPU memory resource for host access)
                    TokenIndexType input_device( this->getDeviceId(), input_shape );
                    Tensor<dtype_t::INT32, CpuMemoryResource> input_cpu( Device::Cpu(), input_shape );

                    // Copy tokens to CPU tensor
                    std::memcpy( input_cpu.data(), tokens.data(), tokens.size() * sizeof(int32_t) );

                    // Transfer to device
                    copy( input_cpu, input_device );

                    // Forward pass - returns [1, seq_len, vocab_size]
                    auto& logits = this->forward( input_device );
                    this->getExecutionContext()->synchronize();

                    // Copy logits for last position to CPU
                    shape_t logits_shape = { 1, seq_len, config_.getVocabSize() };
                    Tensor<TPrecision, CpuMemoryResource> logits_cpu( Device::Cpu(), logits_shape );
                    copy( logits, logits_cpu );

                    // Get logits for last token position
                    size_t last_pos_offset = static_cast<size_t>(seq_len - 1) * config_.getVocabSize();
                    const float* last_logits = logits_cpu.data() + last_pos_offset;

                    // Sample next token
                    int32_t next_token = sampleToken(
                        last_logits,
                        static_cast<size_t>(config_.getVocabSize()),
                        temperature,
                        top_k,
                        rng );

                    tokens.push_back( next_token );

                    // Check for EOS token (50256 for GPT-2)
                    if ( next_token == 50256 )
                    {  // GPT-2's <|endoftext|> token
                        break;
                    }
                }
            }
            catch ( ... )
            {
                // Restore training state before re-throwing
                this->setTraining( was_training );
                throw;
            }

            // Restore training state
            this->setTraining( was_training );

            return tokens;
        }

        // ====================================================================
        // KV Caching
        // ====================================================================

        void initializeKVCache( int64_t max_seq_len )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "GptTransformer must be built before initializeKVCache()." );
            }

            for ( auto& block : transformer_blocks_ )
            {
                block->initializeKVCache( max_seq_len );
            }
        }

        void resetKVCache()
        {
            for ( auto& block : transformer_blocks_ )
            {
                block->resetKVCache();
            }
        }

        // ====================================================================
        // Decoding
        // ====================================================================

        TensorType& forwardPrefill( const TokenIndexType& input )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "GptTransformer must be built before calling forwardPrefill()." );
            }

            encoder_out_ptr_ = &encoder_->forward( input );
            this->getExecutionContext()->synchronize();

            block_input_ptrs_[ 0 ] = encoder_out_ptr_;

            for ( size_t i = 0; i < transformer_blocks_.size(); ++i )
            {
                auto& block_out = transformer_blocks_[ i ]->forwardPrefill( *block_input_ptrs_[ i ] );
                this->getExecutionContext()->synchronize();

                block_output_ptrs_[ i ] = &block_out;

                if ( i + 1 < transformer_blocks_.size() )
                {
                    block_input_ptrs_[ i + 1 ] = &block_out;
                }
            }

            normalized_ptr_ = &final_layernorm_->forward( *block_output_ptrs_.back() );
            this->getExecutionContext()->synchronize();

            auto host = toHost<TensorDataType::FP32>( *normalized_ptr_ );
            Utils::Logger::info( std::format( "ln_final out pos 0 elem 0: {:.4f}",
                host.data()[ 0 ] ) );
            Utils::Logger::info( std::format( "ln_final out pos 3 elem 0: {:.4f}",
                host.data()[ 3 * 768 ] ) );

            logits_ptr_ = &lm_head_->forward( *normalized_ptr_ );
            this->getExecutionContext()->synchronize();

            return *logits_ptr_;
        }

        TensorType& forwardDecode( const TokenIndexType& input, int position )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "GptTransformer must be built before calling forwardDecode()." );
            }

            // DEBUG
            auto input_cpu = toHost<dtype_t::INT32>( input );
            Utils::Logger::info( std::format( "forwardDecode: input token id = {}",
                input_cpu.data()[ 0 ] ) );
            // END DEBUG

            encoder_out_ptr_ = &encoder_->forward( input );
            this->getExecutionContext()->synchronize();

            block_input_ptrs_[ 0 ] = encoder_out_ptr_;

            for ( size_t i = 0; i < transformer_blocks_.size(); ++i )
            {
                auto& block_out = transformer_blocks_[ i ]->forwardDecode( *block_input_ptrs_[ i ], position );
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

        std::vector<int32_t> generate(
            const std::vector<int32_t>& prompt_tokens,
            size_t max_new_tokens = 64,
            float temperature = 1.0f,
            int top_k = 0 )
        {
            if ( !this->isBuilt() )
            {
                throw std::runtime_error( "GptTransformer must be built before calling generate()" );
            }

            bool was_training = this->isTraining();
            this->setTraining( false );

            std::vector<int32_t> tokens = prompt_tokens;
            std::mt19937 rng( std::chrono::high_resolution_clock::now().time_since_epoch().count() );

            try
            {
                initializeKVCache( config_.getMaxSequenceLength() );

                for ( size_t step = 0; step < max_new_tokens; ++step )
                {
                    int64_t seq_len = static_cast<int64_t>( tokens.size() );

                    if ( seq_len > config_.getMaxSequenceLength() )
                    {
                        Utils::Logger::warning( std::format(
                            "Sequence length {} exceeds max {}, truncating from start",
                            seq_len, config_.getMaxSequenceLength() ) );

                        int64_t excess = seq_len - config_.getMaxSequenceLength();
                        tokens.erase( tokens.begin(), tokens.begin() + excess );
                        seq_len = config_.getMaxSequenceLength();
                    }

                    if ( step == 0 )
                    {
                        shape_t input_shape = { 1, seq_len };

                        TokenIndexType input_device( this->getDeviceId(), input_shape );
                        Tensor<dtype_t::INT32, CpuMemoryResource> input_cpu( Device::Cpu(), input_shape );

                        std::memcpy( input_cpu.data(), tokens.data(), tokens.size() * sizeof( int32_t ) );
                        copy( input_cpu, input_device );

                        auto& logits = forwardPrefill( input_device );
                        this->getExecutionContext()->synchronize();

                        shape_t logits_shape = { 1, seq_len, config_.getVocabSize() };
                        Tensor<TPrecision, CpuMemoryResource> logits_cpu( Device::Cpu(), logits_shape );
                        copy( logits, logits_cpu );

                        // DEBUG: Add here
                        size_t last_pos_offset = static_cast<size_t>(seq_len - 1) * config_.getVocabSize();

                        Utils::Logger::info( std::format( "logit token 11 (',') at pos 0: {:.4f}",
                            logits_cpu.data()[ 11 ] ) );
                        Utils::Logger::info( std::format( "logit token 11 (',') at pos 3: {:.4f}",
                            logits_cpu.data()[ last_pos_offset + 11 ] ) );
                        Utils::Logger::info( std::format( "logit token 284 (' to') at pos 0: {:.4f}",
                            logits_cpu.data()[ 284 ] ) );
                        Utils::Logger::info( std::format( "logit token 284 (' to') at pos 3: {:.4f}",
                            logits_cpu.data()[ last_pos_offset + 284 ] ) );
                        // END DEBUG

                        // DEBUG: start
                        Utils::Logger::info( std::format( "prefill logits shape: [1, {}, {}]",
                            seq_len, config_.getVocabSize() ) );
                        Utils::Logger::info( std::format( "last_pos_offset: {} (= {} * {})",
                            (seq_len - 1) * config_.getVocabSize(),
                            seq_len - 1,
                            config_.getVocabSize() ) );

                        // Also log the logit value at token 11 (',') at last position
                        last_pos_offset = static_cast<size_t>(seq_len - 1) * config_.getVocabSize();
                        const float* last_logits = logits_cpu.data() + last_pos_offset;
                        Utils::Logger::info( std::format( "logit for token 11 (',') at last pos: {:.4f}",
                            last_logits[ 11 ] ) );
                        Utils::Logger::info( std::format( "logit for token 284 (' to') at last pos: {:.4f}",
                            last_logits[ 284 ] ) );
                        // END DEBUG

                        int32_t next_token = sampleToken(
                            last_logits,
                            static_cast<size_t>(config_.getVocabSize()),
                            temperature,
                            top_k,
                            rng );

                        tokens.push_back( next_token );

                        if ( next_token == 50256 )
                        {
                            break;
                        }

                        continue;
                    }

                    shape_t input_shape = { 1, 1 };

                    TokenIndexType input_device( this->getDeviceId(), input_shape );
                    Tensor<dtype_t::INT32, CpuMemoryResource> input_cpu( Device::Cpu(), input_shape );

                    input_cpu.data()[ 0 ] = tokens.back();
                    copy( input_cpu, input_device );

                    auto& logits = forwardDecode( input_device, static_cast<int>(seq_len - 1) );
                    this->getExecutionContext()->synchronize();

                    shape_t logits_shape = { 1, 1, config_.getVocabSize() };
                    Tensor<TPrecision, CpuMemoryResource> logits_cpu( Device::Cpu(), logits_shape );
                    copy( logits, logits_cpu );

                    const float* last_logits = logits_cpu.data();

                    std::vector<std::pair<float, int>> top5_logits;
                    for ( int i = 0; i < (int)config_.getVocabSize(); i++ )
                        top5_logits.push_back( { last_logits[ i ], i } );
                    std::partial_sort( top5_logits.begin(), top5_logits.begin() + 5, top5_logits.end(),
                        []( auto& a, auto& b ) { return a.first > b.first; } );
                    for ( int i = 0; i < 5; i++ )
                        Utils::Logger::info( std::format( "  top{}: token={} logit={:.4f}",
                            i, top5_logits[ i ].second, top5_logits[ i ].first ) );

                    int32_t next_token = sampleToken(
                        last_logits,
                        static_cast<size_t>(config_.getVocabSize()),
                        temperature,
                        top_k,
                        rng );

                    tokens.push_back( next_token );

                    if ( next_token == 50256 )
                    {
                        break;
                    }
                }
            }
            catch ( ... )
            {
                this->setTraining( was_training );
                resetKVCache();
                throw;
            }

            resetKVCache();
            this->setTraining( was_training );

            return tokens;
        }

        // ====================================================================


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

            //auto device_id = this->getDeviceId();

            //owned_output_ = std::make_shared<TensorType>( device_id, output_shape_ );
            //owned_output_->setName( this->getName() + ".output" );

            block_input_ptrs_.assign( transformer_blocks_.size(), nullptr );
            block_output_ptrs_.assign( transformer_blocks_.size(), nullptr );

            encoder_out_ptr_ = nullptr;
            normalized_ptr_ = nullptr;
            logits_ptr_ = nullptr;
        }

        //Tensor<TPrecision, MR>& forwardPrefill(
        //    const Tensor<TensorDataType::INT32, MR>& tokens,
        //    int64_t seq_len ) override
        //{
        //    // Standard GPT forward with full sequence
        //    // Attention layers populate their KV caches
        //    return this->forward( tokens );  // Existing forward()
        //}

        //Tensor<TPrecision, MR>& forwardDecode(
        //    const Tensor<TensorDataType::INT32, MR>& token,
        //    int64_t position ) override
        //{
        //    // Single token forward, using cached K,V
        //    // Only compute new Q,K,V
        //    return forwardSingleToken( token, position );
        //}

        //void initializeKVCache( int64_t max_len ) override {

        //    for ( auto& block : transformer_blocks_ ) {
        //        block->initializeKVCache( max_len );
        //    }
        //}

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

        //std::shared_ptr<TensorType> owned_output_{ nullptr };
        //std::unique_ptr<TensorType> output_view_{ nullptr };

        TensorType* encoder_out_ptr_{ nullptr };
        std::vector<TensorType*> block_input_ptrs_;
        std::vector<TensorType*> block_output_ptrs_;

        TensorType* normalized_ptr_{ nullptr };
        TensorType* logits_ptr_{ nullptr };

        /**
         * @brief Sample a token from logits using temperature and optional top-k filtering.
         */
        static int32_t sampleToken(
            const float* logits,
            size_t vocab_size,
            float temperature,
            int top_k,
            std::mt19937& rng )
        {
            // Find max logit for numerical stability
            float max_logit = *std::max_element( logits, logits + vocab_size );

            // Apply temperature and compute exp
            std::vector<float> probs( vocab_size );
            double sum = 0.0;

            for ( size_t i = 0; i < vocab_size; ++i )
            {
                float scaled = (logits[ i ] - max_logit) / temperature;
                float exp_val = std::exp( scaled );
                probs[ i ] = exp_val;
                sum += exp_val;
            }

            // Normalize to probabilities
            for ( size_t i = 0; i < vocab_size; ++i )
            {
                probs[ i ] /= sum;
            }

            // Apply top-k filtering if requested
            if ( top_k > 0 && top_k < static_cast<int>( vocab_size ) )
            {
                // Create indices and sort by probability (descending)
                std::vector<size_t> indices( vocab_size );
                std::iota( indices.begin(), indices.end(), 0 );
                std::partial_sort( indices.begin(), indices.begin() + top_k, indices.end(),
                    [&]( size_t a, size_t b ) { return probs[ a ] > probs[ b ]; } );

                // Zero out probabilities outside top-k
                std::vector<float> top_k_probs( vocab_size, 0.0f );
                double top_k_sum = 0.0;
                for ( int i = 0; i < top_k; ++i )
                {
                    top_k_probs[ indices[ i ] ] = probs[ indices[ i ] ];
                    top_k_sum += probs[ indices[ i ] ];
                }

                // Renormalize
                for ( size_t i = 0; i < vocab_size; ++i )
                {
                    probs[ i ] = top_k_probs[ i ] / top_k_sum;
                }
            }

            // Sample from the distribution
            std::uniform_real_distribution<float> dist( 0.0f, 1.0f );
            float r = dist( rng );
            float cumsum = 0.0f;

            for ( size_t i = 0; i < vocab_size; ++i )
            {
                cumsum += probs[ i ];
                if ( r < cumsum )
                {
                    return static_cast<int32_t>( i );
                }
            }

            // Fallback (should rarely happen)
            return static_cast<int32_t>( vocab_size - 1 );
        }

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

        /**
         * @brief Load parameters (weights and biases) from an already-opened PretrainedModelReader
         *
         * Separated from fromPretrained to allow flexibility in weight loading
         */
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
                .withEmbeddingDim( static_cast<size_t>(config_.getEmbeddingSize()));

            enc_cfg.validate();

            auto encoder = std::make_shared<EncoderType>(
                this->getName() + ".lenc", enc_cfg );

            this->addComponent( encoder );

            for ( int64_t i = 0; i < config_.getNumLayers(); ++i )
            {
                GptBlockConfig block_cfg( 
                    static_cast<dim_t>( config_.getEmbeddingSize() ),
                    static_cast<dim_t>( config_.getNumHeads() ));

                block_cfg.withHiddenSize( static_cast<dim_t>( config_.getHiddenSize() ))
                    .withBias( config_.getUseBias() )
                    .withActivation( ActivationType::Gelu )
                    .withResidualScale( 1.0f ); // FIXME: Was 1.of / sqrtf( static_cast<float>( config_.getNumLayers() ) ) );

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
                .withBias( false )
                .withRowMajor( true );

            auto lm_head = std::make_shared<LinearType>(
                this->getName() + ".lm_head", lm_head_config, std::nullopt );

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