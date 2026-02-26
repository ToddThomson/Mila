/**
 * @file GptModel.ixx
 * @brief GPT inference model.
 *
 * Inference-only wrapper around a loaded GptTransformer network.
 * No training, no optimizer, no gradients.
 *
 * Two loading paths:
 *
 *  fromPretrained() — third-party weights (e.g. HuggingFace GPT-2) via
 *                     PretrainedModelReader. Primary path for Mila chat.
 *
 *  fromCheckpoint() — Mila-native artifact produced by GptTransformer::save()
 *                     via ModelArchive. Round-trip path after training.
 *
 * REVIEW: generate() currently uses naive full-sequence re-evaluation.
 * The optimized KV cache decode path is an open design problem to be
 * resolved in a future iteration.
 */

module;
#include <memory>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
#include <filesystem>
#include <format>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cstring>

export module Dnn.Models.GptModel;

import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.Components.GptTransformer;
import Compute.Precision;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.CpuMemoryResource;
import Compute.ExecutionContextFactory;
import Serialization.ModelArchive;
import Serialization.OpenMode;
import Serialization.Mode;
import Serialization.PretrainedReader;
import Utils.Logger;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief GPT inference model.
     *
     * Owns a loaded, built GptTransformer and exposes generate() for
     * autoregressive text generation.
     *
     * Construction is only possible via fromPretrained() or fromCheckpoint().
     * The network is always in a built, weights-loaded, inference-mode state
     * when generate() is called.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class GptModel
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;
        using TokenIndexType = Tensor<dtype_t::INT32, MR>;
        using GptTransformerType = GptTransformer<TDeviceType, TPrecision>;

        // Non-copyable, movable
        GptModel( const GptModel& ) = delete;
        GptModel& operator=( const GptModel& ) = delete;
        GptModel( GptModel&& ) = default;
        GptModel& operator=( GptModel&& ) = default;

        ~GptModel() = default;

        // ====================================================================
        // Factory — the sole construction paths
        // ====================================================================

        /**
         * @brief Load from third-party pretrained weights.
         *
         * Reads weights from a Mila-compatible pretrained artifact produced
         * by converting third-party checkpoints (e.g. HuggingFace GPT-2)
         * via PretrainedModelReader.
         *
         * @param path      Path to the pretrained artifact.
         * @param device_id Target device.
         * @param strict    Throws on unknown parameter names if true.
         * @return          Inference-ready GptModel.
         */
        static std::unique_ptr<GptModel> fromPretrained(
            const std::filesystem::path& path,
            DeviceId device_id = DeviceId{ TDeviceType, 0 },
            bool strict = true )
        {
            if ( device_id.type != TDeviceType )
                throw std::invalid_argument( std::format(
                    "GptModel::fromPretrained: device type mismatch: expected {}, got {}",
                    deviceTypeToString( TDeviceType ),
                    deviceTypeToString( device_id.type ) ) );

            PretrainedModelReader reader( path );
            const auto& metadata = reader.getPretrainedMetadata();

            GptConfig config = configFromMetadata( metadata );

            auto network = std::make_unique<GptTransformerType>(
                metadata.model_name, config, device_id );

            // Build at max sequence length — position embeddings cover full range
            network->build( shape_t{ 1, config.getMaxSequenceLength() } );
            network->setTraining( false );
            network->loadParameters( reader, strict );

            return std::unique_ptr<GptModel>( new GptModel( std::move( network ), config ) );
        }

        /**
         * @brief Load from a Mila-native serialized artifact.
         *
         * Reads a checkpoint or weights-only artifact produced by
         * GptTransformer::save() via ModelArchive.
         *
         * @param path      Path to the Mila archive.
         * @param device_id Target device.
         * @return          Inference-ready GptModel.
         */
        static std::unique_ptr<GptModel> fromCheckpoint(
            const std::filesystem::path& path,
            DeviceId device_id = DeviceId{ TDeviceType, 0 } )
        {
            if ( device_id.type != TDeviceType )
                throw std::invalid_argument( std::format(
                    "GptModel::fromCheckpoint: device type mismatch: expected {}, got {}",
                    deviceTypeToString( TDeviceType ),
                    deviceTypeToString( device_id.type ) ) );

            ModelArchive archive( path, OpenMode::Read );

            GptConfig config = GptConfig::fromArchive( archive );

            auto network = std::make_unique<GptTransformerType>(
                "GptTransformer" /* FIXME: archive.readNetworkName() */, config, device_id);

            network->build( shape_t{ 1, config.getMaxSequenceLength() } );
            network->setTraining( false );
            network->load( archive, SerializationMode::WeightsOnly );

            return std::unique_ptr<GptModel>( new GptModel( std::move( network ), config ) );
        }

        // ====================================================================
        // Inference API
        // ====================================================================

        std::vector<int32_t> generate(
            const std::vector<int32_t>& prompt_tokens,
            size_t max_new_tokens = 64,
            float temperature = 1.0f,
            int top_k = 0 )
        {
            std::vector<int32_t> tokens = prompt_tokens;
            std::mt19937 rng( std::chrono::high_resolution_clock::now()
                .time_since_epoch().count() );

            // Phase 1: Prefill — process full prompt, populate KV cache
            truncateIfNeeded( tokens );
            int64_t seq_len = static_cast<int64_t>(tokens.size());
            auto prefill_input = makeTokenTensor( tokens );
            auto& logits = network_->forward( prefill_input );
            network_->getExecutionContext()->synchronize();

            // In GptModel::generate() after prefill forward():
            auto logits_cpu = toHost<TensorDataType::FP32>( logits );
            //auto logits_cpu = toHost( logits );
            int vocab_size = 50257;

            // What is Mila's top token at pos 3?
            float max_val = -1e9f;
            int   max_tok = 0;
            for ( int v = 0; v < vocab_size; ++v )
            {
                float val = logits_cpu.data()[ 3 * vocab_size + v ];
                if ( val > max_val )
                {
                    max_val = val; max_tok = v;
                }
            }
            Utils::Logger::info( std::format(
                "Prefill top token at pos 3: token={} logit={:.4f}", max_tok, max_val ) );

            // What is token 11's logit at pos 3?
            Utils::Logger::info( std::format(
                "Prefill token 11 (',') at pos 3: {:.4f}",
                logits_cpu.data()[ 3 * vocab_size + 11 ] ) );


            // Sample first token from last position of prefill
            int32_t next_token = sampleFromLogits(
                logits, seq_len - 1, temperature, top_k, rng );
            tokens.push_back( next_token );

            if ( next_token == eos_token_ )
                return tokens;

            // Phase 2: Decode — single token at a time using KV cache
            int position = static_cast<int>(seq_len);  // next position after prefill

            for ( size_t step = 1; step < max_new_tokens; ++step )
            {
                auto decode_input = makeTokenTensor( { next_token } );
                auto& decode_logits = network_->decode( decode_input, position );
                network_->getExecutionContext()->synchronize();

                next_token = sampleFromLogits(
                    decode_logits, 0, temperature, top_k, rng );  // pos 0 — single token output
                tokens.push_back( next_token );
                ++position;

                if ( next_token == eos_token_ )
                    break;
            }

            return tokens;
        }


        /**
         * @brief Autoregressive text generation from prompt tokens.
         *
         * Generates up to max_new_tokens tokens. Stops early on EOS.
         * Currently uses naive full-sequence re-evaluation each step.
         *
         * @param prompt_tokens  Input token IDs.
         * @param max_new_tokens Maximum tokens to generate.
         * @param temperature    Sampling temperature. 0 = greedy.
         * @param top_k          If > 0, restrict to top-k tokens.
         * @return               All token IDs (prompt + generated).
         */
        std::vector<int32_t> generate_no_decode(
            const std::vector<int32_t>& prompt_tokens,
            size_t max_new_tokens = 64,
            float temperature = 1.0f,
            int top_k = 0 )
        {
            std::vector<int32_t> tokens = prompt_tokens;
            std::mt19937 rng( std::chrono::high_resolution_clock::now()
                .time_since_epoch().count() );

            for ( size_t step = 0; step < max_new_tokens; ++step )
            {
                truncateIfNeeded( tokens );

                int64_t seq_len = static_cast<int64_t>( tokens.size() );
                auto input = makeTokenTensor( tokens );

                auto& logits = network_->forward( input );
                network_->getExecutionContext()->synchronize();

                int32_t next_token = sampleFromLogits(
                    logits, seq_len - 1, temperature, top_k, rng );

                tokens.push_back( next_token );

                if ( next_token == eos_token_ )
                    break;
            }

            return tokens;
        }

        //std::vector<int32_t> generate_optimized(
        //    const std::vector<int32_t>& prompt_tokens,
        //    size_t max_new_tokens = 64,
        //    float temperature = 1.0f,
        //    int top_k = 0 )
        //{
        //    if ( !this->isBuilt() )
        //    {
        //        throw std::runtime_error( "GptTransformer must be built before calling generate()" );
        //    }

        //    bool was_training = this->isTraining();
        //    this->setTraining( false );

        //    std::vector<int32_t> tokens = prompt_tokens;
        //    std::mt19937 rng( std::chrono::high_resolution_clock::now().time_since_epoch().count() );

        //    try
        //    {
        //        initializeKVCache( config_.getMaxSequenceLength() );

        //        for ( size_t step = 0; step < max_new_tokens; ++step )
        //        {
        //            int64_t seq_len = static_cast<int64_t>( tokens.size() );

        //            if ( seq_len > config_.getMaxSequenceLength() )
        //            {
        //                Utils::Logger::warning( std::format(
        //                    "Sequence length {} exceeds max {}, truncating from start",
        //                    seq_len, config_.getMaxSequenceLength() ) );

        //                int64_t excess = seq_len - config_.getMaxSequenceLength();
        //                tokens.erase( tokens.begin(), tokens.begin() + excess );
        //                seq_len = config_.getMaxSequenceLength();
        //            }

        //            if ( step == 0 )
        //            {
        //                shape_t input_shape = { 1, seq_len };

        //                TokenIndexType input_device( this->getDeviceId(), input_shape );
        //                Tensor<dtype_t::INT32, CpuMemoryResource> input_cpu( Device::Cpu(), input_shape );

        //                std::memcpy( input_cpu.data(), tokens.data(), tokens.size() * sizeof( int32_t ) );
        //                copy( input_cpu, input_device );

        //                auto& logits = forwardPrefill( input_device );
        //                this->getExecutionContext()->synchronize();

        //                shape_t logits_shape = { 1, seq_len, config_.getVocabSize() };
        //                Tensor<TPrecision, CpuMemoryResource> logits_cpu( Device::Cpu(), logits_shape );
        //                copy( logits, logits_cpu );

        //                // DEBUG: Add here
        //                size_t last_pos_offset = static_cast<size_t>(seq_len - 1) * config_.getVocabSize();

        //                Utils::Logger::info( std::format( "logit token 11 (',') at pos 0: {:.4f}",
        //                    logits_cpu.data()[ 11 ] ) );
        //                Utils::Logger::info( std::format( "logit token 11 (',') at pos 3: {:.4f}",
        //                    logits_cpu.data()[ last_pos_offset + 11 ] ) );
        //                Utils::Logger::info( std::format( "logit token 284 (' to') at pos 0: {:.4f}",
        //                    logits_cpu.data()[ 284 ] ) );
        //                Utils::Logger::info( std::format( "logit token 284 (' to') at pos 3: {:.4f}",
        //                    logits_cpu.data()[ last_pos_offset + 284 ] ) );
        //                // END DEBUG

        //                // DEBUG: start
        //                Utils::Logger::info( std::format( "prefill logits shape: [1, {}, {}]",
        //                    seq_len, config_.getVocabSize() ) );
        //                Utils::Logger::info( std::format( "last_pos_offset: {} (= {} * {})",
        //                    (seq_len - 1) * config_.getVocabSize(),
        //                    seq_len - 1,
        //                    config_.getVocabSize() ) );

        //                // Also log the logit value at token 11 (',') at last position
        //                last_pos_offset = static_cast<size_t>(seq_len - 1) * config_.getVocabSize();
        //                const float* last_logits = logits_cpu.data() + last_pos_offset;
        //                Utils::Logger::info( std::format( "logit for token 11 (',') at last pos: {:.4f}",
        //                    last_logits[ 11 ] ) );
        //                Utils::Logger::info( std::format( "logit for token 284 (' to') at last pos: {:.4f}",
        //                    last_logits[ 284 ] ) );
        //                // END DEBUG

        //                int32_t next_token = sampleToken(
        //                    last_logits,
        //                    static_cast<size_t>(config_.getVocabSize()),
        //                    temperature,
        //                    top_k,
        //                    rng );

        //                tokens.push_back( next_token );

        //                if ( next_token == 50256 )
        //                {
        //                    break;
        //                }

        //                continue;
        //            }

        //            shape_t input_shape = { 1, 1 };

        //            TokenIndexType input_device( this->getDeviceId(), input_shape );
        //            Tensor<dtype_t::INT32, CpuMemoryResource> input_cpu( Device::Cpu(), input_shape );

        //            input_cpu.data()[ 0 ] = tokens.back();
        //            copy( input_cpu, input_device );

        //            auto& logits = forwardDecode( input_device, static_cast<int>(seq_len - 1) );
        //            this->getExecutionContext()->synchronize();

        //            shape_t logits_shape = { 1, 1, config_.getVocabSize() };
        //            Tensor<TPrecision, CpuMemoryResource> logits_cpu( Device::Cpu(), logits_shape );
        //            copy( logits, logits_cpu );

        //            const float* last_logits = logits_cpu.data();

        //            std::vector<std::pair<float, int>> top5_logits;
        //            for ( int i = 0; i < (int)config_.getVocabSize(); i++ )
        //                top5_logits.push_back( { last_logits[ i ], i } );
        //            std::partial_sort( top5_logits.begin(), top5_logits.begin() + 5, top5_logits.end(),
        //                []( auto& a, auto& b ) { return a.first > b.first; } );
        //            for ( int i = 0; i < 5; i++ )
        //                Utils::Logger::info( std::format( "  top{}: token={} logit={:.4f}",
        //                    i, top5_logits[ i ].second, top5_logits[ i ].first ) );

        //            int32_t next_token = sampleToken(
        //                last_logits,
        //                static_cast<size_t>( config_.getVocabSize() ),
        //                0.0f, //temperature,
        //                1, // top_k,
        //                rng );

        //            tokens.push_back( next_token );

        //            if ( next_token == 50256 )
        //            {
        //                break;
        //            }
        //        }
        //    }
        //    catch ( ... )
        //    {
        //        this->setTraining( was_training );
        //        resetKVCache();
        //        throw;
        //    }

        //    resetKVCache();
        //    this->setTraining( was_training );

        //    return tokens;
        //}

        // ====================================================================
        // Accessors
        // ====================================================================

        const GptConfig& getConfig() const noexcept
        {
            return config_;
        }

        DeviceId getDeviceId() const noexcept
        {
            return network_->getExecutionContext()->getDeviceId();
        }

        // ====================================================================
        // Diagnostics
        // ====================================================================

        std::string toString() const
        {
            std::ostringstream oss;
            oss << "GptModel\n";
            oss << "Device: " << getDeviceId().toString() << "\n";
            oss << "Vocabulary: " << config_.getVocabSize() << " tokens\n";
            oss << "Max sequence length: " << config_.getMaxSequenceLength() << "\n";
            oss << "Embedding dim: " << config_.getEmbeddingSize() << "\n";
            oss << "Layers: " << config_.getNumLayers() << "\n";
            oss << "Heads: " << config_.getNumHeads() << "\n";
            oss << "MLP hidden dim: " << config_.getHiddenSize() << "\n";
            return oss.str();
        }

    private:

        explicit GptModel(
            std::unique_ptr<GptTransformerType> network,
            const GptConfig& config )
            : network_( std::move( network ) )
            , config_( config )
        {}

        std::unique_ptr<GptTransformerType> network_;
        GptConfig config_;

        // REVIEW: Should come from tokenizer metadata when tokenizer support added.
        static constexpr int32_t eos_token_ = 50256;  // GPT-2 <|endoftext|>

        // ====================================================================
        // Generation helpers
        // ====================================================================

        void truncateIfNeeded( std::vector<int32_t>& tokens ) const
        {
            int64_t seq_len = static_cast<int64_t>(tokens.size());
            if ( seq_len > config_.getMaxSequenceLength() )
            {
                Utils::Logger::warning( std::format(
                    "GptModel: sequence length {} exceeds max {}, truncating from start",
                    seq_len, config_.getMaxSequenceLength() ) );

                tokens.erase( tokens.begin(),
                    tokens.begin() + (seq_len - config_.getMaxSequenceLength()) );
            }
        }

        TokenIndexType makeTokenTensor( const std::vector<int32_t>& token_ids ) const
        {
            shape_t shape = { 1, static_cast<int64_t>(token_ids.size()) };
            TokenIndexType device_tensor( getDeviceId(), shape );
            Tensor<dtype_t::INT32, CpuMemoryResource> cpu_tensor( Device::Cpu(), shape );
            std::memcpy( cpu_tensor.data(), token_ids.data(),
                token_ids.size() * sizeof( int32_t ) );
            copy( cpu_tensor, device_tensor );
            return device_tensor;
        }

        int32_t sampleFromLogits(
            const TensorType& logits,
            int64_t position,
            float temperature,
            int top_k,
            std::mt19937& rng ) const
        {
            int64_t seq_len = logits.shape()[ 1 ];
            shape_t shape = { 1, seq_len, config_.getVocabSize() };
            Tensor<TPrecision, CpuMemoryResource> cpu( Device::Cpu(), shape );
            copy( logits, cpu );

            const float* last = cpu.data()
                + static_cast<size_t>(position) * config_.getVocabSize();

            return sampleToken( last,
                static_cast<size_t>(config_.getVocabSize()),
                temperature, top_k, rng );
        }

        static int32_t sampleToken(
            const float* logits,
            size_t vocab_size,
            float temperature,
            int top_k,
            std::mt19937& rng )
        {
            if ( temperature <= 0.0f || top_k == 1 )
                return static_cast<int32_t>(
                    std::max_element( logits, logits + vocab_size ) - logits);

            float max_logit = *std::max_element( logits, logits + vocab_size );

            std::vector<float> probs( vocab_size );
            double sum = 0.0;
            for ( size_t i = 0; i < vocab_size; ++i )
            {
                float v = std::exp( (logits[ i ] - max_logit) / temperature );
                probs[ i ] = v;
                sum += v;
            }
            for ( size_t i = 0; i < vocab_size; ++i )
                probs[ i ] /= static_cast<float>( sum );

            if ( top_k > 0 && top_k < static_cast<int>( vocab_size ) )
            {
                std::vector<size_t> indices( vocab_size );
                std::iota( indices.begin(), indices.end(), 0 );
                std::partial_sort( indices.begin(), indices.begin() + top_k,
                    indices.end(),
                    [&]( size_t a, size_t b ) { return probs[ a ] > probs[ b ]; } );

                std::vector<float> filtered( vocab_size, 0.0f );
                double filtered_sum = 0.0;
                for ( int i = 0; i < top_k; ++i )
                {
                    filtered[ indices[ i ] ] = probs[ indices[ i ] ];
                    filtered_sum += probs[ indices[ i ] ];
                }
                for ( size_t i = 0; i < vocab_size; ++i )
                    probs[ i ] = filtered[ i ] / static_cast<float>( filtered_sum );
            }

            std::uniform_real_distribution<float> dist( 0.0f, 1.0f );
            float r = dist( rng );
            float cumsum = 0.0f;
            for ( size_t i = 0; i < vocab_size; ++i )
            {
                cumsum += probs[ i ];
                if ( r < cumsum )
                    return static_cast<int32_t>( i );
            }

            return static_cast<int32_t>( vocab_size - 1 );
        }

        // ====================================================================
        // KV Caching
        // ====================================================================

        //void initializeKVCache( int64_t max_seq_len )
        //{
        //    if ( !this->isBuilt() )
        //    {
        //        throw std::runtime_error( "GptTransformer must be built before initializeKVCache()." );
        //    }

        //    for ( auto& block : transformer_blocks_ )
        //    {
        //        block->initializeKVCache( max_seq_len );
        //    }
        //}

        //void resetKVCache()
        //{
        //    for ( auto& block : transformer_blocks_ )
        //    {
        //        block->resetKVCache();
        //    }
        //}

        //TensorType& forwardPrefill( const TokenIndexType& input )
        //{
        //    if ( !this->isBuilt() )
        //    {
        //        throw std::runtime_error( "GptTransformer must be built before calling forwardPrefill()." );
        //    }

        //    encoder_out_ptr_ = &encoder_->forward( input );
        //    this->getExecutionContext()->synchronize();

        //    block_input_ptrs_[ 0 ] = encoder_out_ptr_;

        //    for ( size_t i = 0; i < transformer_blocks_.size(); ++i )
        //    {
        //        auto& block_out = transformer_blocks_[ i ]->forwardPrefill( *block_input_ptrs_[ i ] );
        //        this->getExecutionContext()->synchronize();

        //        block_output_ptrs_[ i ] = &block_out;

        //        if ( i + 1 < transformer_blocks_.size() )
        //        {
        //            block_input_ptrs_[ i + 1 ] = &block_out;
        //        }
        //    }

        //    normalized_ptr_ = &final_layernorm_->forward( *block_output_ptrs_.back() );
        //    this->getExecutionContext()->synchronize();

        //    auto host = toHost<TensorDataType::FP32>( *normalized_ptr_ );
        //    Utils::Logger::info( std::format( "ln_final out pos 0 elem 0: {:.4f}",
        //        host.data()[ 0 ] ) );
        //    Utils::Logger::info( std::format( "ln_final out pos 3 elem 0: {:.4f}",
        //        host.data()[ 3 * 768 ] ) );

        //    logits_ptr_ = &lm_head_->forward( *normalized_ptr_ );
        //    this->getExecutionContext()->synchronize();

        //    return *logits_ptr_;
        //}

        //TensorType& forwardDecode( const TokenIndexType& input, int position )
        //{
        //    if ( !this->isBuilt() )
        //    {
        //        throw std::runtime_error( "GptTransformer must be built before calling forwardDecode()." );
        //    }

        //    // DEBUG
        //    auto input_cpu = toHost<dtype_t::INT32>( input );
        //    Utils::Logger::info( std::format( "forwardDecode: input token id = {}",
        //        input_cpu.data()[ 0 ] ) );
        //    // END DEBUG

        //    encoder_out_ptr_ = &encoder_->forward( input );
        //    this->getExecutionContext()->synchronize();

        //    block_input_ptrs_[ 0 ] = encoder_out_ptr_;

        //    for ( size_t i = 0; i < transformer_blocks_.size(); ++i )
        //    {
        //        auto& block_out = transformer_blocks_[ i ]->forwardDecode( *block_input_ptrs_[ i ], position );
        //        this->getExecutionContext()->synchronize();

        //        block_output_ptrs_[ i ] = &block_out;

        //        if ( i + 1 < transformer_blocks_.size() )
        //        {
        //            block_input_ptrs_[ i + 1 ] = &block_out;
        //        }
        //    }

        //    normalized_ptr_ = &final_layernorm_->forward( *block_output_ptrs_.back() );
        //    this->getExecutionContext()->synchronize();

        //    logits_ptr_ = &lm_head_->forward( *normalized_ptr_ );
        //    this->getExecutionContext()->synchronize();

        //    return *logits_ptr_;
        //}

        // ====================================================================
        // Config helpers
        // ====================================================================

        static GptConfig configFromMetadata( const PretrainedMetadata& metadata )
        {
            GptConfig config(
                static_cast<dim_t>(metadata.embedding_dim),
                static_cast<dim_t>(metadata.num_layers) );

            config.withVocabSize( static_cast<dim_t>(metadata.vocab_size) )
                .withMaxSequenceLength( static_cast<dim_t>(metadata.max_seq_length) )
                .withNumHeads( static_cast<dim_t>(metadata.num_heads) )
                .withHiddenSize( static_cast<dim_t>(metadata.hidden_dim) )
                .withBias( metadata.use_bias );

            return config;
        }
    };
}
