/**
 * @file LlamaModel.ixx
 * @brief LLaMA inference model.
 *
 * Inference-only wrapper around a loaded LlamaTransformer network.
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

export module Dnn.Models.LlamaModel;

import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.Components.LlamaTransformer;
import Compute.Precision;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.CpuMemoryResource;
import Compute.ExecutionContextFactory;
import Serialization.PretrainedReader;
import Utils.Logger;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;

    /**
     * @brief LLaMA inference model.
     *
     * Owns a loaded, built LlamaTransformer and exposes generate() for
     * autoregressive text generation. Supports the prefill + KV-cache decode
     * two-phase generation loop identical in structure to GptModel.
     *
     * Construction is only possible via fromPretrained(). The network is always
     * in a built, weights-loaded, inference-mode state when generate() is called.
     *
     * Thread safety: not thread-safe; external synchronization required if shared.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class LlamaModel
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;
        using TokenIndexType = Tensor<dtype_t::INT32, MR>;
        using LlamaTransformerType = LlamaTransformer<TDeviceType, TPrecision>;

        LlamaModel( const LlamaModel& ) = delete;
        LlamaModel& operator=( const LlamaModel& ) = delete;
        LlamaModel( LlamaModel&& ) = default;
        LlamaModel& operator=( LlamaModel&& ) = default;

        ~LlamaModel() = default;

        // ====================================================================
        // Factory — the sole construction path
        // ====================================================================

        /**
         * @brief Load from third-party pretrained weights.
         *
         * Reads a Mila-compatible pretrained artifact (e.g. converted from a
         * HuggingFace LLaMA checkpoint) via PretrainedModelReader. The network
         * is built at max sequence length so RoPE embeddings cover the full range.
         *
         * @param path      Path to the pretrained artifact.
         * @param device_id Target device; must match TDeviceType.
         * @param strict    If true, throws on any unrecognized parameter name.
         * @return          Inference-ready LlamaModel.
         * @throws std::invalid_argument on device type mismatch.
         * @throws std::runtime_error    on load or parameter binding failure.
         */
        static std::unique_ptr<LlamaModel> fromPretrained(
            const std::filesystem::path& path,
            DeviceId device_id = DeviceId{ TDeviceType, 0 },
            bool strict = true )
        {
            if ( device_id.type != TDeviceType )
                throw std::invalid_argument( std::format(
                    "LlamaModel::fromPretrained: device type mismatch: expected {}, got {}",
                    deviceTypeToString( TDeviceType ),
                    deviceTypeToString( device_id.type ) ) );

            PretrainedModelReader reader( path );
            const auto& metadata = reader.getPretrainedMetadata();

            LlamaConfig config = configFromMetadata( metadata );

            auto network = std::make_unique<LlamaTransformerType>(
                metadata.model_name, config, device_id );

            network->build( shape_t{ 1, config.getMaxSequenceLength() } );
            network->setTraining( false );
            network->loadParameters( reader, strict );

            return std::unique_ptr<LlamaModel>( new LlamaModel( std::move( network ), config ) );
        }

        // ====================================================================
        // Inference API
        // ====================================================================

        /**
         * @brief Autoregressively generate tokens from a prompt.
         *
         * Phase 1 (prefill): runs the full prompt through forward() to populate
         * the KV cache and samples the first new token from the last position.
         * Phase 2 (decode): iterates decode() one token at a time until
         * max_new_tokens is reached or the EOS token is emitted.
         *
         * @param prompt_tokens  Input token ids; truncated from the start if
         *                       they exceed the model's max sequence length.
         * @param max_new_tokens Maximum number of tokens to generate beyond the prompt.
         * @param temperature    Sampling temperature; <= 0 selects the argmax.
         * @param top_k          Restrict sampling to the top-k logits; 0 disables.
         * @return               Full token sequence including the original prompt.
         */
        std::vector<int32_t> generate(
            const std::vector<int32_t>& prompt_tokens,
            size_t max_new_tokens = 64,
            float temperature = 1.0f,
            int top_k = 0 )
        {
            std::vector<int32_t> tokens = prompt_tokens;
            std::mt19937 rng( std::chrono::high_resolution_clock::now()
                .time_since_epoch().count() );

            truncateIfNeeded( tokens );
            int64_t seq_len = static_cast<int64_t>(tokens.size());

            auto prefill_input = makeTokenTensor( tokens );
            auto& logits = network_->forward( prefill_input );
            network_->getExecutionContext()->synchronize();

            int32_t next_token = sampleFromLogits(
                logits, seq_len - 1, temperature, top_k, rng );
            tokens.push_back( next_token );

            if ( next_token == eos_token_ )
                return tokens;

            int position = static_cast<int>(seq_len);

            for ( size_t step = 1; step < max_new_tokens; ++step )
            {
                auto decode_input = makeTokenTensor( { next_token } );
                auto& decode_logits = network_->decode( decode_input, position );
                network_->getExecutionContext()->synchronize();

                next_token = sampleFromLogits(
                    decode_logits, 0, temperature, top_k, rng );
                tokens.push_back( next_token );
                ++position;

                if ( next_token == eos_token_ )
                    break;
            }

            return tokens;
        }

        // ====================================================================
        // Accessors
        // ====================================================================

        const LlamaConfig& getConfig() const noexcept
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
            oss << "LlamaModel\n";
            oss << "Device: " << getDeviceId().toString() << "\n";
            oss << "Vocabulary: " << config_.getVocabSize() << " tokens\n";
            oss << "Max sequence length: " << config_.getMaxSequenceLength() << "\n";
            oss << "Embedding dim: " << config_.getModelDim() << "\n";
            oss << "Layers: " << config_.getNumLayers() << "\n";
            oss << "Heads: " << config_.getNumHeads() << "\n";
            oss << "KV heads: " << config_.getNumKVHeads() << "\n";
            oss << "MLP hidden dim: " << config_.getHiddenDimension() << "\n";
            oss << "RoPE theta: " << config_.getRoPETheta() << "\n";
            return oss.str();
        }

    private:

        explicit LlamaModel(
            std::unique_ptr<LlamaTransformerType> network,
            const LlamaConfig& config )
            : network_( std::move( network ) )
            , config_( config )
        {}

        std::unique_ptr<LlamaTransformerType> network_;
        LlamaConfig config_;

        // LLaMA 2 </s> = 2; LLaMA 3 <|end_of_text|> = 128001.
        // Should be sourced from tokenizer metadata once tokenizer integration is added.
        static constexpr int32_t eos_token_ = 2;

        // ====================================================================
        // Generation helpers
        // ====================================================================

        void truncateIfNeeded( std::vector<int32_t>& tokens ) const
        {
            int64_t seq_len = static_cast<int64_t>(tokens.size());

            if ( seq_len > config_.getMaxSequenceLength() )
            {
                Utils::Logger::warning( std::format(
                    "LlamaModel: sequence length {} exceeds max {}, truncating from start",
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

            const float* row = cpu.data()
                + static_cast<size_t>(position) * static_cast<size_t>(config_.getVocabSize());

            return sampleToken( row,
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
        // Config helpers
        // ====================================================================

        static LlamaConfig configFromMetadata( const PretrainedMetadata& metadata )
        {
            LlamaConfig config(
                static_cast<dim_t>(metadata.embedding_dim),
                static_cast<dim_t>(metadata.num_layers) );

            config.withVocabularyLength( static_cast<dim_t>(metadata.vocab_size) )
                .withMaxSequenceLength( static_cast<dim_t>(metadata.max_seq_length) )
                .withNumHeads( static_cast<dim_t>(metadata.num_heads) )
                .withNumKVHeads( static_cast<dim_t>(metadata.num_kv_heads) )
                .withHiddenDimension( static_cast<dim_t>(metadata.hidden_dim) )
                .withRoPETheta( metadata.rope_theta )
                .withBias( metadata.use_bias );

            return config;
        }
    };
}