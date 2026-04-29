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
#include <type_traits>
#include <cstdint>
#include <optional>
#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <format>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <functional>
#include <stop_token>

export module Dnn.Models.LlamaModel;

import Dnn.LanguageModel;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.Component;
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
     * @brief LLaMA 3 compatible inference model.
     *
     * Owns a loaded, built LlamaTransformer and exposes generateStreaming()
     * for autoregressive text generation. Supports the prefill + KV-cache
     * decode two-phase generation loop.
     *
     * Construction is only possible via fromPretrained(). The network is always
     * in a built, weights-loaded, inference-mode state when generation is called.
     *
     * Thread safety: not thread-safe; external synchronization required if shared.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class LlamaModel : public LanguageModel<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using ModelBase = LanguageModel<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using TokenIndexType = Tensor<dtype_t::INT32, MR>;
        using LlamaTransformerType = LlamaTransformer<TDeviceType, TPrecision>;

        LlamaModel( const LlamaModel& ) = delete;
        LlamaModel& operator=( const LlamaModel& ) = delete;
        LlamaModel( LlamaModel&& ) = default;
        LlamaModel& operator=( LlamaModel&& ) = default;

        ~LlamaModel() = default;

        /**
         * @brief Load from third-party pretrained weights.
         *
         * Reads a Mila-compatible pretrained artifact (e.g. converted from a
         * HuggingFace LLaMA checkpoint) via PretrainedModelReader. The network
         * is built at max sequence length so RoPE embeddings cover the full range.
         *
         * @param path           Path to the pretrained artifact.
         * @param context_length Maximum sequence length to build for.
         * @param device_id      Target device; must match TDeviceType.
         * @param strict         If true, throws on any unrecognized parameter name.
         * @return               Inference-ready LlamaModel.
         *
         * @throws std::invalid_argument on device type mismatch.
         * @throws std::runtime_error    on load or parameter binding failure.
         */
        static std::unique_ptr<LlamaModel> fromPretrained(
            const std::filesystem::path& path,
            dim_t context_length,
            DeviceId device_id = DeviceId{ TDeviceType, 0 },
            bool strict = true )
        {
            if ( device_id.type != TDeviceType )
            {
                throw std::invalid_argument( std::format(
                    "LlamaModel::fromPretrained: device type mismatch: expected {}, got {}",
                    deviceTypeToString( TDeviceType ),
                    deviceTypeToString( device_id.type ) ) );
            }

            PretrainedModelReader reader( path );
            const auto& metadata = reader.getPretrainedMetadata();

            LlamaConfig config = configFromMetadata( metadata );

            auto network = std::make_unique<LlamaTransformerType>( metadata.model_name, config, device_id );

            BuildContext build_context( shape_t{ 1, context_length }, RuntimeMode::Inference, 0, false );

            network->build( build_context );

            auto stats = network->getMemoryStats();
            std::cout << "\n--- Memory after build() ---\n";
            std::cout << stats.toString() << std::endl;

            network->loadParameters( reader, strict );

            return std::unique_ptr<LlamaModel>(
                new LlamaModel( std::move( network ), config, RuntimeMode::Inference ) );
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
            return getNetwork().getExecutionContext()->getDeviceId();
        }

        MemoryStats getMemoryStats() const
        {
            return getNetwork().getMemoryStats();
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

        // ====================================================================
        // Profiling
        // ====================================================================

        void profilePrefill( const std::vector<int32_t>& token_ids )
        {
            auto input = makeTokenTensor( token_ids );
            getNetwork().prefill( input );
            getNetwork().getExecutionContext()->synchronize();
        }

    protected:

        // ====================================================================
        // LanguageModel overrides
        // ====================================================================

        /**
         * @brief Prefill + KV-cache decode loop with per-token streaming.
         *
         * Phase 1 (prefill): runs the full prompt through prefill() to populate
         * the KV cache and samples the first new token from the last position.
         * Phase 2 (decode): iterates one token at a time until max_new_tokens
         * is reached, EOS is emitted, or stop is requested.
         *
         * on_token is called for every generated token except EOS.
         *
         * @param prompt_tokens  Input token ids; truncated from the start if
         *                       they exceed the model's max sequence length.
         * @param on_token       Callback invoked once per generated token (not EOS).
         * @param max_new_tokens Maximum number of tokens to generate beyond the prompt.
         * @param temperature    Sampling temperature; <= 0 selects the argmax.
         * @param top_k          Restrict sampling to the top-k logits; 0 disables.
         * @param stop           Stop token for cooperative cancellation.
         */
        void generateStreaming(
            const std::vector<int32_t>& prompt_tokens,
            const std::function<void( int32_t )>& on_token,
            size_t max_new_tokens,
            float temperature,
            int top_k,
            std::stop_token stop ) override
        {
            const auto stop_ids = stopTokens();

            std::vector<int32_t> prefill_tokens = prompt_tokens;
            std::mt19937 rng( std::chrono::high_resolution_clock::now()
                .time_since_epoch().count() );

            truncateIfNeeded( prefill_tokens );
            int64_t seq_len = static_cast<int64_t>(prefill_tokens.size());

            auto prefill_input = makeTokenTensor( prefill_tokens );
            auto& logits = getNetwork().prefill( prefill_input );
            getNetwork().getExecutionContext()->synchronize();

            int32_t next_token = sampleFromLogits( logits, 0, temperature, top_k, rng );

            if ( stop_ids.contains( next_token ) )
                return;

            on_token( next_token );

            int position = static_cast<int>(seq_len);

            for ( size_t step = 1; step < max_new_tokens; ++step )
            {
                if ( stop.stop_requested() ) break;

                auto decode_input = makeTokenTensor( { next_token } );
                auto& decode_logits = getNetwork().decode( decode_input, position );
                getNetwork().getExecutionContext()->synchronize();

                next_token = sampleFromLogits( decode_logits, 0, temperature, top_k, rng );

                if ( stop_ids.contains( next_token ) ) break;

                on_token( next_token );
                ++position;
            }
        }

        /**
         * @brief Maximum sequence length from LLaMA config.
         */
        int64_t maxSequenceLength() const noexcept override
        {
            return static_cast<int64_t>(config_.getMaxSequenceLength());
        }

        /**
         * @brief Vocabulary size from LLaMA config.
         */
        int64_t vocabSize() const noexcept override
        {
            return static_cast<int64_t>(config_.getVocabSize());
        }

        /**
         * @brief Training loop — not yet implemented for LlamaModel.
         *
         * @throws std::runtime_error always.
         */
        void onTraining() override
        {
            throw std::runtime_error(
                "LlamaModel::onTraining: training not yet implemented" );
        }

    private:

        explicit LlamaModel(
            std::unique_ptr<LlamaTransformerType> network,
            const LlamaConfig& config,
            RuntimeMode runtime_mode )
            : ModelBase( std::move( network ), runtime_mode ), config_( config )
        {}

        LlamaTransformerType& getNetwork() noexcept
        {
            return static_cast<LlamaTransformerType&>(*ModelBase::network_);
        }

        const LlamaTransformerType& getNetwork() const noexcept
        {
            return static_cast<const LlamaTransformerType&>(*ModelBase::network_);
        }

        LlamaConfig config_;
        std::unique_ptr<TensorType> prefill_scratch_;
        dim_t context_length_;
        int64_t prefill_chunk_size_{ 64 };

        // LLaMA 2 </s> = 2; LLaMA 3 <|end_of_text|> = 128001.
        // Should be sourced from tokenizer metadata once tokenizer integration is added.
        // DEPRECATED: static constexpr int32_t eos_token_ = 128001;

        /**
         * @brief LLaMA 3.x end-of-sequence token.
         * <|end_of_text|> = 128001.
         */
        int32_t eosToken() const noexcept override
        {
            return 128001;
        }

        /**
         * @brief Llama 3.x generation stop tokens.
         *
         * Halts on <|end_of_text|> (128001), <|eot_id|> (128009),
         * and <|eom_id|> (128008). The latter two are the primary
         * turn and tool-call boundaries in instruct-format generation.
         */
        std::unordered_set<int32_t> stopTokens() const override
        {
            return { 128001, 128009, 128008 };
        }

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
            Tensor<TensorDataType::FP32, CpuMemoryResource> cpu( Device::Cpu(), shape );
            copy( logits, cpu );

            const float* row = cpu.data()
                + static_cast<size_t>(position) * static_cast<size_t>(config_.getVocabSize());

            return sampleToken(
                row,
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
            {
                return static_cast<int32_t>( std::max_element( logits, logits + vocab_size ) - logits );
            }

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