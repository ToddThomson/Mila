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
#include <cstdint>
#include <stdexcept>
#include <filesystem>
#include <format>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <functional>
#include <stop_token>
#include <cstring>
#include <type_traits>

export module Dnn.Models.LlamaModel;

import Dnn.Models.LlamaModelConfig;
import Dnn.LanguageModel;
import Dnn.LanguageNetwork;
import Dnn.Quantization.Weight.Policies;
import Dnn.Quantization.KvCache.Policy;
import Dnn.Quantization.KvCache.QuantPolicy;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.Component;
import Dnn.Components.LlamaTransformer;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.CpuMemoryResource;
import Compute.CudaPinnedMemoryResource;
import Compute.ExecutionContextFactory;
import Serialization.PretrainedReader;
import Logging.Logger;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;
    using namespace Mila::Dnn::Quant::Weight;
    using namespace Mila::Dnn::Quant::KvCache;

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
        using StagingMR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaPinnedMemoryResource, CpuMemoryResource>;

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
         * is built at the context length specified in model_config so RoPE
         * embeddings and KV cache buffers cover the full range.
         *
         * The model_config carries all deployment decisions:
         *   - context_length     — maximum sequence length to build for
         *   - weight_quantization — compile-time dispatch to quantized or BF16 path
         *   - kv_cache_compression — compile-time dispatch to KV cache policy
         *
         * @param path          Path to the pretrained Llama model artifact.
         * @param model_config  Deployment configuration for this load.
         * @param device_id     Target device; must match TDeviceType.
         * @return              Inference-ready LlamaModel.
         *
         * @throws std::invalid_argument on device type mismatch or zero context length.
         * @throws std::runtime_error    on load or parameter binding failure.
         * @throws std::runtime_error    if model_config requests unsupported quantization (e.g. FP4).
         */
        static std::unique_ptr<LlamaModel<TDeviceType, TPrecision>> fromPretrained(
            const std::filesystem::path& path,
            const LlamaModelConfig& model_config,
            DeviceId device_id = DeviceId{ TDeviceType, 0 } )
        {
            if ( device_id.type != TDeviceType )
            {
                throw std::invalid_argument( std::format(
                    "LlamaModel::fromPretrained: device type mismatch: expected {}, got {}",
                    deviceTypeToString( TDeviceType ),
                    deviceTypeToString( device_id.type ) ) );
            }

            if ( model_config.getContextLength() == 0 )
            {
                throw std::invalid_argument(
                    "LlamaModel::fromPretrained: context_length must be greater than zero" );
            }

            // Runtime → compile-time bridge: dispatch on ModelConfig quantization settings.
            switch ( model_config.getWeightQuantization() )
            {
                case WeightQuantization::FP4:
                    // PerGroupInt4<128>: packed INT4 weights, per-group float32 scales,
                    // optional INT4 zero points. W4A16 kernel path (cuda_w4a16_gemm).
                    // KV cache compression is not paired with INT4 weights — it is handled
                    // independently; only None is supported for now.
                    switch ( model_config.getKvCacheCompression() )
                    {
                        case KvCacheCompression::FP8:
                            // FP8 KV cache with INT4 weights — not yet supported.
                        case KvCacheCompression::None:
                            if constexpr ( TPrecision == TensorDataType::BF16 )
                            {
                                return fromPretrainedImpl<PerGroupInt4<128>, NoKvCompression>( path, model_config, device_id );
                            }
                            else
                            {
                                throw std::runtime_error(
                                    "LlamaModel::fromPretrained: FP4 weight quantization requires BF16 compute precision" );
                            }
                    }
                    break;

                case WeightQuantization::FP8:
                    switch ( model_config.getKvCacheCompression() )
                    {
                        case KvCacheCompression::FP8:
                            // FP8 KV cache compression requires CudaGqaOp FP8 support — not yet implemented.
                        case KvCacheCompression::None:
                            if constexpr ( TPrecision == TensorDataType::BF16 )
                            {
                                return fromPretrainedImpl<PerChannelFp8<>, NoKvCompression>( path, model_config, device_id );
                            }
                            else
                            {
                                throw std::runtime_error(
                                    "LlamaModel::fromPretrained: FP8 weight quantization requires BF16 compute precision" );
                            }
                    }
                    break;

                case WeightQuantization::None:
                default:
                    switch ( model_config.getKvCacheCompression() )
                    {
                        case KvCacheCompression::FP8:
                            throw std::runtime_error(
                                "LlamaModel::fromPretrained: FP8 KV cache compression is not yet supported" );
                        case KvCacheCompression::None:
                            return fromPretrainedImpl<NoWeightQuant, NoKvCompression>( path, model_config, device_id );
                    }
                    break;
            }

            // Unreachable — all enum cases handled above.
            throw std::runtime_error( "LlamaModel::fromPretrained: unhandled quantization configuration" );
        }

        // ====================================================================
        // Accessors
        // ====================================================================

        const LlamaConfig& getConfig() const noexcept
        {
            return config_;
        }

        // ====================================================================
        // Diagnostics
        // ====================================================================

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "LlamaModel\n";
            oss << "Device: " << this->getDeviceId().toString() << "\n";
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
            this->getLanguageNetwork().prefill( input );
            this->getLanguageNetwork().getExecutionContext()->synchronize();
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
        void onGenerating(
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

            // ---- Phase 1: Prefill — measure to first token ----
            // synchronize() is called after prefill(), so the wall-clock measurement
            // accurately captures GPU prefill time plus first token sampling overhead.

            const auto prefill_start = std::chrono::high_resolution_clock::now();

            auto& logits = this->getLanguageNetwork().prefill( prefill_input );
            this->getLanguageNetwork().synchronize();

            int32_t next_token = sampleFromLogits( logits, 0, temperature, top_k, rng );

            const auto prefill_end = std::chrono::high_resolution_clock::now();

            // Reset statistics for this generation run.
            this->last_generation_statistics_.prompt_tokens    = prefill_tokens.size();
            this->last_generation_statistics_.tokens_generated = 0;
            this->last_generation_statistics_.prefill_time_ms  =
                std::chrono::duration<float, std::milli>( prefill_end - prefill_start ).count();
            this->last_generation_statistics_.decode_time_ms               = 0.0f;
            this->last_generation_statistics_.decode_tokens_per_second     = 0.0f;

            if ( stop_ids.contains( next_token ) )
                return;

            on_token( next_token );
            this->last_generation_statistics_.tokens_generated = 1;

            int position = static_cast<int>(seq_len);

            // ---- Phase 2: Autoregressive decode ----
            // Each decode step calls synchronize(), so wall-clock time correctly
            // reflects GPU decode latency per token.

            const auto decode_start = std::chrono::high_resolution_clock::now();
            std::size_t decode_token_count = 0;

            for ( size_t step = 1; step < max_new_tokens; ++step )
            {
                if ( stop.stop_requested() )
                    break;

                decode_token_staging_.data()[ 0 ] = next_token;
                copy( decode_token_staging_, decode_token_device_ );

                auto& decode_logits = this->getLanguageNetwork().decode( decode_token_device_, position );
                this->getLanguageNetwork().synchronize();

                next_token = sampleFromLogits( decode_logits, 0, temperature, top_k, rng );

                if ( stop_ids.contains( next_token ) ) break;

                on_token( next_token );
                ++position;
                ++decode_token_count;
            }

            const auto decode_end = std::chrono::high_resolution_clock::now();
            const float decode_ms =
                std::chrono::duration<float, std::milli>( decode_end - decode_start ).count();

            this->last_generation_statistics_.tokens_generated            = 1 + decode_token_count;
            this->last_generation_statistics_.decode_time_ms              = decode_ms;
            this->last_generation_statistics_.decode_tokens_per_second    =
                (decode_ms > 0.0f && decode_token_count > 0)
                ? static_cast<float>( decode_token_count ) / (decode_ms / 1000.0f)
                : 0.0f;
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

    private:

        explicit LlamaModel(
            std::unique_ptr<LanguageNetwork<TDeviceType, TPrecision>> network,
            const LlamaConfig& config,
            RuntimeMode runtime_mode )
            : ModelBase( std::move( network ), runtime_mode )
            , config_( config )
            , decode_token_staging_( TDeviceType == DeviceType::Cuda ? this->getDeviceId() : Device::Cpu(), shape_t{ 1, 1 } )
            , decode_token_device_( this->getDeviceId(), shape_t{ 1, 1 } )
            , logits_staging_( TDeviceType == DeviceType::Cuda ? this->getDeviceId() : Device::Cpu(), shape_t{ 1, 1, static_cast<int64_t>( config.getVocabSize() ) } )
        {}

        template<WeightQuantPolicy TWeightQuantization, KvCachePolicy TKvCachePolicy>
        static std::unique_ptr<LlamaModel<TDeviceType, TPrecision>> fromPretrainedImpl(
            const std::filesystem::path& path,
            const LlamaModelConfig& model_config,
            DeviceId device_id )
        {
            PretrainedModelReader reader( path );
            const auto& metadata = reader.getPretrainedMetadata();

            LlamaConfig network_config = configFromMetadata( metadata );

            if ( model_config.getContextLength() > network_config.getMaxSequenceLength() )
            {
                throw std::invalid_argument( std::format(
                    "LlamaModel::fromPretrained: context_length {} exceeds "
                    "trained max_seq_len {}",
                    model_config.getContextLength(),
                    network_config.getMaxSequenceLength() ) );
            }

            using ConcreteTransformerType = LlamaTransformer<TDeviceType, TPrecision, TWeightQuantization, TKvCachePolicy>;
            auto network = std::make_unique<ConcreteTransformerType>( metadata.model_name, network_config, device_id );

            auto context_length = model_config.getContextLength();

            BuildContext build_context(
                shape_t{ 1, context_length },
                RuntimeMode::Inference,
                false );

            network->build( build_context );

            Logging::Logger::info( network->toString() );

            network->loadParameters( reader );

            return std::unique_ptr<LlamaModel<TDeviceType, TPrecision>>(
                new LlamaModel<TDeviceType, TPrecision>(
                    std::move( network ), network_config, RuntimeMode::Inference ) );
        }

        LlamaConfig config_;
        Tensor<dtype_t::INT32, StagingMR> decode_token_staging_;
        TokenIndexType decode_token_device_;
        Tensor<TensorDataType::FP32, StagingMR> logits_staging_;

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
                Logging::Logger::warning( std::format(
                    "LlamaModel: sequence length {} exceeds max {}, truncating from start",
                    seq_len, config_.getMaxSequenceLength() ) );

                tokens.erase( tokens.begin(),
                    tokens.begin() + (seq_len - config_.getMaxSequenceLength()) );
            }
        }

        TokenIndexType makeTokenTensor( const std::vector<int32_t>& token_ids ) const
        {
            shape_t shape = { 1, static_cast<int64_t>(token_ids.size()) };
            TokenIndexType device_tensor( this->getDeviceId(), shape );

            Tensor<dtype_t::INT32, CpuMemoryResource> cpu_tensor( Device::Cpu(), shape );

            std::memcpy( cpu_tensor.data(), token_ids.data(), token_ids.size() * sizeof( int32_t ) );

            copy( cpu_tensor, device_tensor );

            return device_tensor;
        }

        int32_t sampleFromLogits(
            const TensorType& logits,
            int64_t position,
            float temperature,
            int top_k,
            std::mt19937& rng )
        {
            copy( logits, logits_staging_ );

            const float* row = logits_staging_.data()
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
