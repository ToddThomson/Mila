/**
 * @file GemmaModel.ixx
 * @brief Gemma 4 inference model.
 *
 * Inference-only wrapper around a loaded GemmaTransformer network.
 */

module;
#include <memory>
#include <vector>
#include <span>
#include <unordered_set>
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
#include <cmath>

export module Dnn.Models.GemmaModel;

import Dnn.Models.GemmaModelConfig;
import Dnn.LanguageModel;
import Dnn.LanguageModelConfig;
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
import Dnn.RuntimeMode;
import Dnn.Components.GemmaTransformer;
import Dnn.Components.GemmaConfig;
import Dnn.Samplers.TokenSampler;
import Dnn.Samplers.SamplingConfig;
import Dnn.GenerateParams;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.DeviceTypeTraits.Cpu;
import Compute.CpuMemoryResource;
#ifdef MILA_HAS_CUDA
import Compute.DeviceTypeTraits.Cuda;
import Compute.CudaPinnedMemoryResource;
#endif
import Compute.ExecutionContextFactory;
import Serialization.PretrainedReader;
import Serialization.Mode;
import Logging.Logger;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Dnn::Serialization;
    using namespace Mila::Dnn::Quant::Weight;
    using namespace Mila::Dnn::Quant::KvCache;

    /**
     * @brief A/B sampling path selector (TokenSampling.md section 8.1).
     *
     * HostA is the validated host baseline; DeviceB routes sampling through the device
     * TokenSampler. Flip to DeviceB and rebuild to compare against a HostA build via the
     * Chat harness on the sample prompt; path A is retired once B reproduces it.
     */
    enum class SamplingPath { HostA, DeviceB };
    constexpr SamplingPath kSamplingPath = SamplingPath::DeviceB;

    /**
     * @brief Gemma 4 compatible inference model.
     *
     * Owns a loaded, built GemmaTransformer and drives the prefill + KV-cache
     * decode two-phase generation loop. Construction is only possible via
     * fromPretrained(); the network is always built, weights-loaded, and in
     * inference mode when generation runs.
     *
     * Thread safety: not thread-safe; external synchronization required if shared.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class GemmaModel : public LanguageModel<TDeviceType, TPrecision>
    {
    public:
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using ModelBase = LanguageModel<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using TokenIndexType = Tensor<dtype_t::INT32, MR>;
#ifdef MILA_HAS_CUDA
        using StagingMR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaPinnedMemoryResource, CpuMemoryResource>;
#else
        using StagingMR = CpuMemoryResource;
#endif

        GemmaModel( const GemmaModel& ) = delete;
        GemmaModel& operator=( const GemmaModel& ) = delete;
        GemmaModel( GemmaModel&& ) = default;
        GemmaModel& operator=( GemmaModel&& ) = default;

        ~GemmaModel() = default;

        /**
         * @brief Load from a Mila-converted Gemma 4 pretrained artifact.
         *
         * The model_config carries the deployment decisions (context length,
         * weight quantization, KV-cache compression); every architectural
         * parameter is read from the checkpoint metadata.
         *
         * @param path          Path to the pretrained Gemma model artifact.
         * @param model_config  Deployment configuration for this load.
         * @param device_id     Target device; must match TDeviceType.
         * @return              Inference-ready GemmaModel.
         *
         * @throws std::invalid_argument on device type mismatch or zero context length.
         * @throws std::runtime_error    on load failure or unsupported quantization.
         */
        static std::unique_ptr<GemmaModel<TDeviceType, TPrecision>> fromPretrained(
            const std::filesystem::path& path,
            const GemmaModelConfig& model_config,
            DeviceId device_id = DeviceId{ TDeviceType, 0 } )
        {
            if ( device_id.type != TDeviceType )
            {
                throw std::invalid_argument( std::format(
                    "GemmaModel::fromPretrained: device type mismatch: expected {}, got {}",
                    deviceTypeToString( TDeviceType ),
                    deviceTypeToString( device_id.type ) ) );
            }

            if ( model_config.getContextLength() == 0 )
            {
                throw std::invalid_argument(
                    "GemmaModel::fromPretrained: context_length must be greater than zero" );
            }

            // Runtime -> compile-time bridge: dispatch on the ModelConfig quantization
            // settings, mirroring LlamaModel. Gemma's Linear children (qkv/o/gate_up/down)
            // pick up the weight-quant policy; lm_head stays unquantized.
            switch ( model_config.getWeightQuantization() )
            {
                case WeightQuantization::FP4:
                    switch ( model_config.getKvCacheCompression() )
                    {
                        case KvCacheCompression::FP8:
                        case KvCacheCompression::None:
                            if constexpr ( TPrecision == TensorDataType::BF16 )
                            {
                                return fromPretrainedImpl<PerGroupFp4<128>, NoKvCompression>( path, model_config, device_id );
                            }
                            else
                            {
                                throw std::runtime_error(
                                    "GemmaModel::fromPretrained: FP4 weight quantization requires BF16 compute precision" );
                            }
                    }
                    break;

                case WeightQuantization::FP8:
                    switch ( model_config.getKvCacheCompression() )
                    {
                        case KvCacheCompression::FP8:
                        case KvCacheCompression::None:
                            if constexpr ( TPrecision == TensorDataType::BF16 )
                            {
                                return fromPretrainedImpl<PerChannelFp8<>, NoKvCompression>( path, model_config, device_id );
                            }
                            else
                            {
                                throw std::runtime_error(
                                    "GemmaModel::fromPretrained: FP8 weight quantization requires BF16 compute precision" );
                            }
                    }
                    break;

                case WeightQuantization::None:
                default:
                    switch ( model_config.getKvCacheCompression() )
                    {
                        case KvCacheCompression::FP8:
                            throw std::runtime_error(
                                "GemmaModel::fromPretrained: FP8 KV cache compression is not yet supported" );
                        case KvCacheCompression::None:
                            return fromPretrainedImpl<NoWeightQuant, NoKvCompression>( path, model_config, device_id );
                    }
                    break;
            }

            throw std::runtime_error( "GemmaModel::fromPretrained: unhandled quantization configuration" );
        }

        // ====================================================================
        // Accessors
        // ====================================================================

        const GemmaConfig& getConfig() const noexcept
        {
            return config_;
        }

        // ====================================================================
        // Diagnostics
        // ====================================================================

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "GemmaModel\n";
            oss << "Device: " << this->getDeviceId().toString() << "\n";
            oss << config_.toString();

            return oss.str();
        }

        // ====================================================================
        // Profiling
        // ====================================================================

        void profilePrefill( const std::vector<int32_t>& token_ids )
        {
            auto input = makeTokenTensor( token_ids );
            this->getLanguageNetwork().prefill( input );
            this->getLanguageNetwork().synchronize();
        }

    protected:

        // REVIEW: The onGenerating() should take a GenerationConfig struct instead of a long parameter list.
        // This would make it easier to extend in the future and keep the interface clean.

        // REVIEW: The onGenerating() should return a GenerateResult struct instead of void, containing the statistics and any other relevant information.
        // See GenerateResult.ixx for a starting implementation.

        void onGenerating(
            std::span<const int32_t> prompt_tokens,
            const std::function<void( int32_t )>& on_token,
            size_t max_new_tokens,
            float temperature,
            int top_k,
            std::stop_token stop ) override
        {
            const auto stop_ids = stopTokens();

            std::mt19937 rng( std::chrono::high_resolution_clock::now().time_since_epoch().count() );

            // REVIEW: Why are we copying the prompt tokens?
            // FIX: There is no need to copy the prompt tokens. We can use the span directly.
            //std::vector<int32_t> prefill_tokens = prompt_tokens;

            // REVIEW: This is weak. Requires a more robust solution
            // truncateIfNeeded( prefill_tokens );

            // FIX: The resolution is to return an error if the prompt exceeds the deployment context length.
            // The context management is not our concern. The GemmaModel is built with a context length, and the prompt must fit within it.
            // If the prompt exceeds it, we should throw an error instead of truncating it silently.

            if ( prompt_tokens.size() > static_cast<size_t>( context_length_ ) )
            {
                throw std::invalid_argument( std::format(
                    "GemmaModel::onGenerating: prompt length {} exceeds deployment context length {}",
                    prompt_tokens.size(), context_length_ ) );
            }

            int64_t seq_len = static_cast<int64_t>(prompt_tokens.size());

            // REVIEW: We should support a vector to tensor conversion utility in the Tensor class, to avoid this copy.
            // FIX: The onGenerating() now uses a span which is supported by the Tensor::fill() method
            auto prefill_input = makeTokenTensor( prompt_tokens );

            // ---- Phase 1: Prefill — measure to first token ----
            const auto prefill_start = std::chrono::high_resolution_clock::now();

            auto& logits = this->getLanguageNetwork().prefill( prefill_input );
            this->getLanguageNetwork().synchronize();

            int32_t next_token = sampleNext( logits, temperature, top_k, rng );

            const auto prefill_end = std::chrono::high_resolution_clock::now();

            this->last_generation_statistics_.prompt_tokens    = prompt_tokens.size();
            this->last_generation_statistics_.tokens_generated = 0;
            this->last_generation_statistics_.prefill_time_ms  =
                std::chrono::duration<float, std::milli>( prefill_end - prefill_start ).count();
            this->last_generation_statistics_.decode_time_ms           = 0.0f;
            this->last_generation_statistics_.decode_tokens_per_second = 0.0f;

            if ( stop_ids.contains( next_token ) )
                return;

            on_token( next_token );
            this->last_generation_statistics_.tokens_generated = 1;

            int position = static_cast<int>(seq_len);

            // ---- Phase 2: Autoregressive decode ----
            const auto decode_start = std::chrono::high_resolution_clock::now();
            std::size_t decode_token_count = 0;

            for ( size_t step = 1; step < max_new_tokens; ++step )
            {
                if ( stop.stop_requested() )
                    break;

                // The KV cache is only as deep as the deployment context length; decode
                // cannot write at a position past it. Stop cleanly instead of letting the
                // GQA op throw "position out of range" (hit when max_new_tokens would run
                // generation beyond context_length_).
                if ( position >= context_length_ )
                    break;

                // REVIEW: Copy for a single token is wasteful. We should have a single-token tensor on the device and just write to it.
                // FIX: Once the refactor for the device DecoderOp is done, we can eliminate this copy and just use a single-token tensor on the device.
                decode_token_staging_.data()[ 0 ] = next_token;
                copy( decode_token_staging_, decode_token_device_ );

                auto& decode_logits = this->getLanguageNetwork().decode( decode_token_device_, position );
                this->getLanguageNetwork().synchronize();

                // REVIEW: The LanguageModel sampling should be done on the device, not the host.
                // This is a temporary workaround until we implement a device-side sampler.
                next_token = sampleNext( decode_logits, temperature, top_k, rng );

                if ( stop_ids.contains( next_token ) ) break;

                on_token( next_token );
                ++position;
                ++decode_token_count;
            }

            const auto decode_end = std::chrono::high_resolution_clock::now();
            const float decode_ms =
                std::chrono::duration<float, std::milli>( decode_end - decode_start ).count();

            // REVIEW: stats could be part of the returned GenerateResult struct instead of being stored in the model.

            this->last_generation_statistics_.tokens_generated         = 1 + decode_token_count;
            this->last_generation_statistics_.decode_time_ms           = decode_ms;
            this->last_generation_statistics_.decode_tokens_per_second =
                (decode_ms > 0.0f && decode_token_count > 0)
                ? static_cast<float>( decode_token_count ) / (decode_ms / 1000.0f)
                : 0.0f;
        }

        void onTraining() override
        {
            throw std::runtime_error(
                "GemmaModel::onTraining: Gemma is inference-only" );
        }

        int64_t maxSequenceLength() const noexcept override
        {
            // REVIEW: Settle this once and for all: int64_t vs dim_t. These static_casts are a code smell.
            return static_cast<int64_t>( config_.getMaxSequenceLength() );
        }

        int64_t vocabSize() const noexcept override
        {
            return static_cast<int64_t>(config_.getVocabSize());
        }

    private:

        // REVIEW: context_length_ This differs from LlamaModel. Why?
        // Hmmm, also, context_length is dependent on RuntimeMode. If we are in training mode,
        // context_length_ should be the max sequence length.
        // If we are in inference mode, context_length_ should be the deployment context length.
        // This is a subtle but important distinction.

        explicit GemmaModel(
            std::unique_ptr<LanguageNetwork<TDeviceType, TPrecision>> network,
            const GemmaConfig& config,
            int64_t context_length,
            RuntimeMode runtime_mode )
            : ModelBase( std::move( network ), runtime_mode )
            , config_( config )
            , context_length_( context_length )
            , decode_token_staging_( TDeviceType == DeviceType::Cuda ? this->getDeviceId() : Device::Cpu(), shape_t{ 1, 1 } )
            , decode_token_device_( this->getDeviceId(), shape_t{ 1, 1 } )
            , logits_staging_( TDeviceType == DeviceType::Cuda ? this->getDeviceId() : Device::Cpu(), shape_t{ 1, 1, static_cast<int64_t>( config.getVocabSize() ) } )
            , token_sampler_(
                this->getLanguageNetwork().getExecutionContext(),
                SamplingConfig{}
                    .withVocabularySize( static_cast<int64_t>( config.getVocabSize() ) )
                    .withFinalLogitSoftcap( config.getFinalLogitSoftcapping() ) )
        {}

        template<WeightQuantPolicy TWeightQuantization, KvCachePolicy TKvCachePolicy>
        static std::unique_ptr<GemmaModel<TDeviceType, TPrecision>> fromPretrainedImpl(
            const std::filesystem::path& path,
            const GemmaModelConfig& model_config,
            DeviceId device_id )
        {
            PretrainedModelReader reader( path );
            const auto& metadata = reader.getPretrainedMetadata();

            GemmaConfig network_config = configFromMetadata( metadata );

            if ( model_config.getContextLength() > network_config.getMaxSequenceLength() )
            {
                throw std::invalid_argument( std::format(
                    "GemmaModel::fromPretrained: context_length {} exceeds trained max_seq_len {}",
                    model_config.getContextLength(),
                    network_config.getMaxSequenceLength() ) );
            }

            using ConcreteTransformerType = GemmaTransformer<TDeviceType, TPrecision, TWeightQuantization, TKvCachePolicy>;
            auto network = std::make_unique<ConcreteTransformerType>( metadata.model_name, network_config, device_id );

            auto context_length = model_config.getContextLength();

            BuildContext build_context(
                shape_t{ 1, context_length },
                RuntimeMode::Inference,
                false );

            network->build( build_context );

            Logging::Logger::info( network->toString() );

            network->loadParameters( reader );

            return std::unique_ptr<GemmaModel<TDeviceType, TPrecision>>(
                new GemmaModel<TDeviceType, TPrecision>(
                    std::move( network ), network_config,
                    static_cast<int64_t>( context_length ), RuntimeMode::Inference ) );
        }

        // REVIEW: Why aren't we storing the model config instead of the network config?
        // The model config is what we use to build the network, and it contains the deployment context length.
        // The network config is what we read from the checkpoint metadata,
        // and it contains the architectural max sequence length. We should store both, or at least make sure we are using the right one in the right place.

        GemmaConfig config_;

        // Deployment KV-cache capacity: the context length the network was BUILT with
        // (model_config.getContextLength()), which may be far below the architectural
        // max (config_.getMaxSequenceLength()). Prompt truncation and the decode loop
        // bound against THIS, not the architectural max, or the GQA op throws when a
        // write position reaches the cache size.
        // REVIEW: Why isn't this a const int64_t? It is set in the constructor and never changed. It should be const.
        // REVIEW: Why isn't this a dim_t? It is set from model_config.getContextLength(), which is a dim_t. It should be a dim_t.
        // REVIEW: Why is this a member variable at all? It is only used in onGenerating(),
        // which could just read it from config_.getContextLength(). It seems like an unnecessary duplication of state.
        int64_t context_length_;

        Tensor<dtype_t::INT32, StagingMR> decode_token_staging_;
        TokenIndexType decode_token_device_;
        Tensor<TensorDataType::FP32, StagingMR> logits_staging_;
        TokenSampler<TDeviceType, TPrecision> token_sampler_;

        /**
         * @brief Gemma end-of-sequence token: <eos> = 1.
         *
         * REVIEW: confirm against the Gemma 4 tokenizer config in 5e/5f — the
         * Gemma family uses <eos>=1 and <end_of_turn> as the instruct turn
         * boundary; the exact <end_of_turn> id is verified when the tokenizer
         * is converted.
         */
        int32_t eosToken() const noexcept override
        {
            return 1;
        }

        /**
         * @brief Gemma generation stop tokens: <eos> (1) and <end_of_turn> (106).
         *
         * REVIEW: <end_of_turn> id pending tokenizer-config confirmation (see eosToken).
         */
        std::unordered_set<int32_t> stopTokens() const override
        {
            return { 1, 106 };
        }

        // ====================================================================
        // Generation helpers
        // ====================================================================

        //void truncateIfNeeded( std::vector<int32_t>& tokens ) const
        //{
        //    int64_t seq_len = static_cast<int64_t>(tokens.size());

        //    // Bound the prompt by the deployment context (the KV-cache depth), not the
        //    // architectural max: prefill writes positions 0..seq_len-1 into a cache of
        //    // size context_length_, so a longer prompt overflows it.
        //    if ( seq_len > context_length_ )
        //    {
        //        Logging::Logger::warning( std::format(
        //            "GemmaModel: sequence length {} exceeds deployment context {}, truncating from start",
        //            seq_len, context_length_ ) );

        //        tokens.erase( tokens.begin(),
        //            tokens.begin() + (seq_len - context_length_) );
        //    }
        //}

        TokenIndexType makeTokenTensor( std::span<const int32_t> token_ids ) const
        {
            // REVIEW: The inference path data movement is suboptimal:
            // the token_ids vector is copied to a CPU tensor, then copied to the device tensor.
            // Ideally, we would construct the device tensor directly from the vector without an intermediate copy.

            shape_t shape = { 1, static_cast<int64_t>( token_ids.size() ) };
            TokenIndexType device_tensor( this->getDeviceId(), shape );

            Tensor<dtype_t::INT32, CpuMemoryResource> cpu_tensor( Device::Cpu(), shape );

            std::memcpy( cpu_tensor.data(), token_ids.data(), token_ids.size() * sizeof( int32_t ) );

            copy( cpu_tensor, device_tensor );

            return device_tensor;
        }

        /**
         * @brief A/B dispatch for next-token selection (TokenSampling.md section 8.1).
         *
         * HostA path: the validated host sampleFromLogits. DeviceB path: the device
         * TokenSampler, which writes the token into decode_token_device_ in place and
         * returns the host value. Both consume the last logits row.
         */
        int32_t sampleNext(
            const TensorType& logits,
            float temperature,
            int top_k,
            std::mt19937& rng )
        {
            if constexpr ( kSamplingPath == SamplingPath::HostA )
            {
                return sampleFromLogits( logits, 0, temperature, top_k, rng );
            }
            else
            {
                SamplingParams params;
                params.temperature = temperature;
                params.top_k = top_k;

                return token_sampler_.sample( logits, decode_token_device_, params );
            }
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

        static GemmaConfig configFromMetadata( const PretrainedMetadata& metadata )
        {
            GemmaConfig config(
                static_cast<dim_t>(metadata.embedding_dim),
                static_cast<dim_t>(metadata.num_layers) );

            config.withVocabularyLength( static_cast<dim_t>(metadata.vocab_size) )
                .withMaxSequenceLength( static_cast<dim_t>(metadata.max_seq_length) )
                .withNumHeads( static_cast<dim_t>(metadata.num_heads) )
                .withNumKVHeads( static_cast<dim_t>(metadata.num_kv_heads) )
                .withHeadDim( static_cast<dim_t>(metadata.head_dim) )
                .withGlobalHeadDim( static_cast<dim_t>(metadata.global_head_dim) )
                .withNumGlobalKVHeads( static_cast<dim_t>(metadata.num_global_kv_heads) )
                .withKeyEqualsValue( metadata.key_equals_value )
                .withHiddenDimension( static_cast<dim_t>(metadata.hidden_dim) )
                .withRMSNormEpsilon( metadata.norm_epsilon )
                .withWindow( static_cast<dim_t>(metadata.window) )
                .withSlidingWindowPattern( static_cast<dim_t>(metadata.sliding_window_pattern) )
                .withGlobalRotaryDim( static_cast<dim_t>(metadata.global_rotary_dim) )
                .withRoPETheta( metadata.rope_theta_local )
                .withGlobalRoPETheta( metadata.rope_theta_global )
                .withFinalLogitSoftcapping( metadata.final_logit_softcapping );

            return config;
        }
    };
}
