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
#include <optional>
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
import Dnn.GenerateParams;
import Dnn.GenerateStatus;
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
            //
            // Bounded sliding-window KV ring for Gemma's LOCAL (sliding) layers
            // (SlidingWindowKvCache.md Phase 3): their cache is sized to the window
            // working set instead of the full context. Strictly a memory optimization —
            // tokens are identical to the full cache. Global (full-attention) layers are
            // always NoKvCompression (hardwired in GemmaTransformer). Flip this alias to
            // NoKvCompression to A/B the footprint against the full-context sliding cache.
            using GemmaSlidingKvPolicy = SlidingWindowKvCache;

            switch ( model_config.getWeightQuantization() )
            {
                case WeightQuantization::FP4:
                    switch ( model_config.getKvCacheCompression() )
                    {
                        case KvCacheCompression::FP8:
                        case KvCacheCompression::None:
                            if constexpr ( TPrecision == TensorDataType::BF16 )
                            {
                                return fromPretrainedImpl<PerGroupFp4<128>, GemmaSlidingKvPolicy>( path, model_config, device_id );
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
                                return fromPretrainedImpl<PerChannelFp8<>, GemmaSlidingKvPolicy>( path, model_config, device_id );
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
                            return fromPretrainedImpl<NoWeightQuant, GemmaSlidingKvPolicy>( path, model_config, device_id );
                    }
                    break;
            }

            throw std::runtime_error( "GemmaModel::fromPretrained: unhandled quantization configuration" );
        }

        // ====================================================================
        // Accessors
        // ====================================================================

        /// Architecture/network configuration (read from the checkpoint metadata).
        const GemmaConfig& getNetworkConfig() const noexcept
        {
            return config_;
        }

        /// Deployment configuration (context length, weight-quant, kv-compression) this model was loaded with.
        const GemmaModelConfig& getModelConfig() const noexcept
        {
            return model_config_;
        }

        /// Deployment context length: the KV-cache depth the network was built with.
        int64_t contextLength() const noexcept
        {
            return static_cast<int64_t>( model_config_.getContextLength() );
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
            oss << model_config_.toString();

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

        GenerateStatus onGenerating(
            std::span<const int32_t> prompt_tokens,
            const std::function<void( int32_t )>& on_token,
            const GenerateParams& params,
            std::stop_token stop ) override
        {
            // The prompt must fit the deployment context (the KV-cache depth the network
            // was built with). Context management is the caller's concern; reject rather
            // than silently truncate.
            if ( prompt_tokens.size() > static_cast<size_t>( contextLength() ) )
            {
                throw std::invalid_argument( std::format(
                    "GemmaModel::onGenerating: prompt length {} exceeds deployment context length {}",
                    prompt_tokens.size(), contextLength() ) );
            }

            // Stop set: the model defaults (EOS is a model/tokenizer property), unless the
            // caller overrides them for this call (advanced structured generation).
            std::unordered_set<int32_t> stop_ids;
            if ( params.stop_tokens.empty() )
                stop_ids = stopTokens();
            else
                for ( auto id : params.stop_tokens )
                    stop_ids.insert( static_cast<int32_t>( id ) );

            const int64_t seq_len = static_cast<int64_t>( prompt_tokens.size() );
            auto prefill_input = makeTokenTensor( prompt_tokens );

            auto& logits = this->getLanguageNetwork().prefill( prefill_input );
            this->getLanguageNetwork().synchronize();

            int32_t next_token = this->sampleNext( logits, decode_token_device_, params.sampling );

            if ( stop_ids.contains( next_token ) )
                return GenerateStatus::Success;

            on_token( next_token );

            int position = static_cast<int>( seq_len );

            // nullopt max_new_tokens => run to EOS / the context bound (the guard below).
            const int max_new = params.max_new_tokens.value_or( static_cast<int>( contextLength() ) );

            for ( int step = 1; step < max_new; ++step )
            {
                if ( stop.stop_requested() )
                    return GenerateStatus::ClientCancelled;

                // The KV cache is only as deep as the deployment context length; decode
                // cannot write at a position past it. Stop cleanly instead of letting the
                // GQA op throw "position out of range".
                if ( position >= contextLength() )
                    return GenerateStatus::ContextOverflow;

                // The previous sampleNext() already wrote the sampled token into
                // decode_token_device_ on the device, so it is ready to decode in place --
                // no host round-trip.
                auto& decode_logits = this->getLanguageNetwork().decode( decode_token_device_, position );
                this->getLanguageNetwork().synchronize();

                next_token = this->sampleNext( decode_logits, decode_token_device_, params.sampling );

                if ( stop_ids.contains( next_token ) )
                    return GenerateStatus::Success;

                on_token( next_token );
                ++position;
            }

            return GenerateStatus::MaxNewTokensReached;
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

        explicit GemmaModel(
            std::unique_ptr<LanguageNetwork<TDeviceType, TPrecision>> network,
            const GemmaConfig& config,
            const GemmaModelConfig& model_config,
            RuntimeMode runtime_mode )
            : ModelBase( std::move( network ), runtime_mode )
            , config_( config )
            , model_config_( model_config )
            , decode_token_device_( this->getDeviceId(), shape_t{ 1, 1 } )
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
                    model_config, RuntimeMode::Inference ) );
        }

        // Architecture config (from checkpoint metadata): the trained network geometry.
        GemmaConfig config_;

        // Deployment config this model was loaded with. The deployment context length
        // (model_config_.getContextLength(), exposed via contextLength()) is the KV-cache depth the
        // network was BUILT with -- it may be far below the architectural max
        // (config_.getMaxSequenceLength()), and the prompt check + decode loop bound against THIS,
        // not the architectural max, or the GQA op throws when a write position reaches the cache
        // size. Retained for diagnostics/provenance: the weight-quant / kv-compression were
        // previously discarded after the dispatch switch.
        GemmaModelConfig model_config_;

        // Device decode-input buffer: the sampler writes the next token here in place,
        // and decode() reads it directly -- no host staging round-trip.
        TokenIndexType decode_token_device_;

        // Gemma 4 instruct stop tokens: <eos> = 1, <end_of_turn> = 106 (validated by the
        // token-for-token HF parity run + the live chat). These are the MODEL defaults; the
        // library does not parse the tokenizer -- a harness that owns the tokenizer may
        // override the stop set per call via GenerateParams::stop_tokens.
        static constexpr int32_t kEosToken = 1;
        static constexpr int32_t kEndOfTurnToken = 106;

        int32_t eosToken() const noexcept override
        {
            return kEosToken;
        }

        std::unordered_set<int32_t> stopTokens() const override
        {
            return { kEosToken, kEndOfTurnToken };
        }

        /**
         * @brief Gemma applies a final logit softcap (30 * tanh(logits / 30)) at the sampler.
         */
        float finalLogitSoftcap() const noexcept override
        {
            return config_.getFinalLogitSoftcapping();
        }

        // ====================================================================
        // Generation helpers
        // ====================================================================

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
