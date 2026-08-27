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
#include <limits>

export module Dnn.Models.GemmaModel;

import Dnn.Models.GemmaModelConfig;
import Dnn.LanguageModel;
import Dnn.LanguageModelConfig;
import Dnn.LanguageModelNetwork;
import Dnn.Models.QuantizationDispatch;
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
import Serialization.SafeTensors;
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
        /**
         * @brief KV policy for Gemma's LOCAL (sliding) layers.
         *
         * Bounded sliding-window ring (SlidingWindowKvCache.md Phase 3): their cache is sized
         * to the window working set instead of the full context. Strictly a memory
         * optimization -- tokens are identical to the full cache. GLOBAL (full-attention)
         * layers are always NoKvCompression, hardwired in GemmaTransformer. Flip this alias to
         * NoKvCompression to A/B the footprint against the full-context sliding cache.
         *
         * Class scope rather than per-function so the load and footprint paths cannot be
         * pointed at different policies -- that would make a model report a figure for a
         * cache it does not build.
         */
        using GemmaSlidingKvPolicy = Quant::KvCache::SlidingWindowKvCache;

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

            // Gemma's Linear children (qkv/o/gate_up/down) pick up the weight-quant policy;
            // quantized bodies additionally convert the tied embedding/lm_head table to
            // per-vocab-row FP8 (D4 Design B -- see GemmaTransformer::TableQuantizationPolicy).
            return dispatchWeightQuantization<
                    TPrecision, GemmaSlidingKvPolicy,
                    std::unique_ptr<GemmaModel<TDeviceType, TPrecision>>>(
                model_config.getWeightQuantization(),
                model_config.getKvCacheCompression(),
                "GemmaModel::fromPretrained",
                [&]<WeightQuantPolicy TWeightQuantization, KvCachePolicy TKvCachePolicy>()
                {
                    return fromPretrainedImpl<TWeightQuantization, TKvCachePolicy>(
                        path, model_config, device_id );
                } );
        }

        /**
         * @brief What loading this checkpoint at this context length would cost in VRAM.
         *
         * Reads the artifact header for geometry and constructs the graph, then reports what
         * build() would allocate -- without building it, without reading a weight, and
         * therefore without needing the device to have room. Answers before a multi-gigabyte
         * download and for hardware the caller does not own.
         *
         * Returns measurements only. Whether a given headroom is too tight is a deployment
         * policy and belongs to the adaptor, not here; on Windows in particular WDDM
         * oversubscribes rather than failing, so "fits" is not a property the runtime can
         * decide. See Specifications/MemoryFootprint.md.
         *
         * @throws std::invalid_argument on device type mismatch or zero context length.
         * @throws std::runtime_error    on an unreadable artifact or unsupported quantization.
         */
        static MemoryStats getRequiredMemory(
            const std::filesystem::path& path,
            const GemmaModelConfig& model_config,
            DeviceId device_id = DeviceId{ TDeviceType, 0 } )
        {
            return getDeploymentFootprint( path, model_config, device_id ).memory;
        }

        /**
         * @brief The same prediction, plus how this deployment would chunk its prefill.
         *
         * One graph construction answers both, because they are two readings of the same
         * arithmetic -- the chunk is resolved on the way to sizing the activation workspaces
         * getRequiredMemory reports, and was previously discarded there.
         *
         * The second half is what a caller choosing a context length needs and memory alone
         * cannot tell it: the largest context that fits can be one where the chunk has walked
         * down to its floor, because the activation budget shrinks as the KV cache it shares
         * VRAM with grows. See Specifications/ChatConfiguration.md section 6.
         *
         * @throws std::invalid_argument on device type mismatch or zero context length.
         * @throws std::runtime_error    on an unreadable artifact or unsupported quantization.
         */
        static DeploymentFootprint getDeploymentFootprint(
            const std::filesystem::path& path,
            const GemmaModelConfig& model_config,
            DeviceId device_id = DeviceId{ TDeviceType, 0 } )
        {
            if ( device_id.type != TDeviceType )
            {
                throw std::invalid_argument( std::format(
                    "GemmaModel::getDeploymentFootprint: device type mismatch: expected {}, got {}",
                    deviceTypeToString( TDeviceType ),
                    deviceTypeToString( device_id.type ) ) );
            }

            if ( model_config.getContextLength() == 0 )
            {
                throw std::invalid_argument(
                    "GemmaModel::getDeploymentFootprint: context_length must be greater than zero" );
            }

            // Same dispatcher as fromPretrained, and deliberately so: the footprint path and
            // the load path must reach the identical template instantiation or a model reports
            // a figure it does not allocate.
            return dispatchWeightQuantization<
                    TPrecision, GemmaSlidingKvPolicy, DeploymentFootprint>(
                model_config.getWeightQuantization(),
                model_config.getKvCacheCompression(),
                "GemmaModel::getDeploymentFootprint",
                [&]<WeightQuantPolicy TWeightQuantization, KvCachePolicy TKvCachePolicy>()
                {
                    return deploymentFootprintImpl<TWeightQuantization, TKvCachePolicy>(
                        path, model_config, device_id );
                } );
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
        dim_t contextLength() const noexcept
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
            this->getNetwork().prefill( input );
            this->getNetwork().synchronize();
        }

        /**
         * @brief Logits fingerprint for a fixed prompt, for comparing two loads of one model.
         *
         * Diagnostic. Parameters and config can be proven byte-identical between a
         * quantize-on-load and a pre-quantized load while the models still behave
         * differently, and nothing upstream of inference can see that. This runs one prefill
         * and reports what the model actually computed.
         *
         * Raw token ids rather than text so no tokenizer is involved and two runs are
         * comparable by construction.
         *
         * @return Digest of the last-position logits, the argmax token, and its value.
         */
        std::string fingerprintPrefill( const std::vector<int32_t>& token_ids )
        {
            auto input = makeTokenTensor( token_ids );

            // Per-stage summary on the real prefill path. Logits are computed last, so a NaN
            // there says nothing about where it entered; the first stage reporting a NaN
            // does.
            std::string stages;

            this->getNetwork().setStageProbe(
                [&]( std::string_view stage, const TensorType& value )
                {
                    const auto summary = summarizeActivation( value );

                    // Only the transition matters, so report every stage until the first
                    // NaN and then stop adding noise.
                    if ( stages.empty() || summary.nan_count > 0 )
                    {
                        stages += std::format( "  {:<12} {}\n", stage, summary.text );
                    }

                    if ( summary.nan_count > 0 && !first_nan_stage_.has_value() )
                    {
                        first_nan_stage_ = std::string( stage );
                    }
                } );

            auto& logits = this->getNetwork().prefill( input );
            this->getNetwork().synchronize();

            this->getNetwork().setStageProbe( {} );

            // StagingMR is the class alias: pinned on CUDA, because every reduced precision
            // Mila serves in is is_device_only and a CpuMemoryResource tensor of TPrecision
            // is not a valid template-id.
            Tensor<TPrecision, StagingMR> staged( this->getDeviceId(), logits.shape() );
            copy( logits, staged );
            this->getNetwork().synchronize();

            const auto* bytes = static_cast<const uint8_t*>( staged.rawData() );
            const size_t total_bytes = staged.getStorageSize();

            // FNV-1a rather than a real digest: this only has to distinguish two runs, and
            // a model diagnostic should not depend on the model-download feature being
            // compiled in.
            uint64_t digest = 1469598103934665603ull;

            for ( size_t i = 0; i < total_bytes; ++i )
            {
                digest ^= bytes[ i ];
                digest *= 1099511628211ull;
            }

            // Argmax over the last position, decoded straight from the stored bits so this
            // works for BF16 without a conversion pass. BF16 is the high half of an FP32,
            // and FP32 falls out of the same read.
            const dim_t count = staged.size();
            float best_value = -std::numeric_limits<float>::infinity();
            dim_t best_index = 0;

            for ( dim_t i = 0; i < count; ++i )
            {
                float value = 0.0f;

                if constexpr ( TPrecision == TensorDataType::FP32 )
                {
                    value = reinterpret_cast<const float*>( bytes )[ i ];
                }
                else
                {
                    const uint32_t widened =
                        static_cast<uint32_t>( reinterpret_cast<const uint16_t*>( bytes )[ i ] ) << 16;
                    std::memcpy( &value, &widened, sizeof( value ) );
                }

                if ( value > best_value )
                {
                    best_value = value;
                    best_index = i;
                }
            }

            const std::string origin = first_nan_stage_.has_value()
                ? std::format( "  FIRST NaN AT: {}\n", *first_nan_stage_ )
                : std::string( "  no NaN in any prefill stage\n" );

            first_nan_stage_.reset();

            return std::format(
                "\n{}{}  logits      fnv1a {:016x} | elements {} | argmax {} = {:.6f}",
                stages, origin, digest, count, best_index, best_value );
        }

    private:

        /// Set by the stage probe; names the earliest activation containing a NaN.
        std::optional<std::string> first_nan_stage_;

        struct ActivationSummary
        {
            int64_t nan_count{ 0 };
            std::string text;
        };

        /**
         * @brief Count NaNs and bound the finite range of an activation.
         *
         * Bits are decoded directly rather than converted, so BF16 needs no extra pass:
         * BF16 is the high half of an FP32.
         */
        ActivationSummary summarizeActivation( const TensorType& value )
        {
            Tensor<TPrecision, StagingMR> staged( this->getDeviceId(), value.shape() );
            copy( value, staged );
            this->getNetwork().synchronize();

            const auto* bytes = static_cast<const uint8_t*>( staged.rawData() );
            const dim_t count = staged.size();

            ActivationSummary summary;
            float lowest = std::numeric_limits<float>::infinity();
            float highest = -std::numeric_limits<float>::infinity();

            for ( dim_t i = 0; i < count; ++i )
            {
                float element = 0.0f;

                if constexpr ( TPrecision == TensorDataType::FP32 )
                {
                    element = reinterpret_cast<const float*>( bytes )[ i ];
                }
                else
                {
                    const uint32_t widened =
                        static_cast<uint32_t>( reinterpret_cast<const uint16_t*>( bytes )[ i ] ) << 16;
                    std::memcpy( &element, &widened, sizeof( element ) );
                }

                if ( std::isnan( element ) )
                {
                    ++summary.nan_count;
                    continue;
                }

                lowest = std::min( lowest, element );
                highest = std::max( highest, element );
            }

            summary.text = std::format( "elements {:>8} | NaN {:>8} | range [{:.4f}, {:.4f}]",
                count, summary.nan_count, lowest, highest );

            return summary;
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

            // Transparent KV prefix reuse (PromptCaching.md): cache positions [0, n)
            // are a deterministic function of the first n tokens, so exact token
            // equality against what the caches already hold is the sole validity
            // test -- reuse can never change outputs. Cap at seq_len - 1 so at least
            // the final position prefills and the sampled logits are fresh. A refused
            // rewind (bounded-ring staleness, cold cache) falls back to the full
            // prefill, which positionally overwrites regardless of cache state.
            int64_t common = 0;
            const int64_t comparable = std::min(
                seq_len, static_cast<int64_t>( kv_token_history_.size() ) );

            while ( common < comparable && kv_token_history_[ static_cast<size_t>( common ) ] == prompt_tokens[ static_cast<size_t>( common ) ] )
                ++common;

            const int64_t reuse = std::min( common, seq_len - 1 );

            const bool reused = reuse > 0
                && this->getNetwork().rewindKvCache( reuse );

            auto& logits = reused
                ? this->getNetwork().prefillFrom( prefill_input, reuse )
                : this->getNetwork().prefill( prefill_input );

            if ( reused )
                Logging::Logger::info( std::format(
                    "GemmaModel: KV prefix reuse -- skipped {} of {} prompt tokens", reuse, seq_len ) );

            // The caches now hold exactly the prompt; decode appends below in lockstep.
            kv_token_history_.assign( prompt_tokens.begin(), prompt_tokens.end() );

            // Decode-ahead pipeline: the sampler runs on the network stream (ordered
            // after the forward that produced the logits -- no synchronize needed) and
            // writes the sampled token into decode_token_device_ in place, so the NEXT
            // forward is enqueued before the host has read the token id back. The host
            // readback (awaitSampledToken) then overlaps the GPU forward, hiding the
            // per-token host gap (stream-sync wake-up, stop-check, on_token, and the
            // ~340 kernel-launch enqueues) that showed as saw-tooth decode utilization.
            //
            // Consequence: a sampled stop token has already been decoded into the KV
            // cache by the time the host sees it. The history append below keeps the
            // reuse bookkeeping exact -- and the cached stop-token K/V is itself
            // reusable, since the next turn's chat template starts with it.
            this->enqueueSampleNext( logits, decode_token_device_, params.sampling );

            dim_t position = seq_len;
            int emitted = 0;

            // nullopt max_new_tokens => run to EOS / the context bound (the guard below).
            const int max_new = params.max_new_tokens.value_or( static_cast<int>( contextLength() ) );

            while ( true )
            {
                if ( stop.stop_requested() )
                {
                    // Drain the in-flight sampling step so nothing runs past return.
                    this->getNetwork().synchronize();
                    return GenerateStatus::ClientCancelled;
                }

                // Decode ahead only when another step could consume its logits: within
                // the per-call token budget, and with KV-cache room -- decode cannot
                // write at a position past the deployment context length.
                const bool more_steps_allowed = emitted + 1 < max_new;
                const bool cache_has_room = position < contextLength();

                TensorType* decode_logits = nullptr;

                if ( more_steps_allowed && cache_has_room )
                    decode_logits = &this->getNetwork().decode( decode_token_device_, position );

                const int32_t token = this->awaitSampledToken();

                if ( decode_logits )
                {
                    // The ahead-decode entered this token into the KV cache at
                    // `position`, whatever it turns out to be; the reuse history
                    // must record it in lockstep.
                    kv_token_history_.push_back( token );
                    ++position;
                }

                if ( stop_ids.contains( token ) )
                {
                    // The ahead-decode of the stop token may still be in flight.
                    this->getNetwork().synchronize();
                    return GenerateStatus::Success;
                }

                on_token( token );
                ++emitted;

                if ( !decode_logits )
                {
                    // No ahead-decode was enqueued, so the stream drained at the await
                    // above. Token budget takes precedence over the context bound,
                    // matching the pre-pipeline loop's check order.
                    return more_steps_allowed
                        ? GenerateStatus::ContextOverflow
                        : GenerateStatus::MaxNewTokensReached;
                }

                this->enqueueSampleNext( *decode_logits, decode_token_device_, params.sampling );
            }
        }

        void onTraining() override
        {
            throw std::runtime_error(
                "GemmaModel::onTraining: Gemma is inference-only" );
        }

        dim_t maxSequenceLength() const noexcept override
        {
            return config_.getMaxSequenceLength();
        }

        dim_t vocabSize() const noexcept override
        {
            return config_.getVocabSize();
        }

    private:

        explicit GemmaModel(
            std::unique_ptr<LanguageModelNetwork<TDeviceType, TPrecision>> network,
            const GemmaConfig& config,
            const GemmaModelConfig& model_config,
            const PretrainedMetadata& source_metadata,
            RuntimeMode runtime_mode )
            : ModelBase( std::move( network ), runtime_mode,
                source_metadata, model_config.getWeightQuantization() )
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

            // A pre-quantized artifact declares the policy its bytes were packed with. The
            // storage dtype alone cannot tell FP4 at group 128 from group 64 -- both are
            // U8 -- so loading one into a build expecting the other would reinterpret the
            // nibble layout and produce a model that runs and is wrong. An empty
            // declaration is a full-precision source and any policy may quantize it on load.
            const std::string& artifact_quantization = reader.getWeightQuantization();

            if ( !artifact_quantization.empty() )
            {
                const std::string requested =
                    weightQuantizationName( model_config.getWeightQuantization() );

                if ( artifact_quantization != requested )
                {
                    throw std::runtime_error( std::format(
                        "GemmaModel::fromPretrained: artifact '{}' is pre-quantized as '{}' "
                        "but this load requested '{}'",
                        path.string(), artifact_quantization, requested ) );
                }
            }

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
                    model_config, metadata, RuntimeMode::Inference ) );
        }

        /**
         * @brief The footprint sibling of fromPretrainedImpl: same prologue, stops before build().
         *
         * Everything above network->build() is shared with the load path deliberately -- the
         * artifact check, the geometry, and the context-length validation must be the ones a
         * real load would apply, or the reported figure describes a model that would not load.
         */
        template<WeightQuantPolicy TWeightQuantization, KvCachePolicy TKvCachePolicy>
        static DeploymentFootprint deploymentFootprintImpl(
            const std::filesystem::path& path,
            const GemmaModelConfig& model_config,
            DeviceId device_id )
        {
            PretrainedModelReader reader( path );
            const auto& metadata = reader.getPretrainedMetadata();

            const std::string& artifact_quantization = reader.getWeightQuantization();

            if ( !artifact_quantization.empty() )
            {
                const std::string requested =
                    weightQuantizationName( model_config.getWeightQuantization() );

                if ( artifact_quantization != requested )
                {
                    throw std::runtime_error( std::format(
                        "GemmaModel::getDeploymentFootprint: artifact '{}' is pre-quantized as '{}' "
                        "but this query requested '{}'",
                        path.string(), artifact_quantization, requested ) );
                }
            }

            GemmaConfig network_config = configFromMetadata( metadata );

            if ( model_config.getContextLength() > network_config.getMaxSequenceLength() )
            {
                throw std::invalid_argument( std::format(
                    "GemmaModel::getDeploymentFootprint: context_length {} exceeds trained max_seq_len {}",
                    model_config.getContextLength(),
                    network_config.getMaxSequenceLength() ) );
            }

            using ConcreteTransformerType =
                GemmaTransformer<TDeviceType, TPrecision, TWeightQuantization, TKvCachePolicy>;

            // Construction commits no device memory -- that is the whole premise. The graph
            // exists, correctly shaped, and is then asked rather than built.
            auto network = std::make_unique<ConcreteTransformerType>(
                metadata.model_name, network_config, device_id );

            const dim_t context_length = static_cast<dim_t>( model_config.getContextLength() );

            BuildContext build_context(
                shape_t{ 1, context_length },
                RuntimeMode::Inference,
                false );

            return DeploymentFootprint{
                network->getRequiredMemory( build_context ),
                network->prefillChunking( 1, context_length ) };
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

        // The token ids whose K/V the caches currently hold, in position order:
        // the last prefilled prompt plus every token fed through decode (appended
        // in lockstep with the decode call). Drives the transparent prompt-prefix
        // reuse in onGenerating (PromptCaching.md): host-side bookkeeping only,
        // bounded by the deployment context length.
        std::vector<int32_t> kv_token_history_;

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
                .withFinalLogitSoftcapping( metadata.final_logit_softcapping )
                .withTieWordEmbeddings( metadata.tie_word_embeddings );

            return config;
        }
    };
}
