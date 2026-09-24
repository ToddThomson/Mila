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
#include <cstddef>
#include <expected>
#include <string_view>
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

export module Dnn.Models.GemmaModel;

import Dnn.Models.GemmaModelConfig;
import Dnn.LanguageModel;
import Dnn.LanguageModelConfig;
import Dnn.LanguageModelNetwork;
import Dnn.Models.QuantizationDispatch;
import Deployment.PrefillChunkRule;
import Deployment.DeviceReading;
import Deployment.DeploymentRequest;
import Deployment.DeploymentPlan;
import Deployment.DeploymentPlans;
import Deployment.DeploymentRefusal;
import Deployment.DeploymentRefusedError;
import Deployment.DeploymentPlanner;
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
import Compute.DeviceAllocation;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.DeviceTypeTraits.Cpu;
import Compute.CpuMemoryResource;
#ifdef MILA_HAS_CUDA
import Compute.DeviceTypeTraits.Cuda;
#endif
import Compute.ExecutionContextFactory;
import Serialization.WeightsReader;
import Serialization.SafeTensors;
import Serialization.Mode;
import Logging.Logger;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;
    using namespace Mila::Deployment;
    using namespace Mila::Dnn::Serialization;
    using namespace Mila::Dnn::Quant::Weight;
    using namespace Mila::Dnn::Quant::KvCache;

    /**
     * @brief Gemma 4 compatible inference model.
     *
     * Owns a loaded, built GemmaTransformer and drives the prefill + KV-cache
     * decode two-phase generation loop. Construction is only possible via
     * load(); the network is always built, weights-loaded, and in
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

        GemmaModel( const GemmaModel& ) = delete;
        GemmaModel& operator=( const GemmaModel& ) = delete;
        GemmaModel( GemmaModel&& ) = default;
        GemmaModel& operator=( GemmaModel&& ) = default;

        ~GemmaModel() = default;

        /**
         * @brief Decide how this package would run on the request's device, allocating nothing.
         *
         * Reads the weights header, constructs the graph, takes one reading of the device and plans against
         * it (Specifications/Deployment.md). Nothing fitting is an answer, not an error: the refusal names why.
         *
         * @throws std::invalid_argument on a device type mismatch, or a context length past the trained maximum.
         * @throws std::runtime_error    on an unreadable package or an unsupported quantization.
         */
        static std::expected<DeploymentPlans, DeploymentRefusal> planDeployment(
            const std::filesystem::path& path,
            const DeploymentRequest& request )
        {
            const DeviceId device = requireDevice( "GemmaModel::planDeployment",
                request.getDevice().value_or( DeviceId{ TDeviceType, 0 } ) );

            return dispatchChassis<std::expected<DeploymentPlans, DeploymentRefusal>>(
                path, request.getWeightQuantization(), request.getKvCacheCompression(), "GemmaModel::planDeployment",
                [&]<WeightQuantPolicy TWeightQuantization, KvCachePolicy TKvCachePolicy, bool kMixtureOfExperts>()
                {
                    return planImpl<TWeightQuantization, TKvCachePolicy, kMixtureOfExperts>( path, request, device );
                } );
        }

        /**
         * @brief Load exactly this plan: its device, context length and prefill chunk, decided nowhere else.
         *
         * @throws std::invalid_argument when the package is not the one the plan was priced for.
         * @throws std::runtime_error    on load failure, including an allocation that free memory no longer
         *                               covers -- a load never re-plans.
         */
        static std::unique_ptr<GemmaModel<TDeviceType, TPrecision>> load(
            const std::filesystem::path& path,
            const DeploymentPlan& plan )
        {
            requireDevice( "GemmaModel::load", plan.device() );

            // Gemma's Linear children (qkv/o/gate_up/down) pick up the weight-quant policy;
            // quantized bodies additionally convert the tied embedding/lm_head table to
            // per-vocab-row FP8 (D4 Design B -- see GemmaTransformer::TableQuantizationPolicy).
            return dispatchChassis<std::unique_ptr<GemmaModel<TDeviceType, TPrecision>>>(
                path, plan.weightQuantization(), plan.kvCacheCompression(), "GemmaModel::load",
                [&]<WeightQuantPolicy TWeightQuantization, KvCachePolicy TKvCachePolicy, bool kMixtureOfExperts>()
                {
                    return loadImpl<TWeightQuantization, TKvCachePolicy, kMixtureOfExperts>( path, plan );
                } );
        }

        /**
         * @brief Plan the request and load its best plan.
         *
         * @throws DeploymentRefusedError when nothing fits, carrying the refusal.
         */
        static std::unique_ptr<GemmaModel<TDeviceType, TPrecision>> load(
            const std::filesystem::path& path,
            const DeploymentRequest& request )
        {
            const auto planned = planDeployment( path, request );

            if ( !planned )
            {
                throw DeploymentRefusedError( "GemmaModel::load", planned.error() );
            }

            return load( path, planned->best() );
        }

        /**
         * @brief Load Gemma 4 at a fixed context length: the request every value of which is fixed.
         *
         * The model_config carries the deployment decisions (context length, weight quantization, KV-cache
         * compression); every architectural parameter is read from the checkpoint metadata. A context length
         * that does not fit is refused rather than attempted (Deployment.md 12.2).
         *
         * @throws std::invalid_argument  on device type mismatch or zero context length.
         * @throws DeploymentRefusedError when the context length does not fit the device.
         * @throws std::runtime_error     on load failure or unsupported quantization.
         */
        static std::unique_ptr<GemmaModel<TDeviceType, TPrecision>> load(
            const std::filesystem::path& path,
            const GemmaModelConfig& model_config,
            DeviceId device_id = DeviceId{ TDeviceType, 0 } )
        {
            requireDevice( "GemmaModel::load", device_id );

            if ( model_config.getContextLength() == 0 )
            {
                throw std::invalid_argument(
                    "GemmaModel::load: context_length must be greater than zero" );
            }

            return load( path, DeploymentRequest::fromModelConfig( model_config ).withDevice( device_id ) );
        }

        /// The plan this model was loaded with: what it runs, not a second derivation of it.
        const DeploymentPlan& getDeploymentPlan() const noexcept
        {
            return plan_;
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
         * One graph construction and one reading of free memory answer both: the chunk is chosen
         * against the reading, and the memory is priced at that chunk -- the one a load given the
         * same reading builds with.
         *
         * The second half is what a caller choosing a context length needs and memory alone
         * cannot tell it: the largest context that fits can be one where the chunk has walked
         * down to its floor, because a longer context leaves less of the free memory for the
         * chunk. See Specifications/MemoryFootprint.md section 11.
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

            // Same dispatcher as load, and deliberately so: the footprint path and
            // the load path must reach the identical template instantiation or a model reports
            // a figure it does not allocate.
            if ( isRoutedCheckpoint( path ) )
            {
                return dispatchWeightQuantization<TPrecision, GemmaSlidingKvPolicy, DeploymentFootprint, kRoutedFp4GroupSize>(
                    model_config.getWeightQuantization(),
                    model_config.getKvCacheCompression(),
                    "GemmaModel::getDeploymentFootprint",
                    [&]<WeightQuantPolicy TWeightQuantization, KvCachePolicy TKvCachePolicy>()
                    {
                        return deploymentFootprintImpl<TWeightQuantization, TKvCachePolicy, true>(
                            path, model_config, device_id );
                    } );
            }

            return dispatchWeightQuantization<TPrecision, GemmaSlidingKvPolicy, DeploymentFootprint, kDenseFp4GroupSize>(
                model_config.getWeightQuantization(),
                model_config.getKvCacheCompression(),
                "GemmaModel::getDeploymentFootprint",
                [&]<WeightQuantPolicy TWeightQuantization, KvCachePolicy TKvCachePolicy>()
                {
                    return deploymentFootprintImpl<TWeightQuantization, TKvCachePolicy, false>(
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

        /**
         * @brief The network geometry a load of this file builds.
         *
         * Public so a caller that constructs blocks itself -- the layer-streamed parity harness -- uses
         * the geometry a real load would, rather than a second reading of the metadata.
         */
        static GemmaConfig configFromMetadata( const WeightsMetadata& metadata )
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
                .withTieWordEmbeddings( metadata.tie_word_embeddings )
                .withMixtureOfExperts( static_cast<dim_t>(metadata.num_experts),
                    static_cast<dim_t>(metadata.top_k_experts), static_cast<dim_t>(metadata.expert_hidden_dim) );

            return config;
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

        // FP4 group per chassis. The routed model's expert width (704) and dense-branch width (2112) are
        // not multiples of 128, which CudaLinearOp refuses; every Gemma 4 width is a multiple of 64.
        static constexpr int kDenseFp4GroupSize = 128;
        static constexpr int kRoutedFp4GroupSize = 64;

        explicit GemmaModel(
            std::unique_ptr<LanguageModelNetwork<TDeviceType, TPrecision>> network,
            const GemmaConfig& config,
            const DeploymentPlan& plan,
            const WeightsMetadata& source_metadata,
            int fp4_group_size,
            RuntimeMode runtime_mode )
            : ModelBase( std::move( network ), runtime_mode,
                source_metadata, plan.weightQuantization(), fp4_group_size )
            , config_( config )
            , model_config_( plan.modelConfig<GemmaModelConfig>() )
            , plan_( plan )
            , decode_token_device_( this->getDeviceId(), shape_t{ 1, 1 } )
        {}

        static bool isRoutedCheckpoint( const std::filesystem::path& path )
        {
            WeightsReader reader( path );

            return reader.getWeightsMetadata().num_experts > 0;
        }

        static DeviceId requireDevice( std::string_view caller, DeviceId device_id )
        {
            if ( device_id.type != TDeviceType )
            {
                throw std::invalid_argument( std::format(
                    "{}: device type mismatch: expected {}, got {}", caller,
                    deviceTypeToString( TDeviceType ), deviceTypeToString( device_id.type ) ) );
            }

            return device_id;
        }

        /**
         * @brief The one runtime-to-compile-time bridge for planning and loading.
         *
         * The routed chassis and its FP4 group are both the checkpoint's, so its geometry is read before the
         * dispatch. Planning and loading reach the identical instantiation through here, or a plan would price
         * a network the load does not build.
         */
        template<typename TResult, typename TAction>
        static TResult dispatchChassis(
            const std::filesystem::path& path, WeightQuantization weight_quantization,
            KvCacheCompression kv_cache_compression, std::string_view caller, TAction&& action )
        {
            if ( isRoutedCheckpoint( path ) )
            {
                return dispatchWeightQuantization<TPrecision, GemmaSlidingKvPolicy, TResult, kRoutedFp4GroupSize>(
                    weight_quantization, kv_cache_compression, caller,
                    [&]<WeightQuantPolicy TWeightQuantization, KvCachePolicy TKvCachePolicy>()
                    {
                        return action.template operator()<TWeightQuantization, TKvCachePolicy, true>();
                    } );
            }

            return dispatchWeightQuantization<TPrecision, GemmaSlidingKvPolicy, TResult, kDenseFp4GroupSize>(
                weight_quantization, kv_cache_compression, caller,
                [&]<WeightQuantPolicy TWeightQuantization, KvCachePolicy TKvCachePolicy>()
                {
                    return action.template operator()<TWeightQuantization, TKvCachePolicy, false>();
                } );
        }

        // A routed network delegates its dense branch, so both flags follow the chassis.
        template<WeightQuantPolicy TWeightQuantization, KvCachePolicy TKvCachePolicy, bool kMixtureOfExperts>
        using ChassisTransformer = GemmaTransformer<TDeviceType, TPrecision,
            TWeightQuantization, TKvCachePolicy, kMixtureOfExperts, kMixtureOfExperts>;

        template<WeightQuantPolicy TWeightQuantization, KvCachePolicy TKvCachePolicy, bool kMixtureOfExperts>
        static std::expected<DeploymentPlans, DeploymentRefusal> planImpl(
            const std::filesystem::path& path,
            const DeploymentRequest& request,
            DeviceId device_id )
        {
            WeightsReader reader( path );
            const auto& metadata = reader.getWeightsMetadata();

            requireStoredQuantizationMatches(
                "GemmaModel::planDeployment", path.string(), reader.getWeightQuantization(),
                request.getWeightQuantization(), kMixtureOfExperts ? kRoutedFp4GroupSize : kDenseFp4GroupSize );

            const GemmaConfig network_config = configFromMetadata( metadata );

            // Construction commits no device memory, but it creates the execution context, which holds some;
            // the reading is taken after it, as the load's build will find the device (Deployment.md 9).
            const ChassisTransformer<TWeightQuantization, TKvCachePolicy, kMixtureOfExperts> network(
                metadata.model_name, network_config, device_id );

            return planOnDevice( network, request, DeviceReading::take( device_id ),
                network_config.getMaxSequenceLength(), metadata, reader.getWeightQuantization() );
        }

        template<WeightQuantPolicy TWeightQuantization, KvCachePolicy TKvCachePolicy, bool kMixtureOfExperts>
        static std::unique_ptr<GemmaModel<TDeviceType, TPrecision>> loadImpl(
            const std::filesystem::path& path,
            const DeploymentPlan& plan )
        {
            WeightsReader reader( path );
            const auto& metadata = reader.getWeightsMetadata();
            const int fp4_group_size = kMixtureOfExperts ? kRoutedFp4GroupSize : kDenseFp4GroupSize;

            plan.requirePricedFor( "GemmaModel::load", path.string(), metadata, reader.getWeightQuantization() );

            requireStoredQuantizationMatches(
                "GemmaModel::load", path.string(), reader.getWeightQuantization(),
                plan.weightQuantization(), fp4_group_size );

            const GemmaConfig network_config = configFromMetadata( metadata );

            auto network = std::make_unique<ChassisTransformer<TWeightQuantization, TKvCachePolicy, kMixtureOfExperts>>(
                metadata.model_name, network_config, plan.device() );

            network->build( plan.buildContext() );

            Logging::Logger::info( network->toString() );

            network->loadParameters( reader );

            return std::unique_ptr<GemmaModel<TDeviceType, TPrecision>>(
                new GemmaModel<TDeviceType, TPrecision>(
                    std::move( network ), network_config,
                    plan, metadata, fp4_group_size, RuntimeMode::Inference ) );
        }

        /**
         * @brief The footprint sibling of loadImpl: same prologue, stops before build().
         *
         * Everything above network->build() is shared with the load path deliberately -- the
         * artifact check, the geometry, and the context-length validation must be the ones a
         * real load would apply, or the reported figure describes a model that would not load.
         */
        template<WeightQuantPolicy TWeightQuantization, KvCachePolicy TKvCachePolicy, bool kMixtureOfExperts>
        static DeploymentFootprint deploymentFootprintImpl(
            const std::filesystem::path& path,
            const GemmaModelConfig& model_config,
            DeviceId device_id )
        {
            WeightsReader reader( path );
            const auto& metadata = reader.getWeightsMetadata();

            requireStoredQuantizationMatches(
                "GemmaModel::getDeploymentFootprint", path.string(),
                reader.getWeightQuantization(), model_config.getWeightQuantization(),
                kMixtureOfExperts ? kRoutedFp4GroupSize : kDenseFp4GroupSize );

            GemmaConfig network_config = configFromMetadata( metadata );

            if ( model_config.getContextLength() > network_config.getMaxSequenceLength() )
            {
                throw std::invalid_argument( std::format(
                    "GemmaModel::getDeploymentFootprint: context_length {} exceeds trained max_seq_len {}",
                    model_config.getContextLength(),
                    network_config.getMaxSequenceLength() ) );
            }

            const dim_t context_length = static_cast<dim_t>( model_config.getContextLength() );

            using ConcreteTransformerType = GemmaTransformer<TDeviceType, TPrecision,
                TWeightQuantization, TKvCachePolicy, kMixtureOfExperts, kMixtureOfExperts>;

            // Construction commits no device memory -- that is the whole premise. The graph
            // exists, correctly shaped, and is then asked rather than built.
            auto network = std::make_unique<ConcreteTransformerType>(
                metadata.model_name, network_config, device_id );

            // The one reading of the device this prediction takes, and the graph is priced at the
            // chunk chosen against it (Deployment.md section 4). Taken after construction, as the
            // load takes its own: the execution context construction creates holds device memory,
            // and a reading before it names a chunk no build can get.
            const BuildContext priced =
                BuildContext( shape_t{ 1, context_length }, RuntimeMode::Inference, false )
                .withAllocationGranularity( allocationGranularity( device_id ) );
            const PrefillChunking prefill = choosePrefillChunk( *network, priced, readFreeDeviceBytes( device_id ) );

            return DeploymentFootprint{
                network->getRequiredMemory( priced.withPrefillSize( prefill.chunk_rows ) ), prefill };
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

        // The plan this model executed; reported, never re-derived.
        DeploymentPlan plan_;

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
    };
}
