/**
 * @file LlamaModel.ixx
 * @brief LLaMA inference model.
 *
 * Inference-only wrapper around a loaded LlamaTransformer network.
 */

module;
#include <memory>
#include <vector>
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
#include <optional>
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
import Dnn.LanguageModelConfig;
import Dnn.GenerateParams;
import Dnn.GenerateStatus;
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
import Dnn.Components.LlamaTransformer;
import Compute.Device;
import Compute.DeviceId;
import Compute.DeviceAllocation;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Compute.DeviceTypeTraits.Cpu;
import Compute.CpuMemoryResource;
#ifdef MILA_HAS_CUDA
import Compute.DeviceTypeTraits.Cuda;
import Compute.CudaPinnedMemoryResource;
#endif
import Compute.ExecutionContextFactory;
import Serialization.WeightsReader;
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
     * @brief LLaMA 3 compatible inference model.
     *
     * Owns a loaded, built LlamaTransformer and exposes generate()
     * for autoregressive text generation. Supports the prefill + KV-cache
     * decode two-phase generation loop.
     *
     * Construction is only possible via load(). The network is always
     * in a built, weights-loaded, inference-mode state when generation is called.
     *
     * Thread safety: not thread-safe; external synchronization required if shared.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class LlamaModel : public LanguageModel<TDeviceType, TPrecision>
    {
    public:
        /**
         * @brief KV policy for the Llama chassis.
         *
         * Every Llama layer is full-attention, so there is no sliding window to bound a ring
         * against and the cache spans the whole context. Class scope rather than per-function
         * so the load and footprint paths cannot be pointed at different policies -- that
         * would make a model report a figure for a cache it does not build.
         */
        using LlamaKvPolicy = Quant::KvCache::NoKvCompression;

        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using ModelBase = LanguageModel<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using TokenIndexType = Tensor<dtype_t::INT32, MR>;
#ifdef MILA_HAS_CUDA
        using StagingMR = std::conditional_t<TDeviceType == DeviceType::Cuda, CudaPinnedMemoryResource, CpuMemoryResource>;
#else
        using StagingMR = CpuMemoryResource;
#endif

        LlamaModel( const LlamaModel& ) = delete;
        LlamaModel& operator=( const LlamaModel& ) = delete;
        LlamaModel( LlamaModel&& ) = default;
        LlamaModel& operator=( LlamaModel&& ) = default;

        ~LlamaModel() = default;

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
            const DeviceId device = requireDevice( "LlamaModel::planDeployment",
                request.getDevice().value_or( DeviceId{ TDeviceType, 0 } ) );

            return dispatchWeightQuantization<TPrecision, LlamaKvPolicy, std::expected<DeploymentPlans, DeploymentRefusal>>(
                request.getWeightQuantization(), request.getKvCacheCompression(), "LlamaModel::planDeployment",
                [&]<WeightQuantPolicy TWeightQuantization, KvCachePolicy TKvCachePolicy>()
                {
                    return planImpl<TWeightQuantization, TKvCachePolicy>( path, request, device );
                } );
        }

        /**
         * @brief Load exactly this plan: its device, context length and prefill chunk, decided nowhere else.
         *
         * @throws std::invalid_argument when the package is not the one the plan was priced for.
         * @throws std::runtime_error    on load failure, including an allocation that free memory no longer
         *                               covers -- a load never re-plans.
         */
        static std::unique_ptr<LlamaModel<TDeviceType, TPrecision>> load(
            const std::filesystem::path& path,
            const DeploymentPlan& plan )
        {
            requireDevice( "LlamaModel::load", plan.device() );

            // Runtime -> compile-time bridge. PerGroupFp4<128> quantizes BF16 weights on load
            // to packed FP4 E2M1 nibbles with per-group float32 scales, consumed by the W4A16
            // kernel with E2M1 decode inline. Llama's chassis has no sliding-window layers, so
            // its KV policy is NoKvCompression throughout.
            return dispatchWeightQuantization<TPrecision, LlamaKvPolicy, std::unique_ptr<LlamaModel<TDeviceType, TPrecision>>>(
                plan.weightQuantization(), plan.kvCacheCompression(), "LlamaModel::load",
                [&]<WeightQuantPolicy TWeightQuantization, KvCachePolicy TKvCachePolicy>()
                {
                    return loadImpl<TWeightQuantization, TKvCachePolicy>( path, plan );
                } );
        }

        /**
         * @brief Plan the request and load its best plan.
         *
         * @throws DeploymentRefusedError when nothing fits, carrying the refusal.
         */
        static std::unique_ptr<LlamaModel<TDeviceType, TPrecision>> load(
            const std::filesystem::path& path,
            const DeploymentRequest& request )
        {
            const auto planned = planDeployment( path, request );

            if ( !planned )
            {
                throw DeploymentRefusedError( "LlamaModel::load", planned.error() );
            }

            return load( path, planned->best() );
        }

        /**
         * @brief Load Llama at a fixed context length: the request every value of which is fixed.
         *
         * The model_config carries the deployment decisions (context length, weight quantization, KV-cache
         * compression); every architectural parameter is read from the checkpoint metadata. A context length
         * that does not fit is refused rather than attempted (Deployment.md 12.2).
         *
         * @throws std::invalid_argument  on device type mismatch or zero context length.
         * @throws DeploymentRefusedError when the context length does not fit the device.
         * @throws std::runtime_error     on load failure or unsupported quantization.
         */
        static std::unique_ptr<LlamaModel<TDeviceType, TPrecision>> load(
            const std::filesystem::path& path,
            const LlamaModelConfig& model_config,
            DeviceId device_id = DeviceId{ TDeviceType, 0 } )
        {
            requireDevice( "LlamaModel::load", device_id );

            if ( model_config.getContextLength() == 0 )
            {
                throw std::invalid_argument(
                    "LlamaModel::load: context_length must be greater than zero" );
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
         * build() would allocate -- without building it and without reading a weight.
         * Returns measurements only; the fits/does-not verdict is adaptor policy.
         * See Specifications/MemoryFootprint.md.
         *
         * @throws std::invalid_argument on device type mismatch or zero context length.
         * @throws std::runtime_error    on an unreadable artifact or unsupported quantization.
         */
        static MemoryStats getRequiredMemory(
            const std::filesystem::path& path,
            const LlamaModelConfig& model_config,
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
         * cannot tell it: the scratch cost per chunk row carries a 2 * context_length term, so
         * a longer context buys a smaller chunk. See Specifications/ChatConfiguration.md
         * section 6.
         *
         * @throws std::invalid_argument on device type mismatch or zero context length.
         * @throws std::runtime_error    on an unreadable artifact or unsupported quantization.
         */
        static DeploymentFootprint getDeploymentFootprint(
            const std::filesystem::path& path,
            const LlamaModelConfig& model_config,
            DeviceId device_id = DeviceId{ TDeviceType, 0 } )
        {
            if ( device_id.type != TDeviceType )
            {
                throw std::invalid_argument( std::format(
                    "LlamaModel::getDeploymentFootprint: device type mismatch: expected {}, got {}",
                    deviceTypeToString( TDeviceType ),
                    deviceTypeToString( device_id.type ) ) );
            }

            if ( model_config.getContextLength() == 0 )
            {
                throw std::invalid_argument(
                    "LlamaModel::getDeploymentFootprint: context_length must be greater than zero" );
            }

            // Same dispatcher as load, and deliberately so: the footprint path and
            // the load path must reach the identical template instantiation or a model reports
            // a figure it does not allocate.
            return dispatchWeightQuantization<
                    TPrecision, LlamaKvPolicy, DeploymentFootprint>(
                model_config.getWeightQuantization(),
                model_config.getKvCacheCompression(),
                "LlamaModel::getDeploymentFootprint",
                [&]<WeightQuantPolicy TWeightQuantization, KvCachePolicy TKvCachePolicy>()
                {
                    return deploymentFootprintImpl<TWeightQuantization, TKvCachePolicy>(
                        path, model_config, device_id );
                } );
        }

        // ====================================================================
        // Accessors
        // ====================================================================

        const LlamaConfig& getConfig() const noexcept
        {
            return config_;
        }

        /**
         * @brief The network geometry a load of this file builds.
         *
         * Public so a caller that constructs the network itself uses the geometry a real load
         * would, rather than a second reading of the metadata.
         */
        static LlamaConfig configFromMetadata( const WeightsMetadata& metadata )
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

        // ====================================================================
        // Diagnostics
        // ====================================================================

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "LlamaModel\n";
            oss << "Device: " << this->getDeviceId().toString() << "\n";
            oss << "Vocabulary: " << config_.getVocabSize() << " tokens\n";

            // Two different numbers, and only the first bounds generation: the session was
            // built for context_length_, while config_ carries the maximum the checkpoint
            // declares. Reporting only the latter overstates the usable window.
            oss << "Context length: " << context_length_ << " tokens\n";
            oss << "Architectural maximum: " << config_.getMaxSequenceLength() << " tokens\n";
            oss << "Embedding dim: " << config_.getModelDim() << "\n";
            oss << "Layers: " << config_.getNumLayers() << "\n";
            oss << "Heads: " << config_.getNumHeads() << "\n";
            oss << "KV heads: " << config_.getNumKVHeads() << "\n";
            oss << "MLP hidden dim: " << config_.getHiddenDimension() << "\n";
            oss << "RoPE theta: " << config_.getRoPETheta() << "\n";

            return oss.str();
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
         * @param params         Per-call generation parameters (loop bound + sampling).
         * @param stop           Stop token for cooperative cancellation.
         * @return               Why generation stopped.
         */
        GenerateStatus onGenerating(
            std::span<const int32_t> prompt_tokens,
            const std::function<void( int32_t )>& on_token,
            const GenerateParams& params,
            std::stop_token stop ) override
        {
            // Stop set: model defaults, or the caller's per-call override.
            std::unordered_set<int32_t> stop_ids;
            if ( params.stop_tokens.empty() )
                stop_ids = stopTokens();
            else
                for ( auto id : params.stop_tokens )
                    stop_ids.insert( static_cast<int32_t>( id ) );

            // Host sampler path (device-sampler migration deferred): time-seeded.
            // Seedable/reproducible sampling arrives with the device-sampler migration
            // (LanguageModel::seedSampler), not a per-call parameter.
            std::mt19937 rng( static_cast<std::mt19937::result_type>(
                std::chrono::high_resolution_clock::now().time_since_epoch().count() ) );

            if ( prompt_tokens.size() > static_cast<size_t>( context_length_ ) )
            {
                throw std::invalid_argument( std::format(
                    "LlamaModel::onGenerating: prompt length {} exceeds deployment context length {}",
                    prompt_tokens.size(), context_length_ ) );
            }

            const int64_t seq_len = static_cast<int64_t>( prompt_tokens.size() );

            auto prefill_input = makeTokenTensor( prompt_tokens );

            auto& logits = this->getNetwork().prefill( prefill_input );
            this->getNetwork().synchronize();

            int32_t next_token = sampleFromLogits(
                logits, 0, params.sampling.temperature, params.sampling.top_k, rng );

            if ( stop_ids.contains( next_token ) )
                return GenerateStatus::Success;

            on_token( next_token );

            dim_t position = seq_len;
            const int max_new = params.max_new_tokens.value_or( static_cast<int>( context_length_ ) );

            for ( int step = 1; step < max_new; ++step )
            {
                if ( stop.stop_requested() )
                    return GenerateStatus::ClientCancelled;

                // decode cannot write the KV cache at a position past the deployment context
                // length. RoPE is computed rather than looked up, so there is no positional
                // table to run off the end of and nothing crashes -- the cache is overrun
                // quietly instead, which is the worse failure. Mirrors GemmaModel.
                if ( position >= context_length_ )
                    return GenerateStatus::ContextOverflow;

                decode_token_staging_.data()[ 0 ] = next_token;
                copy( decode_token_staging_, decode_token_device_ );

                auto& decode_logits = this->getNetwork().decode( decode_token_device_, position );
                this->getNetwork().synchronize();

                next_token = sampleFromLogits(
                    decode_logits, 0, params.sampling.temperature, params.sampling.top_k, rng );

                if ( stop_ids.contains( next_token ) )
                    return GenerateStatus::Success;

                on_token( next_token );
                ++position;
            }

            return GenerateStatus::MaxNewTokensReached;
        }

        /**
         * @brief Training loop -- not yet implemented for LlamaModel.
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
        dim_t maxSequenceLength() const noexcept override
        {
            return config_.getMaxSequenceLength();
        }

        /**
         * @brief Vocabulary size from LLaMA config.
         */
        dim_t vocabSize() const noexcept override
        {
            return config_.getVocabSize();
        }

    private:

        explicit LlamaModel(
            std::unique_ptr<LanguageModelNetwork<TDeviceType, TPrecision>> network,
            const LlamaConfig& config,
            const DeploymentPlan& plan,
            RuntimeMode runtime_mode,
            Serialization::WeightsMetadata source_metadata )
            : ModelBase( std::move( network ), runtime_mode,
                std::move( source_metadata ), plan.weightQuantization() )
            , config_( config ), context_length_( plan.contextLength() ), plan_( plan )
            , decode_token_staging_( TDeviceType == DeviceType::Cuda ? this->getDeviceId() : Device::Cpu(), shape_t{ 1, 1 } )
            , decode_token_device_( this->getDeviceId(), shape_t{ 1, 1 } )
            , logits_staging_( TDeviceType == DeviceType::Cuda ? this->getDeviceId() : Device::Cpu(), shape_t{ 1, 1, static_cast<int64_t>( config.getVocabSize() ) } )
        {}

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

        template<WeightQuantPolicy TWeightQuantization, KvCachePolicy TKvCachePolicy>
        static std::expected<DeploymentPlans, DeploymentRefusal> planImpl(
            const std::filesystem::path& path,
            const DeploymentRequest& request,
            DeviceId device_id )
        {
            WeightsReader reader( path );
            const auto& metadata = reader.getWeightsMetadata();

            requireStoredQuantizationMatches(
                "LlamaModel::planDeployment", path.string(), reader.getWeightQuantization(),
                request.getWeightQuantization() );

            const LlamaConfig network_config = configFromMetadata( metadata );

            // Construction commits no device memory, but it creates the execution context, which holds some;
            // the reading is taken after it, as the load's build will find the device (Deployment.md 9).
            const LlamaTransformer<TDeviceType, TPrecision, TWeightQuantization, TKvCachePolicy> network(
                metadata.model_name, network_config, device_id );

            return planOnDevice( network, request, DeviceReading::take( device_id ),
                network_config.getMaxSequenceLength(), metadata, reader.getWeightQuantization() );
        }

        template<WeightQuantPolicy TWeightQuantization, KvCachePolicy TKvCachePolicy>
        static std::unique_ptr<LlamaModel<TDeviceType, TPrecision>> loadImpl(
            const std::filesystem::path& path,
            const DeploymentPlan& plan )
        {
            WeightsReader reader( path );
            const auto& metadata = reader.getWeightsMetadata();

            plan.requirePricedFor( "LlamaModel::load", path.string(), metadata, reader.getWeightQuantization() );

            requireStoredQuantizationMatches(
                "LlamaModel::load", path.string(), reader.getWeightQuantization(),
                plan.weightQuantization() );

            const LlamaConfig network_config = configFromMetadata( metadata );

            using ConcreteTransformerType = LlamaTransformer<TDeviceType, TPrecision, TWeightQuantization, TKvCachePolicy>;
            auto network = std::make_unique<ConcreteTransformerType>( metadata.model_name, network_config, plan.device() );

            network->build( plan.buildContext() );

            Logging::Logger::info( network->toString() );

            network->loadParameters( reader );

            return std::unique_ptr<LlamaModel<TDeviceType, TPrecision>>(
                new LlamaModel<TDeviceType, TPrecision>(
                    std::move( network ), network_config, plan, RuntimeMode::Inference, metadata ) );
        }

        /**
         * @brief The footprint sibling of loadImpl: same prologue, stops before build().
         */
        template<WeightQuantPolicy TWeightQuantization, KvCachePolicy TKvCachePolicy>
        static DeploymentFootprint deploymentFootprintImpl(
            const std::filesystem::path& path,
            const LlamaModelConfig& model_config,
            DeviceId device_id )
        {
            WeightsReader reader( path );
            const auto& metadata = reader.getWeightsMetadata();

            requireStoredQuantizationMatches(
                "LlamaModel::getDeploymentFootprint", path.string(),
                reader.getWeightQuantization(), model_config.getWeightQuantization() );

            LlamaConfig network_config = configFromMetadata( metadata );

            if ( model_config.getContextLength() > network_config.getMaxSequenceLength() )
            {
                throw std::invalid_argument( std::format(
                    "LlamaModel::getDeploymentFootprint: context_length {} exceeds max_seq_len {}",
                    model_config.getContextLength(),
                    network_config.getMaxSequenceLength() ) );
            }

            using ConcreteTransformerType =
                LlamaTransformer<TDeviceType, TPrecision, TWeightQuantization, TKvCachePolicy>;

            // Construction commits no device memory -- the graph exists and is asked, not built.
            auto network = std::make_unique<ConcreteTransformerType>(
                metadata.model_name, network_config, device_id );

            const dim_t context_length = static_cast<dim_t>( model_config.getContextLength() );

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

        LlamaConfig config_;
        int64_t context_length_;

        // The plan this model executed; reported, never re-derived.
        DeploymentPlan plan_;

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

        TokenIndexType makeTokenTensor( std::span<const int32_t> token_ids ) const
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
            dim_t position,
            float temperature,
            int top_k,
            std::mt19937& rng )
        {
            copy( logits, logits_staging_ );

            const float* row = logits_staging_.data() + position * config_.getVocabSize();

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
    };
}
