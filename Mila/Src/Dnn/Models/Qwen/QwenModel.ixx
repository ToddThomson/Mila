/**
 * @file QwenModel.ixx
 * @brief Qwen 3.8 inference model.
 *
 * Inference-only wrapper around a loaded QwenTransformer network. Follows GemmaModel,
 * which is the family this one is modelled on -- both drive a heterogeneous layer stack --
 * with two deliberate divergences, each forced by the Gated DeltaNet mixer:
 *
 *  - NO PROMPT-PREFIX REUSE. A recurrent state is a lossy summary of every position it has
 *    seen, so it cannot be rewound; `QwenDeltaNetBlock::rewindKvCache` always refuses and
 *    the transformer ANDs that into a stack-wide refusal. Gemma's reuse block is therefore
 *    absent here rather than present and permanently failing -- machinery that can never
 *    fire reads as a capability. See Qwen3.8.md section 7.
 *
 *  - THE MIXER STATE IS SELF-CLEANING AT PREFILL, not reset by a call. `prefill` at
 *    position 0 zeroes the recurrent state and starts the convolution window cold
 *    (`GatedDeltaRule::prefill`, `CausalConv1d::prefill`), so a fresh generation cannot
 *    inherit the previous one's state. That is also precisely why `prefillFrom` at a
 *    non-zero offset must never be reached on this family: it would carry state forward
 *    for a prefix the caches no longer hold.
 *
 * REFERENCE PRECISION ONLY, for now. Section 5's allocation is a per-role PLAN over
 * codebook formats, and the artifact that carries it does not exist yet, so this entry
 * point accepts the BF16 reference plan and refuses the uniform quantization modes rather
 * than building something that is not the plan. That refusal is the Phase 4 / Phase 5
 * boundary, and it is named in the error.
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
#include <optional>
#include <algorithm>
#include <functional>
#include <stop_token>
#include <cstring>
#include <type_traits>

export module Dnn.Models.QwenModel;

import Dnn.Models.QwenModelConfig;
import Dnn.LanguageModel;
import Dnn.LanguageModelConfig;
import Dnn.LanguageModelNetwork;
import Dnn.Quantization.Weight.Policies;
import Dnn.Quantization.KvCache.Policy;
import Dnn.Tensor;
import Dnn.ITensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.Component;
import Dnn.RuntimeMode;
import Dnn.Components.QwenTransformer;
import Dnn.Components.QwenConfig;
import Dnn.Components.QwenPrecisionPlan;
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
     * @brief Qwen 3.8 compatible inference model.
     *
     * Owns a loaded, built QwenTransformer and drives the prefill + decode loop.
     * Construction is only possible via fromPretrained().
     *
     * Thread safety: not thread-safe; external synchronization required if shared.
     */
    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class QwenModel : public LanguageModel<TDeviceType, TPrecision>
    {
    public:
        /**
         * @brief KV policy for the full-attention layers.
         *
         * Qwen's cache-holding layers attend the FULL context -- there is no sliding window
         * anywhere in this stack -- so a bounded ring has nothing to bound and the policy is
         * NoKvCompression. The 48 DeltaNet layers hold no cache at all, which is the property
         * that makes a 27B model plausible on 12 GiB (Qwen3.8.md section 3).
         */
        using QwenKvPolicy = NoKvCompression;

        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using ModelBase = LanguageModel<TDeviceType, TPrecision>;
        using TensorType = Tensor<TPrecision, MR>;
        using TokenIndexType = Tensor<dtype_t::INT32, MR>;

        QwenModel( const QwenModel& ) = delete;
        QwenModel& operator=( const QwenModel& ) = delete;
        QwenModel( QwenModel&& ) = default;
        QwenModel& operator=( QwenModel&& ) = default;

        ~QwenModel() = default;

        /**
         * @brief Load from a Mila-converted Qwen 3.8 pretrained artifact.
         *
         * The model_config carries the deployment decisions (context length, weight
         * quantization, KV-cache compression); every architectural parameter is read from
         * the checkpoint metadata.
         *
         * @param path          Path to the pretrained Qwen model artifact.
         * @param model_config  Deployment configuration for this load.
         * @param device_id     Target device; must match TDeviceType.
         * @return              Inference-ready QwenModel.
         *
         * @throws std::invalid_argument on device type mismatch or zero context length.
         * @throws std::runtime_error    on load failure or a quantization mode this chassis
         *                               does not yet carry an artifact for.
         */
        static std::unique_ptr<QwenModel<TDeviceType, TPrecision>> fromPretrained(
            const std::filesystem::path& path,
            const QwenModelConfig& model_config,
            DeviceId device_id = DeviceId{ TDeviceType, 0 } )
        {
            validateRequest( "QwenModel::fromPretrained", model_config, device_id );

            return dispatchQwenWeightPlan<
                    std::unique_ptr<QwenModel<TDeviceType, TPrecision>>>(
                model_config.getWeightQuantization(),
                "QwenModel::fromPretrained",
                [&]<typename TWeightPlan>()
                {
                    return fromPretrainedImpl<TWeightPlan>( path, model_config, device_id );
                } );
        }

        /**
         * @brief What loading this checkpoint at this context length would cost in VRAM.
         *
         * Reads the artifact header for geometry and constructs the graph, then reports what
         * build() would allocate -- without building it, without reading a weight, and
         * therefore without needing the device to have room.
         *
         * @throws std::invalid_argument on device type mismatch or zero context length.
         * @throws std::runtime_error    on an unreadable artifact or an unsupported mode.
         */
        static MemoryStats getRequiredMemory(
            const std::filesystem::path& path,
            const QwenModelConfig& model_config,
            DeviceId device_id = DeviceId{ TDeviceType, 0 } )
        {
            return getDeploymentFootprint( path, model_config, device_id ).memory;
        }

        /**
         * @brief The same prediction, plus how this deployment would chunk its prefill.
         *
         * Everything before the build is shared with fromPretrained deliberately: the
         * artifact check, the geometry and the context-length validation must be the ones a
         * real load would apply, or the reported figure describes a model that would not load.
         *
         * @throws std::invalid_argument on device type mismatch or zero context length.
         * @throws std::runtime_error    on an unreadable artifact or an unsupported mode.
         */
        static DeploymentFootprint getDeploymentFootprint(
            const std::filesystem::path& path,
            const QwenModelConfig& model_config,
            DeviceId device_id = DeviceId{ TDeviceType, 0 } )
        {
            validateRequest( "QwenModel::getDeploymentFootprint", model_config, device_id );

            // The same dispatcher as fromPretrained, deliberately: the footprint path and the
            // load path must reach the identical instantiation, or a model reports a figure it
            // does not allocate. At 2.90 bits against BF16 that error would be a factor of six.
            return dispatchQwenWeightPlan<DeploymentFootprint>(
                model_config.getWeightQuantization(),
                "QwenModel::getDeploymentFootprint",
                [&]<typename TWeightPlan>()
                {
                    return deploymentFootprintImpl<TWeightPlan>(
                        path, model_config, device_id );
                } );
        }

        // ====================================================================
        // Accessors
        // ====================================================================

        /// Architecture/network configuration (read from the checkpoint metadata).
        const QwenConfig& getNetworkConfig() const noexcept
        {
            return config_;
        }

        /// Deployment configuration this model was loaded with.
        const QwenModelConfig& getModelConfig() const noexcept
        {
            return model_config_;
        }

        /// Deployment context length: the KV-cache depth the network was built with.
        dim_t contextLength() const noexcept
        {
            return static_cast<dim_t>( model_config_.getContextLength() );
        }

        /**
         * @brief Teacher-forced log-likelihood of a token sequence -- the perplexity path.
         *
         * Reports the probability the model assigned to each token given the ones before it.
         * Nothing is sampled, so the result depends only on the model and the text, which is
         * what makes it comparable between two quantizations of the same weights.
         *
         * Perplexity over a corpus is exp( -total_log_probability / scored_positions ) after
         * summing several of these. Segment the corpus and sum, rather than averaging each
         * segment, or a short segment weighs as much as a long one.
         *
         * Needs a deployment built with withLanguageModelHeadPositions() above 1 to run at a
         * sensible rate; at the default of 1 it works and costs one head pass per token.
         *
         * @param tokens A single sequence, at least 2 tokens, no longer than contextLength().
         */
        SequenceLogLikelihood scoreTokens( const std::vector<int32_t>& tokens )
        {
            if ( tokens.size() > static_cast<size_t>( contextLength() ) )
            {
                throw std::invalid_argument( std::format(
                    "QwenModel::scoreTokens: sequence of {} tokens exceeds the context length {}",
                    tokens.size(), contextLength() ) );
            }

            auto device_tokens = makeTokenTensor( tokens );

            return this->getNetwork().scoreTokens( device_tokens );
        }

        /**
         * @brief False, always: this stack cannot reuse a prompt prefix.
         *
         * Stated as a model property rather than left for a caller to discover through a
         * refused rewind, because the reason is architectural and permanent -- 48 of the 64
         * layers hold a recurrent state, and a lossy summary cannot be rolled back.
         */
        bool supportsPromptPrefixReuse() const noexcept
        {
            return false;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "QwenModel" << std::endl;
            oss << "  Device: " << this->getDeviceId().toString() << std::endl;
            oss << "  Context length: " << contextLength() << std::endl;
            oss << config_.toString();

            return oss.str();
        }

        dim_t maxSequenceLength() const noexcept override
        {
            return config_.getMaxSequenceLength();
        }

        dim_t vocabSize() const noexcept override
        {
            return config_.getVocabSize();
        }

        /**
         * @brief The network geometry an artifact's metadata declares.
         *
         * Public because loading the model is not the only way to drive it. The Phase 4
         * parity harness streams the 50 GiB artifact one layer at a time and never
         * constructs a QwenModel, but it must build its blocks from the SAME geometry a
         * real load would -- a second reading of these twenty fields would agree until one
         * of them changed, and then measure a different model than the one it is checking.
         */
        static QwenConfig configFromMetadata( const PretrainedMetadata& metadata )
        {
            QwenConfig config(
                static_cast<dim_t>(metadata.embedding_dim),
                static_cast<dim_t>(metadata.num_layers) );

            config.withVocabularyLength( static_cast<dim_t>(metadata.vocab_size) )
                .withMaxSequenceLength( static_cast<dim_t>(metadata.max_seq_length) )
                .withNumHeads( static_cast<dim_t>(metadata.num_heads) )
                .withNumKVHeads( static_cast<dim_t>(metadata.num_kv_heads) )
                .withHeadDim( static_cast<dim_t>(metadata.head_dim) )
                .withAttentionOutputGate( metadata.attention_output_gate )
                .withHiddenDimension( static_cast<dim_t>(metadata.hidden_dim) )
                .withRMSNormEpsilon( metadata.norm_epsilon )
                .withRoPETheta( metadata.rope_theta )
                .withPartialRotaryFactor( metadata.partial_rotary_factor )
                .withFullAttentionInterval( static_cast<dim_t>(metadata.full_attention_interval) )
                .withTieWordEmbeddings( metadata.tie_word_embeddings )
                .withLinearNumKeyHeads( static_cast<dim_t>(metadata.linear_num_key_heads) )
                .withLinearNumValueHeads( static_cast<dim_t>(metadata.linear_num_value_heads) )
                .withLinearHeadDim( static_cast<dim_t>(metadata.linear_head_dim) )
                .withLinearConvKernelDim( static_cast<dim_t>(metadata.linear_conv_kernel_dim) );

            return config;
        }

    protected:

        GenerateStatus onGenerating(
            std::span<const int32_t> prompt_tokens,
            const std::function<void( int32_t )>& on_token,
            const GenerateParams& params,
            std::stop_token stop ) override
        {
            // The prompt must fit the deployment context (the KV-cache depth the network was
            // built with). Context management is the caller's concern; reject rather than
            // silently truncate.
            if ( prompt_tokens.size() > static_cast<size_t>( contextLength() ) )
            {
                throw std::invalid_argument( std::format(
                    "QwenModel::onGenerating: prompt length {} exceeds deployment context length {}",
                    prompt_tokens.size(), contextLength() ) );
            }

            std::unordered_set<int32_t> stop_ids;

            if ( params.stop_tokens.empty() )
                stop_ids = stopTokens();
            else
                for ( auto id : params.stop_tokens )
                    stop_ids.insert( static_cast<int32_t>( id ) );

            const int64_t seq_len = static_cast<int64_t>( prompt_tokens.size() );
            auto prefill_input = makeTokenTensor( prompt_tokens );

            // Always a full prefill from position 0. That is what zeroes the recurrent state
            // and starts the convolution window cold, so this call is also what makes the
            // generation independent of whatever ran before it. No prefix reuse exists here
            // -- see the file header.
            auto& logits = this->getNetwork().prefill( prefill_input );

            // Decode-ahead pipeline: the sampler runs on the network stream (ordered after
            // the forward that produced the logits) and writes the sampled token into
            // decode_token_device_ in place, so the NEXT forward is enqueued before the host
            // has read the token id back.
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

                // Decode ahead only when another step could consume its logits: within the
                // per-call token budget, and with KV-cache room -- decode cannot write at a
                // position past the deployment context length.
                const bool more_steps_allowed = emitted + 1 < max_new;
                const bool cache_has_room = position < contextLength();

                TensorType* decode_logits = nullptr;

                if ( more_steps_allowed && cache_has_room )
                    decode_logits = &this->getNetwork().decode( decode_token_device_, position );

                const int32_t token = this->awaitSampledToken();

                if ( decode_logits )
                    ++position;

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
                    // No ahead-decode was enqueued, so the stream drained at the await above.
                    // Token budget takes precedence over the context bound.
                    return more_steps_allowed
                        ? GenerateStatus::ContextOverflow
                        : GenerateStatus::MaxNewTokensReached;
                }

                this->enqueueSampleNext( *decode_logits, decode_token_device_, params.sampling );
            }
        }

        void onTraining() override
        {
            throw std::runtime_error( "QwenModel::onTraining: Qwen is inference-only" );
        }

        int32_t eosToken() const noexcept override
        {
            return kEndOfTurnToken;
        }

        std::unordered_set<int32_t> stopTokens() const override
        {
            return { kEndOfTurnToken, kEndOfTextToken };
        }

    private:

        explicit QwenModel(
            std::unique_ptr<LanguageModelNetwork<TDeviceType, TPrecision>> network,
            const QwenConfig& config,
            const QwenModelConfig& model_config,
            const PretrainedMetadata& source_metadata,
            RuntimeMode runtime_mode )
            : ModelBase( std::move( network ), runtime_mode,
                source_metadata, model_config.getWeightQuantization() )
            , config_( config )
            , model_config_( model_config )
            , decode_token_device_( this->getDeviceId(), shape_t{ 1, 1 } )
        {}

        // Architecture config (from checkpoint metadata): the trained network geometry.
        QwenConfig config_;

        // Deployment config this model was loaded with. The deployment context length is the
        // KV-cache depth the network was BUILT with -- it may be far below the architectural
        // max, and the prompt check and decode loop bound against THIS.
        QwenModelConfig model_config_;

        // Device decode-input buffer: the sampler writes the next token here in place, and
        // decode() reads it directly -- no host staging round-trip.
        TokenIndexType decode_token_device_;

        // Qwen 3.8 instruct stop tokens, read from the checkpoint's own tokenizer_config and
        // generation_config: <|im_end|> is the conversational EOS and <|endoftext|> the
        // document terminator; generation_config lists both as eos_token_id. These are the
        // MODEL defaults -- a harness that owns the tokenizer may override the stop set per
        // call via GenerateParams::stop_tokens.
        static constexpr int32_t kEndOfTurnToken = 248046;  // <|im_end|>
        static constexpr int32_t kEndOfTextToken = 248044;  // <|endoftext|>

        /**
         * @brief Resolve a deployment's weight setting to one of this family's two plans.
         *
         * Qwen has its own dispatcher rather than using the shared `dispatchWeightQuantization`
         * for the reason `Qwen.PrecisionPlan.ixx` gives for its own lift: the shared one yields
         * a UNIFORM policy, and Section 5's allocation is a per-role plan whose roles include
         * two this family invented. Routing Qwen through it would either widen a shared table
         * with a family's types or silently build the wrong body.
         *
         * Three plans are reachable, and the third is a measurement rather than a deployment.
         * Uniform FP4 is not "Qwen at lower precision" in any sense the chassis is designed
         * around -- it spends 4.125 bits everywhere, 12.31 GiB of weights, and does not fit
         * the 12 GiB target card. It exists because Section 5's exit criterion is a RATIO:
         * the 2.82-bit allocation has to be measured against a near-lossless build of the
         * same weights on the same corpus, and that oracle is what the 16 GiB card is for
         * (Qwen3.8.md, "The 16 GiB oracle"). It loads from the reference BF16 blob and
         * quantizes on the way in, so no artifact needs to carry it.
         *
         * Uniform FP8 stays refused. Nothing measures against it and it fits no card here.
         */
        template<typename TResult, typename TAction>
        static TResult dispatchQwenWeightPlan(
            WeightQuantization weight_quantization,
            std::string_view caller,
            TAction&& action )
        {
            switch ( weight_quantization )
            {
                case WeightQuantization::None:
                    return action.template operator()<QwenReferencePrecisionPlan>();

                case WeightQuantization::FP4:
                    if constexpr ( TPrecision == TensorDataType::BF16 )
                    {
                        return action.template operator()<QwenOraclePrecisionPlan>();
                    }
                    else
                    {
                        throw std::runtime_error( std::format(
                            "{}: the FP4 oracle quantizes BF16 weights on load and requires "
                            "BF16 compute precision", caller ) );
                    }

                case WeightQuantization::Plan:
                    if constexpr ( TPrecision == TensorDataType::BF16 )
                    {
                        return action.template operator()<QwenPrecisionPlan>();
                    }
                    else
                    {
                        throw std::runtime_error( std::format(
                            "{}: the Section 5 allocation stages its codebook weights through "
                            "BF16 and requires BF16 compute precision", caller ) );
                    }

                default:
                    throw std::runtime_error( std::format(
                        "{}: this chassis loads Qwen at reference precision, under its own "
                        "per-role plan, or at uniform FP4 as the Section 5 oracle. A uniform "
                        "FP8 body is a different allocation than Section 5's, nothing measures "
                        "against it, and no artifact carries it", caller ) );
            }
        }

        /**
         * @brief The load path, once the plan is a type.
         */
        template<typename TWeightPlan>
        static std::unique_ptr<QwenModel<TDeviceType, TPrecision>> fromPretrainedImpl(
            const std::filesystem::path& path,
            const QwenModelConfig& model_config,
            DeviceId device_id )
        {
            using ConcreteTransformerType =
                QwenTransformer<TDeviceType, TPrecision, TWeightPlan, QwenKvPolicy>;

            PretrainedModelReader reader( path );
            const auto& metadata = reader.getPretrainedMetadata();

            QwenConfig network_config = configFromMetadata( metadata );

            // A deployment choice rather than checkpoint geometry, so it is applied AFTER the
            // metadata: the artifact says how wide the vocabulary is, the caller says how many
            // rows of it this deployment needs at once.
            network_config.withLanguageModelHeadPositions(
                model_config.getLanguageModelHeadPositions() );

            validateArtifact( "QwenModel::fromPretrained", path, reader, model_config,
                network_config );

            auto network = std::make_unique<ConcreteTransformerType>(
                metadata.model_name, network_config, device_id );

            BuildContext build_context(
                shape_t{ 1, static_cast<dim_t>( model_config.getContextLength() ) },
                RuntimeMode::Inference,
                false );

            network->build( build_context );

            Logging::Logger::info( network->toString() );

            network->loadParameters( reader );

            return std::unique_ptr<QwenModel<TDeviceType, TPrecision>>(
                new QwenModel<TDeviceType, TPrecision>(
                    std::move( network ), network_config,
                    model_config, metadata, RuntimeMode::Inference ) );
        }

        /**
         * @brief The footprint sibling: the same prologue, stopping before build().
         *
         * Everything above `network->build()` is shared with the load path deliberately -- the
         * artifact check, the geometry and the context-length validation must be the ones a
         * real load would apply, or the reported figure describes a model that would not load.
         */
        template<typename TWeightPlan>
        static DeploymentFootprint deploymentFootprintImpl(
            const std::filesystem::path& path,
            const QwenModelConfig& model_config,
            DeviceId device_id )
        {
            using ConcreteTransformerType =
                QwenTransformer<TDeviceType, TPrecision, TWeightPlan, QwenKvPolicy>;

            PretrainedModelReader reader( path );
            const auto& metadata = reader.getPretrainedMetadata();

            QwenConfig network_config = configFromMetadata( metadata );

            // A deployment choice rather than checkpoint geometry, so it is applied AFTER the
            // metadata: the artifact says how wide the vocabulary is, the caller says how many
            // rows of it this deployment needs at once.
            network_config.withLanguageModelHeadPositions(
                model_config.getLanguageModelHeadPositions() );

            validateArtifact( "QwenModel::getDeploymentFootprint", path, reader, model_config,
                network_config );

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

        static void validateRequest(
            std::string_view caller,
            const QwenModelConfig& model_config,
            DeviceId device_id )
        {
            if ( device_id.type != TDeviceType )
            {
                throw std::invalid_argument( std::format(
                    "{}: device type mismatch: expected {}, got {}",
                    caller,
                    deviceTypeToString( TDeviceType ),
                    deviceTypeToString( device_id.type ) ) );
            }

            if ( model_config.getContextLength() == 0 )
            {
                throw std::invalid_argument(
                    std::format( "{}: context_length must be greater than zero", caller ) );
            }

            // The uniform modes are refused in dispatchQwenWeightPlan, where the reason is a
            // scope boundary rather than a dispatch gap. Nothing to check here.

            if ( model_config.getKvCacheCompression() == KvCacheCompression::FP8 )
            {
                throw std::runtime_error( std::format(
                    "{}: FP8 KV cache compression is not yet supported", caller ) );
            }
        }

        /**
         * @brief Checks that must hold for both the load and the footprint path.
         *
         * The geometry check is the one that matters here: a Qwen artifact whose metadata
         * predates the Qwen fields parses as all-zero, and a zeroed interleave would build a
         * stack of the wrong block kinds. Caught against the artifact rather than left to
         * surface as a missing-tensor error 40 layers in.
         */
        static void validateArtifact(
            std::string_view caller,
            const std::filesystem::path& path,
            const PretrainedModelReader& reader,
            const QwenModelConfig& model_config,
            const QwenConfig& network_config )
        {
            // This family is the one that most needs the derivable-format asymmetry: the
            // Phase 5 FP4 oracle is uniform PerGroupFp4 over the reference blob, and refusing
            // it would put that behind a repack reproducing what the load does anyway.
            requireStoredQuantizationMatches(
                caller, path.string(), reader.getWeightQuantization(),
                model_config.getWeightQuantization() );

            const auto& metadata = reader.getPretrainedMetadata();

            if ( metadata.full_attention_interval == 0 )
            {
                throw std::runtime_error( std::format(
                    "{}: artifact '{}' carries no full_attention_interval -- it was written "
                    "before the converter emitted the Qwen geometry, and the mixer interleave "
                    "cannot be guessed", caller, path.string() ) );
            }

            if ( model_config.getContextLength() > network_config.getMaxSequenceLength() )
            {
                throw std::invalid_argument( std::format(
                    "{}: context_length {} exceeds trained max_seq_len {}",
                    caller, model_config.getContextLength(),
                    network_config.getMaxSequenceLength() ) );
            }
        }

        TokenIndexType makeTokenTensor( std::span<const int32_t> token_ids ) const
        {
            shape_t shape = { 1, static_cast<int64_t>( token_ids.size() ) };
            TokenIndexType device_tensor( this->getDeviceId(), shape );

            Tensor<dtype_t::INT32, CpuMemoryResource> cpu_tensor( Device::Cpu(), shape );

            std::memcpy( cpu_tensor.data(), token_ids.data(), token_ids.size() * sizeof( int32_t ) );

            copy( cpu_tensor, device_tensor );

            return device_tensor;
        }

    };
}
