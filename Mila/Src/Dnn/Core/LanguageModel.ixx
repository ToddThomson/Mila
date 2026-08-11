/**
 * @file LanguageModel.ixx
 * @brief Abstract base for Mila autoregressive language models.
 *
 * Extends Model with blocking and streaming generation APIs common to all
 * language models. Derived classes implement the prefill + decode loop via
 * the protected onGenerating() hook.
 */
module;
#include <vector>
#include <span>
#include <unordered_set>
#include <memory>
#include <string>
#include <stdexcept>
#include <format>
#include <functional>
#include <stop_token>
#include <cstddef>
#include <cstdint>
#include <filesystem>

export module Dnn.LanguageModel;

import Dnn.Model;
import Dnn.LanguageNetwork;
import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.RuntimeMode;
import Dnn.GenerateParams;
import Dnn.SamplingParams;
import Dnn.GenerateStatus;
import Dnn.Samplers.TokenSampler;
import Dnn.Samplers.SamplingConfig;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;
import Dnn.LanguageModelConfig;
import Serialization.SafeTensors;
import Serialization.PretrainedReader;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class LanguageModel : public Model<TDeviceType, TPrecision>
    {
    public:

        using Base = Model<TDeviceType, TPrecision>;

        LanguageModel( const LanguageModel& ) = delete;
        LanguageModel& operator=( const LanguageModel& ) = delete;
        LanguageModel( LanguageModel&& ) = default;
        LanguageModel& operator=( LanguageModel&& ) = default;

        virtual ~LanguageModel() = default;

        // ====================================================================
        // Public generation API
        // ====================================================================

        /**
         * @brief Generate tokens from a prompt, streaming each through on_token.
         *
         * Blocking, serial token generation: the model owns the decode loop (it owns
         * the KV cache and the device stream) and pushes every generated token (EOS
         * excluded) to on_token on the caller's thread until it stops. Returns why it
         * stopped -- the one outcome the caller cannot reconstruct from the token
         * stream. Timing/throughput are the harness's to measure from the callback
         * cadence; the model keeps no stopwatch. Callers that want asynchrony own the
         * threading (e.g. the Python ModelWorker runs this on its own thread).
         *
         * @param prompt_tokens  Input token ids.
         * @param on_token       Per-token callback invoked on the caller's thread.
         * @param params         Per-call generation parameters (loop bound + sampling).
         * @param stop           Stop token for cooperative cancellation.
         * @return               Why generation stopped.
         */
        [[nodiscard]] GenerateStatus generate(
            std::span<const int32_t> prompt_tokens,
            const std::function<void( int32_t )>& on_token,
            const GenerateParams& params = {},
            std::stop_token stop = {} )
        {
            return onGenerating( prompt_tokens, on_token, params, stop );
        }

        /**
         * @brief Write this model's live weights as a safetensors artifact.
         *
         * The inverse of fromPretrained, and named for it: what this writes is what that
         * reads. Weights go out as they currently sit on the device, so a model loaded under
         * FP4 or FP8 produces a PRE-QUANTIZED artifact -- packed storage plus its scale
         * companions. That is the point of the operation: quantization is a load-time policy,
         * so the quantized bytes exist nowhere until a model has been built with one.
         *
         * The source artifact's metadata is written back verbatim, so the result loads by the
         * same path that loaded the original and is readable by any safetensors reader without
         * Mila.
         *
         * Family-agnostic by construction: the tensor vocabulary comes from the network's own
         * flat-save traversal, so a family is covered as soon as every composite that owns a
         * parameter drives both halves of it. A composite that only recurses and silently drops
         * its own tensors is the failure this cannot see -- the export tool's source
         * reconciliation is what catches that.
         *
         * @param path Destination artifact path; parent directories are created.
         *
         * @throws std::runtime_error if the model carries no pretrained provenance (it was
         *         reconstructed from a checkpoint), or if the file cannot be written.
         */
        void savePretrained( const std::filesystem::path& path ) const
        {
            if ( source_metadata_.architecture.empty() )
            {
                throw std::runtime_error(
                    "LanguageModel::savePretrained: this model carries no pretrained metadata, so "
                    "the artifact would declare no architecture and could not be loaded back. "
                    "Only a model built by fromPretrained can be written as an artifact." );
            }

            Serialization::SafeTensorsWriter writer( path );

            writer.setMetadata(
                Serialization::kMilaConfigMetadataKey,
                Serialization::toMetadataJSON( source_metadata_ ) );

            writer.setMetadata(
                Serialization::kMilaQuantizationMetadataKey,
                weightQuantizationName( weight_quantization_ ) );

            const auto& network = this->getLanguageNetwork();

            // Empty prefix: the root's own name is dropped so tensors land under
            // "tf_layer_0.qkv_proj.weight", the vocabulary loadParameters() reads back.
            network.saveFlatTensors( writer, "", Serialization::TensorSavePass::Declare );

            writer.beginData();

            network.saveFlatTensors( writer, "", Serialization::TensorSavePass::Write );

            writer.close();
        }

        /**
         * @brief Seed the sampler's RNG for reproducible generation.
         *
         * Reproducibility is a property of the RNG stream, not of a single call: seed
         * once (before a run or a session), then the token stream is deterministic for
         * a given prompt and model. Deliberately not a per-call GenerateParams field,
         * so a caller cannot accidentally reset the stream on every call.
         */
        void seedSampler( uint64_t seed )
        {
            ensureSampler();
            token_sampler_->reseed( seed );
        }

    protected:

        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;
        using TokenTensor = Tensor<TensorDataType::INT32, MR>;

        /**
         * @param source_metadata      The loaded artifact's metadata, written back verbatim by
         *                             savePretrained so the result loads by the same path.
         * @param weight_quantization  What the live weights actually are, which is a load-time
         *                             policy rather than a property of the source file.
         *
         * Both default, because a model reconstructed from a checkpoint has no pretrained
         * provenance to carry; savePretrained refuses rather than writing an artifact that
         * declares nothing.
         */
        explicit LanguageModel(
            std::unique_ptr<LanguageNetwork<TDeviceType, TPrecision>> network,
            RuntimeMode runtime_mode,
            Serialization::PretrainedMetadata source_metadata = {},
            WeightQuantization weight_quantization = WeightQuantization::None )
            : Base( std::move( network ), runtime_mode )
            , source_metadata_( std::move( source_metadata ) )
            , weight_quantization_( weight_quantization )
        {}

        // ====================================================================
        // Token sampling (shared device sampler)
        // ====================================================================

        /**
         * @brief Optional final-logit softcap the sampler applies (0 disables).
         *
         * Gemma overrides this with its 30.0 cap; other models leave it at 0.
         */
        virtual float finalLogitSoftcap() const noexcept
        {
            return 0.0f;
        }

        /**
         * @brief Sample the next token from a logits row on the device.
         *
         * Lazily constructs the model-owned TokenSampler on first use (the network is
         * built and the execution context valid by the time generation runs), then samples
         * from the final row of @p logits, writing the int32 token into @p token_out in
         * place (ready for the next decode step) and returning the host value.
         */
        int32_t sampleNext(
            const TensorType& logits,
            TokenTensor& token_out,
            const SamplingParams& params )
        {
            ensureSampler();

            return token_sampler_->sample( logits, token_out, params );
        }

        /**
         * @brief Enqueue a sampling step without waiting for the host readback.
         *
         * Decode-ahead half of the split sampleNext(): the token is written into
         * @p token_out on the device (ready for the next decode step) and its id
         * travels to the host asynchronously. awaitSampledToken() completes the pair.
         * At most one enqueue may be outstanding.
         */
        void enqueueSampleNext(
            const TensorType& logits,
            TokenTensor& token_out,
            const SamplingParams& params )
        {
            ensureSampler();

            token_sampler_->enqueueSample( logits, token_out, params );
        }

        /**
         * @brief Block until the last enqueueSampleNext()'s token id is host-visible.
         *
         * Waits only for that sampling step -- device work enqueued after it (the
         * ahead-decoded forward) keeps running, which is what hides the per-token
         * host gap.
         */
        int32_t awaitSampledToken()
        {
            return token_sampler_->awaitToken();
        }

        // ====================================================================
        // Network accessor
        // ====================================================================

        LanguageNetwork<TDeviceType, TPrecision>& getLanguageNetwork() noexcept
        {
            return static_cast<LanguageNetwork<TDeviceType, TPrecision>&>( *this->network_ );
        }

        const LanguageNetwork<TDeviceType, TPrecision>& getLanguageNetwork() const noexcept
        {
            return static_cast<const LanguageNetwork<TDeviceType, TPrecision>&>( *this->network_ );
        }

        // ====================================================================
        // Hook -- derived class implements the prefill + decode loop
        // ====================================================================

        /**
         * @brief Prefill + decode implementation hook.
         *
         * Derived classes own the full autoregressive generation loop.
         * on_token must be called for every generated token except EOS.
         * stop.stop_requested() must be checked on each decode step and
         * generation must abort early when signalled, returning the
         * GenerateStatus that reflects why the loop stopped.
         *
         * @param prompt_tokens  Input token ids.
         * @param on_token       Per-token callback.
         * @param params         Per-call generation parameters (loop bound + sampling).
         * @param stop           Stop token for cooperative cancellation.
         * @return               Why generation stopped.
         */
        virtual GenerateStatus onGenerating(
            std::span<const int32_t> prompt_tokens,
            const std::function<void( int32_t )>& on_token,
            const GenerateParams& params,
            std::stop_token stop ) = 0;

        // ====================================================================
        // Pure virtual accessors
        // ====================================================================

        virtual int32_t eosToken() const noexcept = 0;

        virtual std::unordered_set<int32_t> stopTokens() const
        {
            return { eosToken() };
        }

        virtual dim_t maxSequenceLength() const noexcept = 0;
        virtual dim_t vocabSize() const noexcept = 0;

        /// The loaded artifact's metadata, carried so savePretrained can write it back verbatim.
        /// Empty for a model reconstructed from a checkpoint, which savePretrained refuses.
        Serialization::PretrainedMetadata source_metadata_;

        /// What the live weights are, not what the source file was.
        WeightQuantization weight_quantization_{ WeightQuantization::None };

    private:

        /// Lazily construct the model-owned sampler on first use (network built, context valid).
        void ensureSampler()
        {
            if ( !token_sampler_ )
            {
                SamplingConfig config = SamplingConfig{}
                    .withVocabularySize( this->vocabSize() )
                    .withFinalLogitSoftcap( this->finalLogitSoftcap() );

                token_sampler_ = std::make_unique<TokenSampler<TDeviceType, TPrecision>>(
                    this->getLanguageNetwork().getExecutionContext(), config );
            }
        }

        std::unique_ptr<TokenSampler<TDeviceType, TPrecision>> token_sampler_;
    };
}
