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

export module Dnn.LanguageModel;

import Dnn.Model;
import Dnn.LanguageNetwork;
import Dnn.Tensor;
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

        explicit LanguageModel(
            std::unique_ptr<LanguageNetwork<TDeviceType, TPrecision>> network,
            RuntimeMode runtime_mode )
            : Base( std::move( network ), runtime_mode )
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
        // Hook — derived class implements the prefill + decode loop
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

        virtual int64_t maxSequenceLength() const noexcept = 0;
        virtual int64_t vocabSize() const noexcept = 0;

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
