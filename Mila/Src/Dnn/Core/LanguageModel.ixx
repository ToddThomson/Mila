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
import Dnn.GenerateStatus;
import Dnn.Samplers.TokenSampler;
import Dnn.Samplers.SamplingConfig;
import Compute.DeviceType;
import Compute.DeviceTypeTraits;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    /**
     * @brief Statistics captured during a single generateStreaming() call.
     *
     * Carried on the GenerateResult returned by generate() / generateStreaming().
     */
    export struct GenerationStatistics
    {
        /// Number of input prompt tokens processed during prefill.
        std::size_t prompt_tokens{ 0 };

        /// Total tokens generated including the first token produced by prefill.
        std::size_t tokens_generated{ 0 };

        /// Time to first token: prefill forward pass + synchronization + first token sampling (ms).
        float prefill_time_ms{ 0.0f };

        /// Total time spent in the autoregressive decode loop (ms); 0 when only one token was generated.
        float decode_time_ms{ 0.0f };

        /// Decode throughput in tokens per second; 0 when decode loop produced no tokens.
        float decode_tokens_per_second{ 0.0f };

        /// Returns true when at least one generation run has been recorded.
        [[nodiscard]] bool valid() const noexcept
        {
            return prefill_time_ms > 0.0f;
        }
    };

    /**
     * @brief Outcome of a single generate() / generateStreaming() call.
     *
     * Returned by value so generation is reentrant and the model holds no
     * per-run mutable state. @ref status reports why generation stopped,
     * @ref statistics carries the timing/throughput figures, and
     * @ref tokens_generated counts emitted tokens (EOS excluded).
     */
    export struct GenerateResult
    {
        /// Why generation stopped.
        GenerateStatus status{ GenerateStatus::Success };

        /// Timing and throughput for this run.
        GenerationStatistics statistics{};

        /// Tokens emitted to on_token (EOS excluded).
        std::size_t tokens_generated{ 0 };
    };

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
         * @brief Blocking generation. Returns the prompt tokens followed by all
         * generated tokens (EOS excluded).
         *
         * @param prompt_tokens  Input token ids.
         * @param config         Per-call generation configuration.
         * @return               Full token sequence including the prompt.
         */
        std::vector<int32_t> generate(
            const std::vector<int32_t>& prompt_tokens,
            const GenerateConfig& config = {} )
        {
            std::vector<int32_t> out = prompt_tokens;
            out.reserve( prompt_tokens.size() + static_cast<size_t>( config.max_new_tokens ) );

            generateStreaming( prompt_tokens,
                [&]( int32_t tok ) { out.push_back( tok ); },
                config, {} );

            return out;
        }

        /**
         * @brief Synchronous per-token streaming. Blocks on the caller's thread
         * until generation completes or stop is requested.
         *
         * on_token is invoked on the caller's thread for every generated token
         * (EOS excluded). Callers that own their own threading — such as the
         * Python ModelWorker's single-thread executor — should use this directly.
         *
         * @param prompt_tokens  Input token ids.
         * @param on_token       Per-token callback invoked on the caller's thread.
         * @param config         Per-call generation configuration.
         * @param stop           Stop token for cooperative cancellation.
         * @return               Outcome of the run (status + statistics + token count).
         */
        GenerateResult generateStreaming(
            std::span<const int32_t> prompt_tokens,
            std::function<void(int32_t)> on_token,
            const GenerateConfig& config = {},
            std::stop_token stop = {} )
        {
            return onGenerating( prompt_tokens, on_token, config, stop );
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

        /**
         * @brief Seed the device sampler's host RNG for reproducible generation.
         *
         * Call once at the start of a generation run (not per token) when the caller
         * supplied GenerateConfig::seed; the same seed then yields the same token stream
         * for a given prompt and model.
         */
        void seedSampler( uint64_t seed )
        {
            ensureSampler();
            token_sampler_->reseed( seed );
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
         * generation must abort early when signalled, returning a GenerateResult
         * whose status reflects why the loop stopped.
         *
         * @param prompt_tokens  Input token ids.
         * @param on_token       Per-token callback.
         * @param config         Per-call generation configuration.
         * @param stop           Stop token for cooperative cancellation.
         * @return               Outcome of the run (status + statistics + token count).
         */
        virtual GenerateResult onGenerating(
            std::span<const int32_t> prompt_tokens,
            const std::function<void(int32_t)>& on_token,
            const GenerateConfig& config,
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
