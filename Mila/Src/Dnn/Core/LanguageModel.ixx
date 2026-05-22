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
#include <unordered_set>
#include <memory>
#include <string>
#include <stdexcept>
#include <format>
#include <functional>
#include <stop_token>

export module Dnn.LanguageModel;

import Dnn.Model;
import Dnn.LanguageNetwork;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Compute.DeviceType;

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
         * @brief Blocking generation. Returns the prompt tokens followed by all
         * generated tokens (EOS excluded).
         *
         * @param prompt_tokens  Input token ids.
         * @param max_new_tokens Maximum tokens to generate beyond the prompt.
         * @param temperature    Sampling temperature; <= 0 selects argmax.
         * @param top_k          Top-k filter; 0 disables.
         * @return               Full token sequence including the prompt.
         */
        std::vector<int32_t> generate(
            const std::vector<int32_t>& prompt_tokens,
            size_t max_new_tokens = 64,
            float temperature = 1.0f,
            int top_k = 0 )
        {
            std::vector<int32_t> out = prompt_tokens;
            out.reserve( prompt_tokens.size() + max_new_tokens );

            generateStreaming( prompt_tokens,
                [&]( int32_t tok ) { out.push_back( tok ); },
                max_new_tokens, temperature, top_k, {} );

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
         * @param max_new_tokens Maximum tokens to generate beyond the prompt.
         * @param temperature    Sampling temperature; <= 0 selects argmax.
         * @param top_k          Top-k filter; 0 disables.
         * @param stop           Stop token for cooperative cancellation.
         */
        void generateStreaming(
            const std::vector<int32_t>& prompt_tokens,
            std::function<void(int32_t)> on_token,
            size_t max_new_tokens = 64,
            float temperature = 1.0f,
            int top_k = 0,
            std::stop_token stop = {} )
        {
            onGenerating( prompt_tokens, on_token, max_new_tokens, temperature, top_k, stop );
        }

    protected:

        explicit LanguageModel(
            std::unique_ptr<LanguageNetwork<TDeviceType, TPrecision>> network,
            RuntimeMode runtime_mode )
            : Base( std::move( network ), runtime_mode )
        {}

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
         * generation must abort early when signalled.
         *
         * @param prompt_tokens  Input token ids.
         * @param on_token       Per-token callback.
         * @param max_new_tokens Maximum tokens to generate beyond the prompt.
         * @param temperature    Sampling temperature; <= 0 selects argmax.
         * @param top_k          Top-k filter; 0 disables.
         * @param stop           Stop token for cooperative cancellation.
         */
        virtual void onGenerating(
            const std::vector<int32_t>& prompt_tokens,
            const std::function<void(int32_t)>& on_token,
            size_t max_new_tokens,
            float temperature,
            int top_k,
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
    };
}
