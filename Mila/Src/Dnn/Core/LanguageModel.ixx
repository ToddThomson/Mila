/**
 * @file LanguageModel.ixx
 * @brief Abstract base for Mila language models.
 *
 * Extends Model with the abstract contract and generation helpers
 * common to all autoregressive language models.
 *
 * ## Architecture
 *
 *   Model
 *   └── LanguageModel
 *       ├── GptModel
 *       └── LlamaModel
 *
 * ## Contract
 *
 * Derived classes must implement generateStreaming() — the full prefill
 * + decode loop — plus the pure virtual accessors below. The public
 * generate() and generateAsync() are concrete and delegate to it.
 *
 * ## Helpers
 *
 * Protected helpers are provided for derived classes to use in their
 * generateStreaming() implementation. A model with a fundamentally
 * different generation strategy may ignore them entirely.
 *
 * ## Threading
 *
 * Not thread-safe. External synchronization required if shared.
 * generateAsync() must not be called while a previous call's future
 * is still pending on the same model instance.
 */
module;
#include <vector>
#include <span>
#include <memory>
#include <string>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <format>
#include <functional>
#include <future>
#include <stop_token>

export module Dnn.LanguageModel;

import Dnn.Model;
import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
import Dnn.TokenStreamer;
import Compute.Device;
import Compute.DeviceType;
import Compute.DeviceId;
import Compute.DeviceTypeTraits;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class LanguageModel : public Model<TDeviceType, TPrecision>
    {
    public:

        using Base = Model<TDeviceType, TPrecision>;
        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;
        using TokenIndexType = Tensor<TensorDataType::INT32, MR>;

        LanguageModel( const LanguageModel& ) = delete;
        LanguageModel& operator=( const LanguageModel& ) = delete;
        LanguageModel( LanguageModel&& ) = default;
        LanguageModel& operator=( LanguageModel&& ) = default;

        virtual ~LanguageModel() = default;

        // ====================================================================
        // Inference API
        // ====================================================================

        /**
         * @brief Blocking wrapper that collects the full token sequence.
         *
         * Delegates to generateStreaming() with a collector callback.
         * Returns the prompt tokens followed by all generated tokens
         * (EOS excluded).
         *
         * @param prompt_tokens  Input token ids.
         * @param max_new_tokens Maximum tokens to generate beyond the prompt.
         * @param temperature    Sampling temperature. <= 0 selects argmax.
         * @param top_k          Top-k filter. 0 disables.
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
         * @brief Launch generateStreaming() on a background thread.
         *
         * on_token is invoked on the worker thread for every generated token
         * (EOS excluded). The caller must ensure on_token is safe to call
         * from a thread other than the caller's.
         *
         * Not re-entrant: do not call while a previous generateAsync() future
         * is still pending on this model instance.
         *
         * @param prompt_tokens  Input token ids.
         * @param on_token       Per-token callback; invoked on the worker thread.
         * @param max_new_tokens Maximum tokens to generate beyond the prompt.
         * @param temperature    Sampling temperature. <= 0 selects argmax.
         * @param top_k          Top-k filter. 0 disables.
         * @param stop           Stop token; stop_requested() aborts generation early.
         * @return               Future that becomes ready when generation completes
         *                       or propagates an exception on failure.
         */
        std::future<void> generateAsync(
            std::vector<int32_t> prompt_tokens,
            std::function<void(int32_t)> on_token,
            size_t max_new_tokens = 64,
            float temperature = 1.0f,
            int top_k = 0,
            std::stop_token stop = {} )
        {
            return std::async( std::launch::async,
                [this,
                 prompt = std::move( prompt_tokens ),
                 cb = std::move( on_token ),
                 max_new_tokens, temperature, top_k,
                 stop = std::move( stop )]() mutable
                {
                    generateStreaming( prompt, cb, max_new_tokens, temperature, top_k, stop );
                } );
        }

    protected:

        /**
         * @brief Construct with a fully built network and runtime mode.
         *
         * @param network      Fully built and loaded Network.
         * @param runtime_mode Inference or Training — immutable after construction.
         */
        explicit LanguageModel(
            std::unique_ptr<typename Base::NetworkType> network,
            RuntimeMode runtime_mode )
            : Base( std::move( network ), runtime_mode )
        {}

        // ====================================================================
        // Streaming hook — derived class implements the full decode loop
        // ====================================================================

        /**
         * @brief Autoregressively generate tokens, calling on_token for each.
         *
         * Derived classes own the full prefill + decode loop. on_token is
         * invoked for every generated token except EOS. Implementors must
         * check stop.stop_requested() on each decode step and break early
         * when signalled.
         *
         * @param prompt_tokens  Input token ids.
         * @param on_token       Callback invoked once per generated token (not EOS).
         * @param max_new_tokens Maximum tokens to generate beyond the prompt.
         * @param temperature    Sampling temperature. <= 0 selects argmax.
         * @param top_k          Top-k filter. 0 disables.
         * @param stop           Stop token for cooperative cancellation.
         */
        virtual void generateStreaming(
            const std::vector<int32_t>& prompt_tokens,
            const std::function<void(int32_t)>& on_token,
            size_t max_new_tokens,
            float temperature,
            int top_k,
            std::stop_token stop ) = 0;

        // ====================================================================
        // Pure virtual accessors — derived class provides from its config
        // ====================================================================

        /**
         * @brief End-of-sequence token id for this model.
         */
        virtual int32_t eosToken() const noexcept = 0;

        /**
         * @brief Maximum sequence length for this model.
         */
        virtual int64_t maxSequenceLength() const noexcept = 0;

        /**
         * @brief Vocabulary size for this model.
         */
        virtual int64_t vocabSize() const noexcept = 0;

        // ====================================================================
        // Generation helpers — available to derived classes, not imposed
        // ====================================================================

        /**
         * @brief Truncate token sequence to fit within maxSequenceLength().
         *
         * Removes tokens from the start, preserving the most recent context.
         *
         * @param tokens Token sequence to truncate in place.
         */
        void truncateIfNeeded( std::vector<int32_t>& tokens ) const
        {
            int64_t seq_len = static_cast<int64_t>(tokens.size());

            if ( seq_len > maxSequenceLength() )
            {
                tokens.erase(
                    tokens.begin(),
                    tokens.begin() + (seq_len - maxSequenceLength()) );
            }
        }

        /**
         * @brief Create a device token tensor from a vector of token ids.
         *
         * @param token_ids Token ids to copy to device.
         * @return          Device tensor of shape [1, token_ids.size()].
         */
        TokenIndexType makeTokenTensor( const std::vector<int32_t>& token_ids ) const
        {
            shape_t shape = { 1, static_cast<int64_t>(token_ids.size()) };
            TokenIndexType device_tensor( Base::getDeviceId(), shape );
            Tensor<TensorDataType::INT32, CpuMemoryResource> cpu_tensor(
                Device::Cpu(), shape );
            std::memcpy( cpu_tensor.data(), token_ids.data(),
                token_ids.size() * sizeof( int32_t ) );
            copy( cpu_tensor, device_tensor );

            return device_tensor;
        }

        /**
         * @brief Sample the next token from logits at a given sequence position.
         *
         * Copies logits to host, extracts the row at position, then
         * delegates to sampleToken().
         *
         * @param logits      Device logits tensor of shape [1, seq_len, vocab_size].
         * @param position    Sequence position to sample from.
         * @param temperature Sampling temperature.
         * @param top_k       Top-k filter. 0 disables.
         * @param rng         Random number generator.
         * @return            Sampled token id.
         */
        int32_t sampleFromLogits(
            const TensorType& logits,
            int64_t position,
            float temperature,
            int top_k,
            std::mt19937& rng ) const
        {
            int64_t seq_len = logits.shape()[ 1 ];
            shape_t shape = { 1, seq_len, vocabSize() };
            Tensor<TPrecision, CpuMemoryResource> cpu( Device::Cpu(), shape );
            copy( logits, cpu );

            const float* row = cpu.data()
                + static_cast<size_t>(position)
                * static_cast<size_t>(vocabSize());

            return sampleToken( row,
                static_cast<size_t>(vocabSize()),
                temperature, top_k, rng );
        }

        /**
         * @brief Sample a token from a raw logit distribution.
         *
         * If temperature <= 0 or top_k == 1, returns the argmax. Otherwise
         * applies temperature scaling, optional top-k filtering, and samples
         * from the resulting categorical distribution.
         *
         * @param logits      Pointer to vocab_size raw logit values.
         * @param vocab_size  Number of logit values.
         * @param temperature Sampling temperature.
         * @param top_k       Top-k filter. 0 disables.
         * @param rng         Random number generator.
         * @return            Sampled token id.
         */
        static int32_t sampleToken(
            const float* logits,
            size_t vocab_size,
            float temperature,
            int top_k,
            std::mt19937& rng )
        {
            if ( temperature <= 0.0f || top_k == 1 )
            {
                return static_cast<int32_t>(
                    std::max_element( logits, logits + vocab_size ) - logits);
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
                std::partial_sort(
                    indices.begin(), indices.begin() + top_k, indices.end(),
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