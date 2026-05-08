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
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <format>
#include <functional>
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
            std::unique_ptr<typename Base::NetworkType> network,
            RuntimeMode runtime_mode )
            : Base( std::move( network ), runtime_mode )
        {}

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

        // ====================================================================
        // Generation helpers
        // ====================================================================

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
                    std::max_element( logits, logits + vocab_size ) - logits );
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