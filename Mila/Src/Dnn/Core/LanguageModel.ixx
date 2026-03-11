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
 * Derived classes must implement the full generation contract:
 *
 *   generate()         — autoregressive token generation
 *   eosToken()         — end-of-sequence token id
 *   maxSequenceLength()— maximum supported sequence length
 *   vocabSize()        — vocabulary size
 *   toString()         — human-readable model summary
 *   onTraining()       — training loop (if RuntimeMode::Training)
 *
 * ## Helpers
 *
 * Protected helpers are provided for derived classes to use in
 * their generate() implementation. They are not imposed — a model
 * with a fundamentally different generation strategy (e.g.
 * encoder-decoder) may ignore them entirely.
 *
 * ## Threading
 *
 * Not thread-safe. External synchronization required if shared.
 */
module;
#include <vector>
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

export module Dnn.LanguageModel;

import Dnn.Model;
import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Dnn.TensorDataTypeTraits;
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

        // Non-copyable, movable
        LanguageModel( const LanguageModel& ) = delete;
        LanguageModel& operator=( const LanguageModel& ) = delete;
        LanguageModel( LanguageModel&& ) = default;
        LanguageModel& operator=( LanguageModel&& ) = default;

        virtual ~LanguageModel() = default;

        // ====================================================================
        // Inference API — pure virtual
        // ====================================================================

        /**
         * @brief Autoregressively generate tokens from a prompt.
         *
         * Derived classes own the full implementation. Protected helpers
         * are available for use — truncateIfNeeded(), makeTokenTensor(),
         * sampleFromLogits(), sampleToken() — but are not required.
         *
         * @param prompt_tokens  Input token ids.
         * @param max_new_tokens Maximum tokens to generate beyond the prompt.
         * @param temperature    Sampling temperature. <= 0 selects argmax.
         * @param top_k          Restrict to top-k logits. 0 disables.
         * @return               Full token sequence including the prompt.
         */
        virtual std::vector<int32_t> generate(
            const std::vector<int32_t>& prompt_tokens,
            size_t max_new_tokens = 64,
            float temperature = 1.0f,
            int top_k = 0 ) = 0;

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
        // Pure virtual accessors — derived class provides from its config
        // ====================================================================

        /**
         * @brief End-of-sequence token id for this model.
         *
         * Should be sourced from tokenizer metadata once tokenizer
         * integration is complete.
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
         * Removes tokens from the start, preserving the most recent
         * context. Derived classes may call this at the start of
         * their generate() implementation.
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
         * @return          Device tensor of shape [ 1, token_ids.size() ].
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
         * @param logits      Device logits tensor of shape [ 1, seq_len, vocab_size ].
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
         * @brief Sample a token from a probability distribution.
         *
         * If temperature <= 0 or top_k == 1, returns the argmax.
         * Otherwise applies temperature scaling, optional top-k
         * filtering, and samples from the resulting categorical
         * distribution.
         *
         * @param logits     Pointer to vocab_size raw logit values.
         * @param vocab_size Number of logit values.
         * @param temperature Sampling temperature.
         * @param top_k      Top-k filter. 0 disables.
         * @param rng        Random number generator.
         * @return           Sampled token id.
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
            {
                probs[ i ] /= static_cast<float>( sum );
            }

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
                {
                    probs[ i ] = filtered[ i ] / static_cast<float>( filtered_sum );
                }
            }

            std::uniform_real_distribution<float> dist( 0.0f, 1.0f );
            float r = dist( rng );
            float cumsum = 0.0f;

            for ( size_t i = 0; i < vocab_size; ++i )
            {
                cumsum += probs[ i ];
                if ( r < cumsum )
                {
                    return static_cast<int32_t>( i );
                }
            }

            return static_cast<int32_t>( vocab_size - 1 );
        }
    };
}