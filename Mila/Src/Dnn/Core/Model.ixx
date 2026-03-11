/**
 * @file Model.ixx
 * @brief Abstract base class for Mila DNN models.
 *
 * Provides the shared lifecycle, runtime mode, generation helpers,
 * and user-facing API boundary for all Mila models.
 *
 * ## Architecture
 *
 * Model sits at the top of the Mila DNN pipeline:
 *
 *   Component          — leaf node, Operation, shape-driven buffer allocation
 *   CompositeComponent — structural aggregation, cascades BuildConfig
 *   Network            — graph topology, forward/backward, KV cache lifecycle
 *   Model              — RuntimeMode, user-facing API, generate/eval/sample
 *
 * ## RuntimeMode
 *
 * A Model is constructed for either Inference or Training — immutable
 * after construction. The mode governs which public API methods are valid:
 *
 * | Mode      | Valid methods                        |
 * |-----------|--------------------------------------|
 * | Inference | generate()                           |
 * | Training  | eval(), sample(), forward/backward   |
 *
 * ## Ownership
 *
 * Model owns the Network via unique_ptr. Derived classes own their
 * model-specific config. The tokenizer is caller-supplied.
 *
 * ## Thread safety
 *
 * Not thread-safe. External synchronization required if shared.
 */

module;
#include <vector>
#include <string>
#include <memory>
#include <random>
#include <chrono>
#include <stdexcept>
#include <format>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>

export module Dnn.Model;
export import :RuntimeMode;

import Dnn.Component;
import Dnn.Network;
import Dnn.Tensor;
import Dnn.TensorTypes;
import Dnn.TensorDataType;
import Compute.DeviceType;
import Compute.DeviceId;
import Compute.DeviceTypeTraits;
import Compute.MemoryResource;
import Compute.CpuMemoryResource;
import Compute.Device;

namespace Mila::Dnn
{
    using namespace Mila::Dnn::Compute;

    export template<DeviceType TDeviceType, TensorDataType TPrecision>
        requires PrecisionSupportedOnDevice<TPrecision, TDeviceType>
    class Model
    {
    public:

        using MR = typename DeviceTypeTraits<TDeviceType>::memory_resource;
        using TensorType = Tensor<TPrecision, MR>;
        using TokenIndexType = Tensor<TensorDataType::INT32, MR>;
        using NetworkType = Network<TDeviceType, TPrecision>;

        // Non-copyable, movable
        Model( const Model& ) = delete;
        Model& operator=( const Model& ) = delete;
        Model( Model&& ) = default;
        Model& operator=( Model&& ) = default;

        virtual ~Model() = default;

        // ====================================================================
        // Runtime Mode
        // ====================================================================

        /**
         * @brief The runtime mode this model was constructed for.
         *
         * Immutable after construction. Governs which public API methods
         * are valid and how the Network allocated its buffers.
         */
        RuntimeMode getRuntimeMode() const noexcept
        {
            return runtime_mode_;
        }

        /**
         * @brief True if this model was constructed for inference.
         *
         * Only generate() is valid. eval() and sample() will throw.
         */
        bool isInferenceMode() const noexcept
        {
            return runtime_mode_ == RuntimeMode::Inference;
        }

        /**
         * @brief True if this model was constructed for training.
         *
         * eval() and sample() are valid. generate() will throw.
         */
        bool isTrainingMode() const noexcept
        {
            return runtime_mode_ == RuntimeMode::Training;
        }

        // ====================================================================
        // Inference API
        // ====================================================================

        /**
         * @brief Autoregressively generate tokens from a prompt.
         *
         * Phase 1 (prefill): runs the full prompt through forward() to
         * populate the KV cache and samples the first new token from the
         * last prompt position.
         *
         * Phase 2 (decode): iterates decode() one token at a time until
         * max_new_tokens is reached or the EOS token is emitted.
         *
         * @param prompt_tokens  Input token ids. Truncated from the start
         *                       if they exceed the model's max sequence length.
         * @param max_new_tokens Maximum tokens to generate beyond the prompt.
         * @param temperature    Sampling temperature. <= 0 selects argmax.
         * @param top_k          Restrict to top-k logits. 0 disables.
         * @return               Full token sequence including the prompt.
         *
         * @throws std::runtime_error if called on a Training-mode model.
         */
        virtual std::vector<int32_t> generate(
            const std::vector<int32_t>& prompt_tokens,
            size_t max_new_tokens = 64,
            float temperature = 1.0f,
            int top_k = 0 )
        {
            ensureInferenceMode( "generate" );

            std::vector<int32_t> tokens = prompt_tokens;
            std::mt19937 rng( std::chrono::high_resolution_clock::now()
                .time_since_epoch().count() );

            truncateIfNeeded( tokens );

            int64_t seq_len = static_cast<int64_t>(tokens.size());
            auto prefill_input = makeTokenTensor( tokens );

            auto& logits = network_->forward( prefill_input );
            network_->getExecutionContext()->synchronize();

            int32_t next_token = sampleFromLogits(
                logits, seq_len - 1, temperature, top_k, rng );
            tokens.push_back( next_token );

            if ( next_token == eosToken() )
            {
                return tokens;
            }

            int position = static_cast<int>(seq_len);

            for ( size_t step = 1; step < max_new_tokens; ++step )
            {
                auto decode_input = makeTokenTensor( { next_token } );
                auto& decode_logits = network_->decode( decode_input, position );
                network_->getExecutionContext()->synchronize();

                next_token = sampleFromLogits(
                    decode_logits, 0, temperature, top_k, rng );
                tokens.push_back( next_token );
                ++position;

                if ( next_token == eosToken() )
                {
                    break;
                }
            }

            return tokens;
        }

        // ====================================================================
        // Training API
        // ====================================================================

        /**
         * @brief Run a single evaluation pass over the provided logits and targets.
         *
         * Internally transitions the network to evaluation mode via the
         * protected setEvaluation() pathway, runs the forward pass, then
         * restores training mode. The mode transition is invisible to the caller.
         *
         * @throws std::runtime_error if called on an Inference-mode model.
         */
        virtual float eval(
            const TokenIndexType& input,
            const TokenIndexType& targets )
        {
            ensureTrainingMode( "eval" );

            network_->setEvaluation( true );

            auto& logits = network_->forward( input );
            network_->synchronize();

            float loss = computeLoss( logits, targets );

            network_->setEvaluation( false );

            return loss;
        }

        // ====================================================================
        // Accessors
        // ====================================================================

        /**
         * @brief The device this model runs on.
         */
        DeviceId getDeviceId() const noexcept
        {
            return network_->getExecutionContext()->getDeviceId();
        }

        /**
         * @brief Current memory allocation breakdown for this model.
         */
        MemoryStats getMemoryStats() const
        {
            return network_->getMemoryStats();
        }

        // ====================================================================
        // Diagnostics
        // ====================================================================

        virtual std::string toString() const = 0;

    protected:

        /**
         * @brief Construct a Model with the given network and runtime mode.
         *
         * Called by derived class constructors only. The network must already
         * be built and have parameters loaded before this constructor is called.
         *
         * @param network      Fully built and loaded Network.
         * @param runtime_mode Inference or Training — immutable after construction.
         */
        explicit Model( std::unique_ptr<NetworkType> network, RuntimeMode runtime_mode )
            : network_( std::move( network ) ), runtime_mode_( runtime_mode )
        {}

        /**
         * @brief The owned Network instance.
         *
         * Derived classes may access network_ directly for model-specific
         * operations not covered by the base class API.
         */
        std::unique_ptr<NetworkType> network_;

        // ====================================================================
        // EOS token — derived class provides model-specific value
        // ====================================================================

        /**
         * @brief End-of-sequence token id for this model.
         *
         * Derived classes must override to return the correct EOS token
         * for their vocabulary. Should be sourced from tokenizer metadata
         * once tokenizer integration is complete.
         */
        virtual int32_t eosToken() const noexcept = 0;

        /**
         * @brief Maximum sequence length for this model.
         *
         * Used by truncateIfNeeded() and sampleFromLogits().
         * Derived classes must override to return from their config.
         */
        virtual int64_t maxSequenceLength() const noexcept = 0;

        /**
         * @brief Vocabulary size for this model.
         *
         * Used by sampleFromLogits(). Derived classes must override
         * to return from their config.
         */
        virtual int64_t vocabSize() const noexcept = 0;

        /**
         * @brief Compute loss over logits and targets.
         *
         * Default implementation returns 0. Derived classes or the
         * training harness should override for meaningful eval loss.
         */
        virtual float computeLoss(
            const TensorType& logits,
            const TokenIndexType& targets )
        {
            return 0.0f;
        }

        // ====================================================================
        // Generation helpers
        // ====================================================================

        /**
         * @brief Truncate token sequence to fit within max sequence length.
         *
         * Removes tokens from the start of the sequence, preserving
         * the most recent context.
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
         */
        TokenIndexType makeTokenTensor( const std::vector<int32_t>& token_ids ) const
        {
            shape_t shape = { 1, static_cast<int64_t>(token_ids.size()) };
            TokenIndexType device_tensor( getDeviceId(), shape );
            Tensor<TensorDataType::INT32, CpuMemoryResource> cpu_tensor( Device::Cpu(), shape );
            
            std::memcpy( cpu_tensor.data(), token_ids.data(),
                token_ids.size() * sizeof( int32_t ) );
            
            copy( cpu_tensor, device_tensor );
            
            return device_tensor;
        }

        /**
         * @brief Sample the next token from logits at a given sequence position.
         *
         * Copies logits to host, extracts the row at `position`, then
         * delegates to sampleToken().
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
         * Otherwise applies temperature scaling, optional top-k filtering,
         * and samples from the resulting categorical distribution.
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

    private:

        RuntimeMode runtime_mode_;

        // ====================================================================
        // Precondition guards
        // ====================================================================

        void ensureInferenceMode( const char* method ) const
        {
            if ( runtime_mode_ != RuntimeMode::Inference )
            {
                throw std::runtime_error(
                    std::format( "Model::{}: only valid in Inference mode", method ) );
            }
        }

        void ensureTrainingMode( const char* method ) const
        {
            if ( runtime_mode_ != RuntimeMode::Training )
            {
                throw std::runtime_error(
                    std::format( "Model::{}: only valid in Training mode", method ) );
            }
        }
    };
}