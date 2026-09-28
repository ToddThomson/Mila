/**
 * @file SequenceLogLikelihood.ixx
 * @brief The log-likelihood a language model assigns to a given token sequence.
 */

module;
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <vector>

export module Dnn.SequenceLogLikelihood;

import Dnn.TensorTypes;

namespace Mila::Dnn
{
    /**
     * @brief The log-probability one row of logits gives its target token.
     *
     * The softcap, when positive, is applied to every logit as cap * tanh( logit / cap ) before the
     * log-softmax. The row maximum is subtracted before exponentiating, and the exponentials are summed in
     * double. The device reduction (NextTokenLogProbabilityOp) computes the same quantity; this is its host
     * reference and the CPU op's arithmetic.
     *
     * @param logits              One row of vocab_size logits, host, FP32.
     * @param vocab_size          Width of the row.
     * @param target              The token that actually followed; the caller checks it is in the vocabulary.
     * @param final_logit_softcap 0 means none.
     */
    export inline double nextTokenLogProbability(
        const float* logits, dim_t vocab_size, std::int32_t target, float final_logit_softcap )
    {
        const auto capped = [&]( dim_t v )
        {
            return final_logit_softcap > 0.0f
                ? final_logit_softcap * std::tanh( logits[ v ] / final_logit_softcap )
                : logits[ v ];
        };

        float max_logit = capped( 0 );

        for ( dim_t v = 1; v < vocab_size; ++v )
        {
            max_logit = std::fmax( max_logit, capped( v ) );
        }

        double sum_exponentials = 0.0;

        for ( dim_t v = 0; v < vocab_size; ++v )
        {
            sum_exponentials += std::exp( static_cast<double>( capped( v ) - max_logit ) );
        }

        return static_cast<double>( capped( target ) - max_logit ) - std::log( sum_exponentials );
    }

    /**
     * @brief Refuse a sequence whose targets -- every token after the first -- are not all in the vocabulary.
     *
     * Checked on the host before any window runs: a device reduction cannot throw, and a target outside the
     * vocabulary would otherwise score as NaN.
     *
     * @throws std::out_of_range naming the first token outside [0, vocab_size).
     */
    export inline void requireTokensInVocabulary( const std::int32_t* tokens, dim_t sequence_length, dim_t vocab_size )
    {
        for ( dim_t position = 1; position < sequence_length; ++position )
        {
            const std::int32_t token = tokens[ position ];

            if ( token < 0 || token >= vocab_size )
                throw std::out_of_range( std::format(
                    "sequenceLogLikelihood: token {} at position {} is outside the vocabulary ({})",
                    token, position, vocab_size ) );
        }
    }

    /**
     * @brief Teacher-forced log-likelihood of a token sequence under a model.
     *
     * The two fields are reported separately rather than pre-averaged because the caller
     * usually sums several sequences before dividing: perplexity over a corpus is
     * exp( -total_log_probability / total_scored_positions ), and averaging per sequence
     * first would weight a short sequence like a long one.
     *
     * Accumulated in double. Per-position log-probabilities are small negative numbers and a
     * corpus contributes tens of thousands of them, so the running total is where precision
     * is actually at risk -- not in any single term.
     */
    export struct SequenceLogLikelihood
    {
        /// Summed natural log of the probability the model assigned to each actual next token.
        double total_log_probability{ 0.0 };

        /// Positions that contributed. One less than the sequence length: the first token has
        /// no preceding context, so nothing predicts it.
        dim_t scored_positions{ 0 };

        /**
         * @brief Add the log-probability each logit row gave to the token that actually followed it.
         *
         * Row r holds the logits at absolute position first_position + r, so its target is the token at
         * first_position + r + 1. Rows at or past the sequence's last position predict nothing and are
         * skipped. Each row is reduced by nextTokenLogProbability.
         *
         * @param logits              [rows, vocab_size], host, FP32.
         * @param rows                Logit rows in this window.
         * @param vocab_size          Width of each row.
         * @param tokens              The whole sequence, host.
         * @param first_position      Absolute position of row 0.
         * @param sequence_length     Tokens in the whole sequence.
         * @param final_logit_softcap Applied to every logit as cap * tanh( logit / cap ) before the
         *                            log-softmax, as the model applies it before sampling. 0 means none.
         *
         * @throws std::out_of_range when a target token is outside the vocabulary.
         */
        void addNextTokenLogProbabilities(
            const float* logits, dim_t rows, dim_t vocab_size,
            const std::int32_t* tokens, dim_t first_position, dim_t sequence_length,
            float final_logit_softcap )
        {
            for ( dim_t row = 0; row < rows; ++row )
            {
                const dim_t position = first_position + row;

                if ( position + 1 >= sequence_length )
                    break;

                const std::int32_t target = tokens[ position + 1 ];

                if ( target < 0 || target >= vocab_size )
                    throw std::out_of_range( std::format(
                        "sequenceLogLikelihood: token {} at position {} is outside the vocabulary ({})",
                        target, position + 1, vocab_size ) );

                total_log_probability +=
                    nextTokenLogProbability( logits + row * vocab_size, vocab_size, target, final_logit_softcap );

                ++scored_positions;
            }
        }
    };
}
