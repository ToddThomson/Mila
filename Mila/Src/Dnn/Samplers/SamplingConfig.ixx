/**
 * @file SamplingConfig.ixx
 * @brief Construction-time configuration for the token sampler.
 *
 * Holds the model-fixed sampling inputs (vocabulary size, final logit softcap).
 * Per-call knobs (temperature / top_k / top_p / seed) are GenerateParams, not here.
 */

module;
#include <cstdint>
#include <stdexcept>
#include <utility>

export module Dnn.Samplers.SamplingConfig;

import Dnn.TensorTypes;

namespace Mila::Dnn
{
    /**
     * @brief Model-fixed configuration for TokenSampler.
     *
     * `vocab_size` is the logits width; `final_logit_softcap` is the Gemma-style
     * c*tanh(logits/c) cap applied before temperature (0 disables it). Per-request
     * sampling parameters live in GenerateParams (see TokenSampling.md section 4.1).
     */
    export class SamplingConfig
    {
    public:
        template<typename Self>
        decltype(auto) withVocabularySize( this Self&& self, int64_t vocab_size ) noexcept
        {
            self.vocab_size_ = vocab_size;
            return std::forward<Self>( self );
        }

        template<typename Self>
        decltype(auto) withFinalLogitSoftcap( this Self&& self, float softcap ) noexcept
        {
            self.final_logit_softcap_ = softcap;
            return std::forward<Self>( self );
        }

        /**
         * @brief The most logits rows one call samples, which sizes the sampler's working stores.
         *
         * 1 for the ordinary sampler; a verify's K + 1 for the speculative sampler (Gemma4Mtp.md 4.8).
         */
        template<typename Self>
        decltype(auto) withMaximumRows( this Self&& self, dim_t rows ) noexcept
        {
            self.maximum_rows_ = rows;
            return std::forward<Self>( self );
        }

        int64_t getVocabularySize() const noexcept
        {
            return vocab_size_;
        }

        float getFinalLogitSoftcap() const noexcept
        {
            return final_logit_softcap_;
        }

        dim_t getMaximumRows() const noexcept
        {
            return maximum_rows_;
        }

        /**
         * @brief Validate configuration parameters.
         *
         * Throws std::invalid_argument on invalid configuration.
         */
        void validate() const
        {
            if ( vocab_size_ <= 0 )
            {
                throw std::invalid_argument( "SamplingConfig: vocabulary size must be positive" );
            }

            if ( final_logit_softcap_ < 0.0f )
            {
                throw std::invalid_argument( "SamplingConfig: final logit softcap must be non-negative" );
            }

            if ( maximum_rows_ < 1 )
            {
                throw std::invalid_argument( "SamplingConfig: maximum rows must be at least 1" );
            }
        }

    private:
        int64_t vocab_size_{ 0 };
        float final_logit_softcap_{ 0.0f };
        dim_t maximum_rows_{ 1 };
    };
}
