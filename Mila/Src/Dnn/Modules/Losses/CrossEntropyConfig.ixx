/**
 * @file CrossEntropyConfig.ixx
 * @brief Configuration interface for the fused SoftmaxCrossEntropy module.
 *
 * Provides minimal configuration matching the actual CPU/CUDA kernel implementations.
 * This is the configuration for the FUSED softmax + cross-entropy loss operation.
 *
 */

module;
#include <stdexcept>
#include <cstdint>

export module Dnn.Modules.SoftmaxCrossEntropy:Config;

import Dnn.Module;
import Dnn.ConfigurationBase;

namespace Mila::Dnn
{
    /**
     * @brief Configuration for fused SoftmaxCrossEntropy loss.
     */
    export class CrossEntropyConfig : public ConfigurationBase
    {
    public:
        /**
         * @brief Constructor with required vocabulary size parameter.
         *
         * @param vocab_size The size of the vocabulary (number of classes).
         *                   Must be > 0. Kernels validate: 0 <= target < vocab_size.
         *
         * Example:
         *   CrossEntropyConfig(50257)  // GPT-2 vocab size
         */
        explicit CrossEntropyConfig( int64_t vocab_size )
            : vocab_size_( vocab_size )
        {
        }

        /**
         * @brief Get the vocabulary size.
         *
         * Used by kernels to validate target indices.
         * Targets outside [0, vocab_size) are automatically ignored (loss/grad = 0).
         *
         * @return int64_t The vocabulary size
         */
        int64_t getVocabSize() const
        {
            return vocab_size_;
        }

        /**
         * @brief Validate configuration parameters.
         *
         * Checks that vocabulary size is positive.
         *
         * @throws std::invalid_argument If vocab_size <= 0
         */
        void validate() const override
        {
            ConfigurationBase::validate();

            if (vocab_size_ <= 0)
            {
                throw std::invalid_argument(
                    "CrossEntropyConfig: vocabulary size must be greater than zero" );
            }
        }

    private:
        int64_t vocab_size_;  ///< Number of classes in the vocabulary
    };
}