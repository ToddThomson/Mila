/**
 * @file EncoderConfig.ixx
 * @brief Configuration interface for the Encoder module in the Mila DNN framework.
 *
 * Defines the EncoderConfig class, providing a type-safe fluent interface for configuring
 * Encoder modules. Inherits from ComponentConfig CRTP base and adds Encoder-specific options
 * such as embedding dimension, number of heads, and feed-forward layer size.
 *
 * Exposed as part of the Encoder module via module partitions.
 */

module;
#include <stdexcept>

export module Dnn.Modules.Encoder:Config;

import Dnn.Module;
import Dnn.ComponentConfig;
import Dnn.ActivationType;

namespace Mila::Dnn
{
    /**
     * @brief Configuration class for Encoder module.
     *
     * Provides a type-safe fluent interface for configuring Encoder modules.
     */
    export class EncoderConfig : public ComponentConfig<EncoderConfig> {
    public:
        /**
         * @brief Default constructor.
         */
        EncoderConfig() = default;

        /**
         * @brief Set the embedding dimension.
         *
         * @param channels The embedding dimension size
         * @return EncoderConfig& Reference to this for method chaining
         */
        EncoderConfig& withChannels( size_t channels ) {
            channels_ = channels;
            return *this;
        }

        /**
         * @brief Set the maximum sequence length.
         *
         * @param max_seq_len Maximum sequence length
         * @return EncoderConfig& Reference to this for method chaining
         */
        EncoderConfig& withMaxSequenceLength( size_t max_seq_len ) {
            max_seq_len_ = max_seq_len;
            return *this;
        }

        /**
         * @brief Set the vocabulary length.
         *
         * @param vocab_len Size of the vocabulary
         * @return EncoderConfig& Reference to this for method chaining
         */
        EncoderConfig& withVocabularyLength( size_t vocab_len ) {
            vocab_len_ = vocab_len;
            return *this;
        }

        /**
         * @brief Get the configured embedding dimension.
         *
         * @return size_t The embedding dimension
         */
        size_t getChannels() const { return channels_; }

        /**
         * @brief Get the configured maximum sequence length.
         *
         * @return size_t The maximum sequence length
         */
        size_t getMaxSequenceLength() const { return max_seq_len_; }

        /**
         * @brief Get the configured vocabulary length.
         *
         * @return size_t The vocabulary length
         */
        size_t getVocabularyLength() const { return vocab_len_; }

        /**
         * @brief Validate configuration parameters.
         *
         * @throws std::invalid_argument If validation fails
         */
        void validate() const {
            ComponentConfig<EncoderConfig>::validate();

            if ( channels_ == 0 ) {
                throw std::invalid_argument( "Embedding dimension (channels) must be greater than zero" );
            }

            if ( max_seq_len_ == 0 ) {
                throw std::invalid_argument( "Maximum sequence length must be greater than zero" );
            }

            if ( vocab_len_ == 0 ) {
                throw std::invalid_argument( "Vocabulary length must be greater than zero" );
            }
        }

    private:
        size_t channels_ = 512;         ///< The embedding dimension size
        size_t max_seq_len_ = 512;      ///< The maximum sequence length
        size_t vocab_len_ = 50000;      ///< The vocabulary size
    };
}