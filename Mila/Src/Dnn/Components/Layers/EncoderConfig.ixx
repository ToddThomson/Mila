/**
 * @file EncoderConfig.ixx
 * @brief Configuration interface for the Encoder module in the Mila DNN framework.
 *
 * Defines the EncoderConfig class, providing a type-safe fluent interface for configuring
 * Encoder modules. Inherits from ModuleConfig CRTP base and adds Encoder-specific options
 * such as embedding dimension, number of heads, and feed-forward layer size.
 *
 * Exposed as part of the Encoder module via module partitions.
 */

module;
#include <stdexcept>
#include <string>
#include <utility>
#include <sstream>

export module Dnn.Components.Encoder:Config;

import Dnn.Component;
import Dnn.ComponentConfig;
import Dnn.ActivationType;
import nlohmann.json;

namespace Mila::Dnn
{
    using json = nlohmann::json;

    /**
     * @brief Configuration class for Encoder module.
     *
     * Provides a type-safe fluent interface for configuring Encoder modules.
     */
    export class EncoderConfig : public ComponentConfig {
    public:
        /**
         * @brief Default constructor.
         */
        EncoderConfig() = default;

        /**
         * @brief C++23-style fluent setter for embedding dimension.
         */
        template <typename Self>
        decltype(auto) withChannels( this Self&& self, size_t channels )
        {
            self.channels_ = channels;
            return std::forward<Self>( self );
        }

        /**
         * @brief C++23-style fluent setter for maximum sequence length.
         */
        template <typename Self>
        decltype(auto) withMaxSequenceLength( this Self&& self, size_t max_seq_len )
        {
            self.max_seq_len_ = max_seq_len;
            return std::forward<Self>( self );
        }

        /**
         * @brief C++23-style fluent setter for vocabulary length.
         */
        template <typename Self>
        decltype(auto) withVocabularyLength( this Self&& self, size_t vocab_len )
        {
            self.vocab_len_ = vocab_len;
            return std::forward<Self>( self );
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
        void validate() const override {

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

        /**
         * @brief Serialize this configuration to JSON.
         *
         * Keys:
         * - "name" : string
         * - "precision" : integer (underlying value of ComputePrecision::Policy)
         * - "channels" : integer
         * - "max_sequence_length" : integer
         * - "vocabulary_length" : integer
         */
        json toJson() const
        {
            json j;
            //j["name"] = name_;
            ////j["precision"] = static_cast<int>( precision_ );
            //j["channels"] = channels_;
            //j["max_sequence_length"] = max_seq_len_;
            //j["vocabulary_length"] = vocab_len_;

            return j;
        }

        /**
         * @brief Deserialize configuration from JSON.
         *
         * Missing keys leave fields at their current values. Unknown or
         * ill-typed values will throw via nlohmann::json exceptions.
         */
        void fromJson( const json& j )
        {
            //if ( j.contains( "name" ) ) {
            //    name_ = j.at( "name" ).get<std::string>();
            //}

            //if ( j.contains( "precision" ) ) {
            //    //precision_ = static_cast<decltype(precision_)>( j.at( "precision" ).get<int>() );
            //}

            //if ( j.contains( "channels" ) ) {
            //    channels_ = j.at( "channels" ).get<size_t>();
            //}

            //if ( j.contains( "max_sequence_length" ) ) {
            //    max_seq_len_ = j.at( "max_sequence_length" ).get<size_t>();
            //}

            //if ( j.contains( "vocabulary_length" ) ) {
            //    vocab_len_ = j.at( "vocabulary_length" ).get<size_t>();
            //}
        }

        /**
         * @brief Produce a short, human-readable summary of this configuration.
         *
         * Overrides ComponentConfig::toString() to include Encoder-specific fields.
		 */
        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "channels=" << channels_
                << ", max_sequence_length=" << max_seq_len_
                << ", vocabulary_length=" << vocab_len_;
            return oss.str();
		}

    private:
        size_t channels_ = 0;         ///< The embedding dimension size
        size_t max_seq_len_ = 512;      ///< The maximum sequence length
        size_t vocab_len_ = 50000;      ///< The vocabulary size
    };
}