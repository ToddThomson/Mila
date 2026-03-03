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
#include <string>
#include <utility>
#include <sstream>

export module Dnn.Components.Lpe:Config;

import Dnn.Component;
import Dnn.ComponentConfig;
import Dnn.ActivationType;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    /**
     * @brief Configuration class for the Learned Positional Encoder.
     *
     * Provides a type-safe fluent interface for configuring Encoder.
     */
    export class LpeConfig : public ComponentConfig {
    public:
        
        /**
         * @brief C++23-style fluent setter for embedding dimension.
         */
        template <typename Self>
        decltype(auto) withEmbeddingDim( this Self&& self, size_t embedding_dim )
        {
            self.embedding_dim_ = embedding_dim;
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
        size_t getEmbeddingDim() const { return embedding_dim_; }

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

            if ( embedding_dim_ == 0 ) {
                throw std::invalid_argument( "EncoderConfig: channels must be > 0" );
            }

            if ( max_seq_len_ == 0 ) {
                throw std::invalid_argument( "EncoderConfig: max_sequence_length must be > 0" );
            }

            if ( vocab_len_ == 0 ) {
                throw std::invalid_argument( "EncoderConfig: vocabulary_length must be > 0" );
            }
        }

        /**
         * @brief Convert configuration to serialization metadata.
         *
         * Produces a SerializationMetadata object containing the configuration
         * fields suitable for writing into an archive by the caller.
         */
        SerializationMetadata toMetadata() const
        {
            SerializationMetadata meta;
            meta.set( "precision", static_cast<int64_t>( precision_ ) )
                .set( "channels", static_cast<int64_t>( embedding_dim_ ) )
                .set( "max_sequence_length", static_cast<int64_t>( max_seq_len_ ) )
                .set( "vocabulary_length", static_cast<int64_t>( vocab_len_ ) );

            return meta;
        }

        /**
         * @brief Populate configuration from serialization metadata.
         *
         * Reads available fields from the provided metadata and updates the
         * configuration object accordingly.
         */
        void fromMetadata( const SerializationMetadata& meta )
        {
            if ( auto prec = meta.tryGetInt( "precision" ) )
            {
                precision_ = static_cast<decltype( precision_ )>( *prec );
            }

            if ( auto ch = meta.tryGetInt( "channels" ) )
            {
                embedding_dim_ = static_cast<size_t>( *ch );
            }

            if ( auto ms = meta.tryGetInt( "max_sequence_length" ) )
            {
                max_seq_len_ = static_cast<size_t>( *ms );
            }

            if ( auto vl = meta.tryGetInt( "vocabulary_length" ) )
            {
                vocab_len_ = static_cast<size_t>( *vl );
            }
        }

        /**
         * @brief Produce a short, human-readable summary of this configuration.
         *
         * Overrides ComponentConfig::toString() to include Encoder-specific fields.
         */
        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "channels=" << embedding_dim_
                << ", max_sequence_length=" << max_seq_len_
                << ", vocabulary_length=" << vocab_len_;

            return oss.str();
        }

    private:
        size_t embedding_dim_ = 0;      ///< The embedding dimension size
        size_t max_seq_len_ = 512;      ///< The maximum sequence length
        size_t vocab_len_ = 50000;      ///< The vocabulary size
    };
}