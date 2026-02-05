/**
 * @file Gpt2.Config.ixx
 * @brief Configuration for the GPT-2 sample network.
 */

module;
#include <sstream>
#include <string>
#include <stdexcept>

export module Gpt2.Config;

import Dnn.ComponentConfig;
import Serialization.Metadata;

namespace Mila::Gpt2
{
    using Mila::Dnn::ComponentConfig;
    using Mila::Dnn::Serialization::SerializationMetadata;

    /**
     * @brief Construction-time configuration for GPT-2 sample.
     *
     * Provides fluent setters and validation. Call `validate()` before using
     * the configuration to construct a network.
     */
    export class Gpt2Config : public ComponentConfig
    {
    public:
        Gpt2Config() = default;

        // Fluent setters
        Gpt2Config& withMaxSequenceLength( int v ) noexcept { max_seq_len = v; return *this; }
        Gpt2Config& withVocabularySize( int v ) noexcept { vocab_size = v; return *this; }
        Gpt2Config& withPaddedVocabularySize( int v ) noexcept { padded_vocab_size = v; return *this; }
        Gpt2Config& withNumLayers( int v ) noexcept { num_layers = v; return *this; }
        Gpt2Config& withNumHeads( int v ) noexcept { num_heads = v; return *this; }
        Gpt2Config& withChannels( int v ) noexcept { channels = v; return *this; }

        /**
         * @brief Validate configuration parameters.
         *
         * @throws std::invalid_argument if any constraint is violated.
         */
        void validate() const override
        {
            if ( vocab_size <= 0 )
            {
                throw std::invalid_argument( "Gpt2Config: vocab_size must be positive" );
            }

            if ( padded_vocab_size <= 0 )
            {
                throw std::invalid_argument( "Gpt2Config: padded_vocab_size must be positive" );
            }

            if ( max_seq_len <= 0 )
            {
                throw std::invalid_argument( "Gpt2Config: max_seq_len must be positive" );
            }

            if ( num_layers <= 0 )
            {
                throw std::invalid_argument( "Gpt2Config: num_layers must be positive" );
            }

            if ( num_heads <= 0 )
            {
                throw std::invalid_argument( "Gpt2Config: num_heads must be positive" );
            }

            if ( channels <= 0 )
            {
                throw std::invalid_argument( "Gpt2Config: channels must be positive" );
            }

            if ( channels % num_heads != 0 )
            {
                throw std::invalid_argument( "Gpt2Config: channels must be divisible by num_heads" );
            }
        }

        /**
         * @brief Convert configuration to SerializationMetadata.
         */
        SerializationMetadata toMetadata() const override
        {
            SerializationMetadata meta;
            meta.set( "vocab_size", int64_t( vocab_size ) )
                .set( "padded_vocab_size", int64_t( padded_vocab_size ) )
                .set( "max_seq_len", int64_t( max_seq_len ) )
                .set( "num_layers", int64_t( num_layers ) )
                .set( "num_heads", int64_t( num_heads ) )
                .set( "channels", int64_t( channels ) );

            return meta;
        }

        /**
         * @brief Populate configuration from provided metadata.
         *
         * Missing keys are ignored so defaults remain intact.
         */
        void fromMetadata( const SerializationMetadata& meta ) override
        {
            if ( auto v = meta.tryGetInt( "vocab_size" ) )           vocab_size = static_cast<int>( *v );
            if ( auto v = meta.tryGetInt( "padded_vocab_size" ) )    padded_vocab_size = static_cast<int>( *v );
            if ( auto v = meta.tryGetInt( "max_seq_len" ) )          max_seq_len = static_cast<int>( *v );
            if ( auto v = meta.tryGetInt( "num_layers" ) )           num_layers = static_cast<int>( *v );
            if ( auto v = meta.tryGetInt( "num_heads" ) )            num_heads = static_cast<int>( *v );
            if ( auto v = meta.tryGetInt( "channels" ) )             channels = static_cast<int>( *v );
        }

        /**
         * @brief Human-readable summary for logging.
         */
        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "Gpt2Config(vocab_size=" << vocab_size
                << ", padded_vocab_size=" << padded_vocab_size
                << ", max_seq_len=" << max_seq_len
                << ", channels=" << channels
                << ", num_heads=" << num_heads
                << ", num_layers=" << num_layers
                << ")";
            return oss.str();
        }

        // Public fields (value-type style)
        int max_seq_len{ 1024 };
        int vocab_size{ 50257 };
        int padded_vocab_size{ 50304 };
        int num_layers{ 12 };
        int num_heads{ 12 };
        int channels{ 768 };
    };
}
