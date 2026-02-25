/**
 * @file Rope.Config.ixx
 * @brief Configuration for Rotary Position Embedding (RoPE) component.
 *
 * Provides fluent setters, validation and serialization for RoPE configuration.
 */

    module;
#include <stdexcept>
#include <string>
#include <sstream>
#include <utility>

export module Dnn.Components.Rope:Config;

import Dnn.Component;
import Dnn.ComponentConfig;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    /**
     * @brief Configuration for Rotary Positional Embeddings (RoPE).
     *
     * Typical usage:
     * - set embedding channels via withChannels()
     * - optionally restrict rotary embedding to a sub-dimension via withRotaryDim()
     * - set maximum sequence length (for precomputing sin/cos tables) via withMaxSequenceLength()
     * - set the base frequency via withBase()
     */
    export class RopeConfig : public ComponentConfig
    {
    public:
        RopeConfig() = default;

        /**
         * @brief Set embedding channel dimension (C).
         */
        template <typename Self>
        decltype(auto) withChannels( this Self&& self, size_t channels )
        {
            self.channels_ = channels;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set maximum sequence length for precomputation.
         */
        template <typename Self>
        decltype(auto) withMaxSequenceLength( this Self&& self, size_t max_seq_len )
        {
            self.max_seq_len_ = max_seq_len;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set rotary sub-dimension (number of channels to apply RoPE to).
         *
         * If not set (0) the full channels dimension is used.
         */
        template <typename Self>
        decltype(auto) withRotaryDim( this Self&& self, size_t rotary_dim )
        {
            self.rotary_dim_ = rotary_dim;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set frequency base used to compute the rotary angles (default 10000.0).
         */
        template <typename Self>
        decltype(auto) withBase( this Self&& self, float base )
        {
            self.base_ = base;
            return std::forward<Self>( self );
        }

        size_t getEmbeddingDim() const noexcept {
            return channels_;
        }

        size_t getMaxSequenceLength() const noexcept {
            return max_seq_len_;
        }

        size_t getRotaryDim() const noexcept {
            return rotary_dim_;
        }

        float getBase() const noexcept {
            return base_;
        }

        /**
         * @brief Validate configuration.
         *
         * Ensures required fields are positive and rotary_dim (if set) does not exceed channels.
         */
        void validate() const override
        {
            if ( channels_ == 0 )
            {
                throw std::invalid_argument( "RopeConfig: channels must be > 0" );
            }

            if ( max_seq_len_ == 0 )
            {
                throw std::invalid_argument( "RopeConfig: max_sequence_length must be > 0" );
            }

            if ( base_ <= 0.0f )
            {
                throw std::invalid_argument( "RopeConfig: base must be > 0" );
            }

            if ( rotary_dim_ != 0 && rotary_dim_ > channels_ )
            {
                throw std::invalid_argument( "RopeConfig: rotary_dim must be <= channels" );
            }
        }

        /**
         * @brief Serialize configuration to metadata.
         */
        SerializationMetadata toMetadata() const override
        {
            SerializationMetadata meta;
            meta.set( "precision", static_cast<int64_t>(precision_) )
                .set( "channels", static_cast<int64_t>(channels_) )
                .set( "max_sequence_length", static_cast<int64_t>(max_seq_len_) )
                .set( "base", base_ );

            if ( rotary_dim_ != 0 )
            {
                meta.set( "rotary_dim", static_cast<int64_t>(rotary_dim_) );
            }

            return meta;
        }

        /**
         * @brief Populate configuration from metadata.
         */
        void fromMetadata( const SerializationMetadata& meta ) override
        {
            if ( auto prec = meta.tryGetInt( "precision" ) )
            {
                precision_ = static_cast<decltype(precision_)>(*prec);
            }

            if ( auto ch = meta.tryGetInt( "channels" ) )
            {
                channels_ = static_cast<size_t>(*ch);
            }

            if ( auto ms = meta.tryGetInt( "max_sequence_length" ) )
            {
                max_seq_len_ = static_cast<size_t>(*ms);
            }

            if ( auto b = meta.tryGetFloat( "base" ) )
            {
                base_ = *b;
            }

            if ( auto rd = meta.tryGetInt( "rotary_dim" ) )
            {
                rotary_dim_ = static_cast<size_t>(*rd);
            }
        }

        /**
         * @brief Human-readable summary of the config.
         */
        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "RopeConfig{ channels=" << channels_
                << ", max_sequence_length=" << max_seq_len_
                << ", rotary_dim=" << rotary_dim_
                << ", base=" << base_ << " }";
            return oss.str();
        }

    private:
        size_t channels_ = 0;
        size_t max_seq_len_ = 2048;
        size_t rotary_dim_ = 0;   ///< 0 means use full channels_
        float base_ = 10000.0f;
    };
}