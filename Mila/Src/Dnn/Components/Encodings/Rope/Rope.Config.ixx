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
     * Defines the shape and frequency parameters required to build the cos/sin
     * cache and validate Q/K tensors at runtime.
     *
     * Required fields: channels, n_heads, n_kv_heads, max_sequence_length.
     * head_dim is derived as channels / n_heads and must be even.
     *
     * Typical usage:
     * @code
     * auto cfg = RopeConfig{}
     *     .withChannels( 512 )
     *     .withNumHeads( 8 )
     *     .withNumKvHeads( 2 )
     *     .withMaxSequenceLength( 2048 );
     * @endcode
     */
    export class RopeConfig : public ComponentConfig
    {
    public:
        RopeConfig() = default;

        /**
         * @brief Set total Q embedding channel dimension (C = n_heads * head_dim).
         */
        template <typename Self>
        decltype(auto) withChannels( this Self&& self, size_t channels )
        {
            self.channels_ = channels;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set number of query heads.
         */
        template <typename Self>
        decltype(auto) withNumHeads( this Self&& self, size_t n_heads )
        {
            self.n_heads_ = n_heads;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set number of key/value heads (GQA: must be <= n_heads).
         */
        template <typename Self>
        decltype(auto) withNumKvHeads( this Self&& self, size_t n_kv_heads )
        {
            self.n_kv_heads_ = n_kv_heads;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set maximum sequence length for cos/sin cache precomputation.
         */
        template <typename Self>
        decltype(auto) withMaxSequenceLength( this Self&& self, size_t max_seq_len )
        {
            self.max_seq_len_ = max_seq_len;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set rotary sub-dimension per head (number of channels to rotate).
         *
         * If not set (0) the full head_dim is used.
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

        size_t getEmbeddingDim() const noexcept
        {
            return channels_;
        }

        size_t getNumHeads() const noexcept
        {
            return n_heads_;
        }

        size_t getNumKvHeads() const noexcept
        {
            return n_kv_heads_;
        }

        /**
         * @brief Per-head embedding dimension, derived as channels / n_heads.
         *
         * Valid only after validate() has confirmed channels and n_heads are consistent.
         */
        size_t getHeadDim() const noexcept
        {
            return (n_heads_ > 0) ? (channels_ / n_heads_) : 0;
        }

        size_t getMaxSequenceLength() const noexcept
        {
            return max_seq_len_;
        }

        size_t getRotaryDim() const noexcept
        {
            return rotary_dim_;
        }

        float getBase() const noexcept
        {
            return base_;
        }

        /**
         * @brief Validate configuration.
         *
         * Enforces: required fields are positive, channels is divisible by n_heads,
         * head_dim is even (RoPE requires paired dimensions), n_kv_heads <= n_heads,
         * and rotary_dim (if set) does not exceed head_dim.
         *
         * @throws std::invalid_argument on any violated constraint.
         */
        void validate() const override
        {
            if ( channels_ == 0 )
                throw std::invalid_argument( "RopeConfig: channels must be > 0" );

            if ( n_heads_ == 0 )
                throw std::invalid_argument( "RopeConfig: n_heads must be > 0" );

            if ( n_kv_heads_ == 0 )
                throw std::invalid_argument( "RopeConfig: n_kv_heads must be > 0" );

            if ( channels_ % n_heads_ != 0 )
                throw std::invalid_argument( "RopeConfig: channels must be divisible by n_heads" );

            const size_t head_dim = channels_ / n_heads_;

            if ( head_dim % 2 != 0 )
                throw std::invalid_argument( "RopeConfig: head_dim (channels / n_heads) must be even" );

            if ( n_kv_heads_ > n_heads_ )
                throw std::invalid_argument( "RopeConfig: n_kv_heads must be <= n_heads" );

            if ( max_seq_len_ == 0 )
                throw std::invalid_argument( "RopeConfig: max_sequence_length must be > 0" );

            if ( base_ <= 0.0f )
                throw std::invalid_argument( "RopeConfig: base must be > 0" );

            if ( rotary_dim_ != 0 && rotary_dim_ > head_dim )
                throw std::invalid_argument( "RopeConfig: rotary_dim must be <= head_dim" );
        }

        /**
         * @brief Serialize configuration to metadata.
         */
        SerializationMetadata toMetadata() const override
        {
            SerializationMetadata meta;
            meta.set( "precision", static_cast<int64_t>(precision_) )
                .set( "channels", static_cast<int64_t>(channels_) )
                .set( "n_heads", static_cast<int64_t>(n_heads_) )
                .set( "n_kv_heads", static_cast<int64_t>(n_kv_heads_) )
                .set( "max_sequence_length", static_cast<int64_t>(max_seq_len_) )
                .set( "base", base_ );

            if ( rotary_dim_ != 0 )
                meta.set( "rotary_dim", static_cast<int64_t>(rotary_dim_) );

            return meta;
        }

        /**
         * @brief Populate configuration from metadata.
         */
        void fromMetadata( const SerializationMetadata& meta ) override
        {
            if ( auto v = meta.tryGetInt( "precision" ) )
                precision_ = static_cast<decltype(precision_)>(*v);

            if ( auto v = meta.tryGetInt( "channels" ) )
                channels_ = static_cast<size_t>(*v);

            if ( auto v = meta.tryGetInt( "n_heads" ) )
                n_heads_ = static_cast<size_t>(*v);

            if ( auto v = meta.tryGetInt( "n_kv_heads" ) )
                n_kv_heads_ = static_cast<size_t>(*v);

            if ( auto v = meta.tryGetInt( "max_sequence_length" ) )
                max_seq_len_ = static_cast<size_t>(*v);

            if ( auto v = meta.tryGetFloat( "base" ) )
                base_ = *v;

            if ( auto v = meta.tryGetInt( "rotary_dim" ) )
                rotary_dim_ = static_cast<size_t>(*v);
        }

        /**
         * @brief Human-readable summary of the config.
         */
        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "RopeConfig{ channels=" << channels_
                << ", n_heads=" << n_heads_
                << ", n_kv_heads=" << n_kv_heads_
                << ", head_dim=" << getHeadDim()
                << ", max_sequence_length=" << max_seq_len_
                << ", rotary_dim=" << rotary_dim_
                << ", base=" << base_ << " }";
            return oss.str();
        }

    private:
        size_t channels_   = 0;
        size_t n_heads_    = 0;
        size_t n_kv_heads_ = 0;
        size_t max_seq_len_ = 2048;
        size_t rotary_dim_ = 0;      ///< 0 means use full head_dim
        float  base_       = 10000.0f;
    };
}