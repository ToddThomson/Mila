/**
 * @file Rope.Config.ixx
 * @brief Configuration for Rotary Position Embedding (RoPE) component.
 *
 * Provides construction, validation and serialization for RoPE configuration.
 *
 * Design principle (Mila-wide):
 *   - Constructor parameters are structurally required — no sensible default exists.
 *   - Fluent setters are reserved for optional behavioural parameters that have
 *     well-known defaults. There are no fluent overrides for constructor parameters.
 *
 * Required (constructor): channels, n_heads, n_kv_heads, max_seq_len.
 * Optional (fluent):      base (default 10000.0f), rotary_dim (default 0 = full head_dim).
 *
 * Typical usage:
 * @code
 * auto cfg = RopeConfig( model_dim, n_heads, n_kv_heads, max_seq_len )
 *     .withBase( 500000.0f );  // Llama 3 theta
 * @endcode
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

    export class RopeConfig : public ComponentConfig
    {
    public:

        /**
         * @brief Construct with all structurally required parameters.
         *
         * @param channels     Total Q embedding width (n_heads * head_dim).
         * @param n_heads      Number of query heads.
         * @param n_kv_heads   Number of key/value heads (GQA: <= n_heads).
         * @param max_seq_len  Maximum sequence length for cos/sin cache precomputation.
         */
        RopeConfig( size_t channels, size_t n_heads, size_t n_kv_heads, size_t max_seq_len )
            : channels_( channels ), n_heads_( n_heads ), n_kv_heads_( n_kv_heads ), max_seq_len_( max_seq_len )
        {}

        // ====================================================================
        // Optional fluent setters — behavioural parameters with sensible defaults.
        // No fluent overrides exist for constructor parameters.
        // ====================================================================

        /**
         * @brief Set frequency base for rotary angle computation.
         *
         * Standard RoPE default is 10000.0f.
         * Llama 3 uses 500000.0f.
         * Default: 10000.0f.
         */
        template <typename Self>
        decltype(auto) withBase( this Self&& self, float base )
        {
            self.base_ = base;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set rotary sub-dimension per head (number of channels to rotate).
         *
         * Default: 0 — the full head_dim is rotated.
         */
        template <typename Self>
        decltype(auto) withRotaryDim( this Self&& self, size_t rotary_dim )
        {
            self.rotary_dim_ = rotary_dim;
            return std::forward<Self>( self );
        }

        // ====================================================================
        // Accessors
        // ====================================================================

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
         * @brief Per-head dimension, derived as channels / n_heads.
         *
         * Valid only after validate() has confirmed consistency.
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

        // ====================================================================
        // Validation
        // ====================================================================

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
            {
                throw std::invalid_argument( "RopeConfig: channels must be > 0" );
            }

            if ( n_heads_ == 0 )
            {
                throw std::invalid_argument( "RopeConfig: n_heads must be > 0" );
            }

            if ( n_kv_heads_ == 0 )
            {
                throw std::invalid_argument( "RopeConfig: n_kv_heads must be > 0" );
            }

            if ( channels_ % n_heads_ != 0 )
            {
                throw std::invalid_argument( "RopeConfig: channels must be divisible by n_heads" );
            }

            const size_t head_dim = channels_ / n_heads_;

            if ( head_dim % 2 != 0 )
            {
                throw std::invalid_argument( "RopeConfig: head_dim (channels / n_heads) must be even" );
            }

            if ( n_kv_heads_ > n_heads_ )
            {
                throw std::invalid_argument( "RopeConfig: n_kv_heads must be <= n_heads" );
            }

            if ( max_seq_len_ == 0 )
            {
                throw std::invalid_argument( "RopeConfig: max_sequence_length must be > 0" );
            }

            if ( base_ <= 0.0f )
            {
                throw std::invalid_argument( "RopeConfig: base must be > 0" );
            }

            if ( rotary_dim_ != 0 && rotary_dim_ > head_dim )
            {
                throw std::invalid_argument( "RopeConfig: rotary_dim must be <= head_dim" );
            }
        }

        // ====================================================================
        // Serialization
        // ====================================================================

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
            {
                meta.set( "rotary_dim", static_cast<int64_t>(rotary_dim_) );
            }

            return meta;
        }

        void fromMetadata( const SerializationMetadata& meta ) override
        {
            if ( auto v = meta.tryGetInt( "precision" ) )
            {
                precision_ = static_cast<decltype(precision_)>(*v);
            }

            if ( auto v = meta.tryGetInt( "channels" ) )
            {
                channels_ = static_cast<size_t>(*v);
            }

            if ( auto v = meta.tryGetInt( "n_heads" ) )
            {
                n_heads_ = static_cast<size_t>(*v);
            }

            if ( auto v = meta.tryGetInt( "n_kv_heads" ) )
            {
                n_kv_heads_ = static_cast<size_t>(*v);
            }

            if ( auto v = meta.tryGetInt( "max_sequence_length" ) )
            {
                max_seq_len_ = static_cast<size_t>(*v);
            }

            if ( auto v = meta.tryGetFloat( "base" ) )
            {
                base_ = *v;
            }

            if ( auto v = meta.tryGetInt( "rotary_dim" ) )
            {
                rotary_dim_ = static_cast<size_t>(*v);
            }
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "RopeConfig{ "
                << "channels=" << channels_
                << ", n_heads=" << n_heads_
                << ", n_kv_heads=" << n_kv_heads_
                << ", head_dim=" << getHeadDim()
                << ", max_sequence_length=" << max_seq_len_
                << ", rotary_dim=" << rotary_dim_
                << ", base=" << base_
                << " }";
            return oss.str();
        }

    private:
        size_t channels_{ 0 };
        size_t n_heads_{ 0 };
        size_t n_kv_heads_{ 0 };
        size_t max_seq_len_{ 0 };
        size_t rotary_dim_{ 0 };       ///< 0 = use full head_dim
        float  base_{ 10000.0f };
    };
}
