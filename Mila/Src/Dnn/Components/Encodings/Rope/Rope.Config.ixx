/**
 * @file Rope.Config.ixx
 * @brief Configuration for Rotary Position Embedding (RoPE) component.
 *
 * Provides construction, validation and serialization for RoPE configuration.
 *
 * Design principle (Mila-wide):
 *   - Constructor parameters are structurally required -- no sensible default exists.
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

export module Dnn.Components.RopeConfig;

export import Dnn.Components.RotaryLayout;
export import Dnn.Components.RopeFrequencyScaling;

import Dnn.Component;
import Dnn.ComponentConfig;
import Dnn.TensorTypes;
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
         * @param max_seq_len  Trained maximum sequence length. A build longer than this is
         *                     refused; the cos/sin tables are sized to the build's own length.
         */
        RopeConfig( dim_t channels, dim_t n_heads, dim_t n_kv_heads, dim_t max_seq_len )
            : channels_( channels ), n_heads_( n_heads ), n_kv_heads_( n_kv_heads ), max_seq_len_( max_seq_len )
        {}

        // ====================================================================
        // Optional fluent setters -- behavioural parameters with sensible defaults.
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
         * Default: 0 -- the full head_dim is rotated.
         */
        template <typename Self>
        decltype(auto) withRotaryDim( this Self&& self, dim_t rotary_dim )
        {
            self.rotary_dim_ = rotary_dim;
            return std::forward<Self>( self );
        }

        /**
         * @brief Select which dimension pairs a partial rotary rotates.
         *
         * Default: WholeHead, which is what Gemma's reference does and what every family in
         * the tree used before Qwen. Only Qwen needs RotaryPrefix.
         */
        template <typename Self>
        decltype(auto) withRotaryLayout( this Self&& self, RotaryLayout rotary_layout )
        {
            self.rotary_layout_ = rotary_layout;
            return std::forward<Self>( self );
        }

        /**
         * @brief Rescale the rotary frequencies as the checkpoint was trained to read them.
         *
         * Default: no scaling. Llama 3.1 and 3.2 need it at every position, not only past their
         * original context.
         */
        template <typename Self>
        decltype(auto) withFrequencyScaling( this Self&& self, const RopeFrequencyScaling& frequency_scaling )
        {
            self.frequency_scaling_ = frequency_scaling;
            return std::forward<Self>( self );
        }

        // ====================================================================
        // Accessors
        // ====================================================================

        dim_t getEmbeddingDim() const noexcept
        {
            return channels_;
        }

        dim_t getNumHeads() const noexcept
        {
            return n_heads_;
        }

        dim_t getNumKVHeads() const noexcept
        {
            return n_kv_heads_;
        }

        /**
         * @brief Per-head dimension, derived as channels / n_heads.
         *
         * Valid only after validate() has confirmed consistency.
         */
        dim_t getHeadDim() const noexcept
        {
            return (n_heads_ > 0) ? (channels_ / n_heads_) : 0;
        }

        /**
         * @brief Returns the training maximum sequence length.
         * @return The maximum sequence length.
         */
        dim_t getMaxSequenceLength() const noexcept
        {
            // REVIEW: See LlamaConfig::withMaxSequenceLength() for discussion on naming and semantics of this parameter.
            return max_seq_len_;
        }

        dim_t getRotaryDim() const noexcept
        {
            return rotary_dim_;
        }

        RotaryLayout getRotaryLayout() const noexcept
        {
            return rotary_layout_;
        }

        float getBase() const noexcept
        {
            return base_;
        }

        const RopeFrequencyScaling& getFrequencyScaling() const noexcept
        {
            return frequency_scaling_;
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
            if ( channels_ <= 0 )
            {
                throw std::invalid_argument( "RopeConfig: channels must be > 0" );
            }

            if ( n_heads_ <= 0 )
            {
                throw std::invalid_argument( "RopeConfig: n_heads must be > 0" );
            }

            if ( n_kv_heads_ <= 0 )
            {
                throw std::invalid_argument( "RopeConfig: n_kv_heads must be > 0" );
            }

            if ( channels_ % n_heads_ != 0 )
            {
                throw std::invalid_argument( "RopeConfig: channels must be divisible by n_heads" );
            }

            const dim_t head_dim = channels_ / n_heads_;

            if ( head_dim % 2 != 0 )
            {
                throw std::invalid_argument( "RopeConfig: head_dim (channels / n_heads) must be even" );
            }

            if ( n_kv_heads_ > n_heads_ )
            {
                throw std::invalid_argument( "RopeConfig: n_kv_heads must be <= n_heads" );
            }

            if ( max_seq_len_ <= 0 )
            {
                throw std::invalid_argument( "RopeConfig: max_sequence_length must be > 0" );
            }

            if ( base_ <= 0.0f )
            {
                throw std::invalid_argument( "RopeConfig: base must be > 0" );
            }

            if ( rotary_dim_ < 0 )
            {
                throw std::invalid_argument( "RopeConfig: rotary_dim must be >= 0 (0 = full head_dim)" );
            }

            if ( rotary_dim_ != 0 && rotary_dim_ > head_dim )
            {
                throw std::invalid_argument( "RopeConfig: rotary_dim must be <= head_dim" );
            }

            frequency_scaling_.validate();
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        SerializationMetadata toMetadata() const override
        {
            SerializationMetadata meta;

            meta.set( "channels", static_cast<int64_t>(channels_) )
                .set( "n_heads", static_cast<int64_t>(n_heads_) )
                .set( "n_kv_heads", static_cast<int64_t>(n_kv_heads_) )
                .set( "max_sequence_length", static_cast<int64_t>(max_seq_len_) )
                .set( "base", base_ );

            if ( rotary_dim_ != 0 )
            {
                meta.set( "rotary_dim", static_cast<int64_t>(rotary_dim_) );
            }

            if ( frequency_scaling_.isScaled() )
            {
                meta.set( "scaling_factor", frequency_scaling_.factor )
                    .set( "scaling_low_frequency_factor", frequency_scaling_.low_frequency_factor )
                    .set( "scaling_high_frequency_factor", frequency_scaling_.high_frequency_factor )
                    .set( "scaling_original_context_length", static_cast<int64_t>(frequency_scaling_.original_context_length) );
            }

            return meta;
        }

        void fromMetadata( const SerializationMetadata& meta ) override
        {
            if ( auto v = meta.tryGetInt( "channels" ) )
            {
                channels_ = static_cast<dim_t>(*v);
            }

            if ( auto v = meta.tryGetInt( "n_heads" ) )
            {
                n_heads_ = static_cast<dim_t>(*v);
            }

            if ( auto v = meta.tryGetInt( "n_kv_heads" ) )
            {
                n_kv_heads_ = static_cast<dim_t>(*v);
            }

            if ( auto v = meta.tryGetInt( "max_sequence_length" ) )
            {
                max_seq_len_ = static_cast<dim_t>(*v);
            }

            if ( auto v = meta.tryGetFloat( "base" ) )
            {
                base_ = *v;
            }

            if ( auto v = meta.tryGetInt( "rotary_dim" ) )
            {
                rotary_dim_ = static_cast<dim_t>(*v);
            }

            if ( auto v = meta.tryGetInt( "scaling_original_context_length" ) )
            {
                frequency_scaling_.original_context_length = static_cast<dim_t>(*v);
                frequency_scaling_.factor =
                    static_cast<float>( meta.tryGetFloat( "scaling_factor" ).value_or( 1.0 ) );
                frequency_scaling_.low_frequency_factor =
                    static_cast<float>( meta.tryGetFloat( "scaling_low_frequency_factor" ).value_or( 1.0 ) );
                frequency_scaling_.high_frequency_factor =
                    static_cast<float>( meta.tryGetFloat( "scaling_high_frequency_factor" ).value_or( 1.0 ) );
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
                << ", base=" << base_;

            if ( frequency_scaling_.isScaled() )
            {
                oss << ", scaling={ factor=" << frequency_scaling_.factor
                    << ", low_frequency_factor=" << frequency_scaling_.low_frequency_factor
                    << ", high_frequency_factor=" << frequency_scaling_.high_frequency_factor
                    << ", original_context_length=" << frequency_scaling_.original_context_length << " }";
            }

            oss << " }";
            return oss.str();
        }

    private:

        dim_t channels_{ 0 };
        dim_t n_heads_{ 0 };
        dim_t n_kv_heads_{ 0 };
        dim_t max_seq_len_{ 0 };
        dim_t rotary_dim_{ 0 };        ///< 0 = use full head_dim
        RotaryLayout rotary_layout_{ RotaryLayout::WholeHead };
        float base_{ 10000.0f };
        RopeFrequencyScaling frequency_scaling_{};
    };
}
