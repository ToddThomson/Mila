/**
 * @file RmsNorm.Config.ixx
 * @brief Configuration for RMS Normalization component.
 *
 * Design principle (Mila-wide):
 *   - Constructor parameters are structurally required — no sensible default exists.
 *   - Fluent setters are reserved for optional behavioural parameters that have
 *     well-known defaults. There are no fluent overrides for constructor parameters.
 *
 * RmsNormConfig supports two mutually exclusive normalization modes selected
 * by constructor overload:
 *   - Shape mode:  RmsNormConfig( shape_t )   — normalize over a trailing shape.
 *   - Axis mode:   RmsNormConfig( int64_t )   — normalize over a single axis.
 *
 * The overloads are unambiguous — shape_t and int64_t cannot collide.
 *
 * Typical usage:
 * @code
 * // Shape mode (most common for transformers)
 * auto cfg = RmsNormConfig( shape_t{ model_dim } )
 *     .withEpsilon( config_.getRMSNormEpsilon() )
 *     .withBias( false );
 *
 * // Axis mode
 * auto cfg = RmsNormConfig( int64_t{ -1 } )
 *     .withBias( false );
 * @endcode
 */

module;
#include <stdexcept>
#include <string>
#include <sstream>
#include <optional>
#include <utility>

export module Dnn.Components.RmsNormConfig;

import Dnn.Component;
import Dnn.ComponentConfig;
import Dnn.TensorTypes;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    export class RmsNormConfig : public ComponentConfig
    {
    public:

        /**
         * @brief Construct in shape mode.
         *
         * Normalizes over the trailing dimensions described by @p shape.
         *
         * @param shape  Trailing dimensions to normalize over (e.g. shape_t{ model_dim }).
         */
        explicit RmsNormConfig( shape_t normalized_shape )
            : normalized_shape_( std::move( normalized_shape ) )
        {}

        /**
         * @brief Construct in axis mode.
         *
         * Normalizes over a single axis.
         *
         * @param axis  Axis along which to normalize (negative indexing supported).
         */
        explicit RmsNormConfig( int64_t axis )
            : axis_( axis )
        {}

        // ====================================================================
        // Optional fluent setters — behavioural parameters with sensible defaults.
        // No fluent overrides exist for constructor parameters.
        // ====================================================================

        /**
         * @brief Enable or disable learnable bias.
         *
         * Default: true.
         * Llama 3 uses false.
         */
        template <typename Self>
        decltype(auto) withBias( this Self&& self, bool has_bias )
        {
            self.has_bias_ = has_bias;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set epsilon for numerical stability.
         *
         * Default: 1e-5f.
         * Llama 3 uses 1e-5f; some models use 1e-6f.
         */
        template <typename Self>
        decltype(auto) withEpsilon( this Self&& self, float epsilon )
        {
            self.epsilon_ = epsilon;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set the unit offset added to the loaded weight before scaling.
         *
         * The normalized activation is scaled by (weight + unit_offset). Default 0.0
         * reproduces standard RMSNorm (x_norm * weight) -- used by Llama 3 / GPT-2.
         * Gemma sets 1.0: its RMSNorm is x_norm * (1 + weight), with weights stored
         * raw (zero-centered, weight-decay-friendly). The offset is applied at the
         * kernel so the stored/loaded weights remain identical to the source checkpoint.
         */
        template <typename Self>
        decltype(auto) withUnitOffset( this Self&& self, float unit_offset )
        {
            self.unit_offset_ = unit_offset;
            return std::forward<Self>( self );
        }

        // ====================================================================
        // Accessors
        // ====================================================================

        const shape_t& getNormalizedShape() const noexcept
        {
            return normalized_shape_;
        }

        bool hasNormalizedShape() const noexcept
        {
            return !normalized_shape_.empty();
        }

        std::optional<int64_t> getAxis() const noexcept
        {
            return axis_;
        }

        bool hasBias() const noexcept
        {
            return has_bias_;
        }

        float getEpsilon() const noexcept
        {
            return epsilon_;
        }

        float getUnitOffset() const noexcept
        {
            return unit_offset_;
        }

        // ====================================================================
        // Validation
        // ====================================================================

        void validate() const override
        {
            if ( epsilon_ <= 0.0f )
            {
                throw std::invalid_argument( "RmsNormConfig: epsilon must be > 0" );
            }

            const bool has_shape = !normalized_shape_.empty();
            const bool has_axis = axis_.has_value();

            if ( !has_shape && !has_axis )
            {
                throw std::invalid_argument(
                    "RmsNormConfig: use RmsNormConfig( shape_t ) or RmsNormConfig( int64_t axis )" );
            }

            if ( has_shape )
            {
                for ( size_t i = 0; i < normalized_shape_.size(); ++i )
                {
                    if ( normalized_shape_[ i ] <= 0 )
                    {
                        throw std::invalid_argument(
                            "RmsNormConfig: all normalized_shape dimensions must be > 0, "
                            "invalid at index " + std::to_string( i ) );
                    }
                }
            }
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        SerializationMetadata toMetadata() const override
        {
            SerializationMetadata meta;

            meta.set( "has_bias", has_bias_ )
                .set( "epsilon", epsilon_ )
                .set( "unit_offset", unit_offset_ );

            if ( !normalized_shape_.empty() )
            {
                meta.set( "normalized_shape", normalized_shape_ );
            }
            else if ( axis_.has_value() )
            {
                meta.set( "axis", axis_.value() );
            }

            return meta;
        }

        void fromMetadata( const SerializationMetadata& meta ) override
        {
            const bool has_ns = meta.has( "normalized_shape" );
            const bool has_ax = meta.has( "axis" );

            if ( has_ns && has_ax )
            {
                throw std::invalid_argument(
                    "RmsNormConfig::fromMetadata: both normalized_shape and axis present" );
            }

            if ( has_ns )
            {
                auto maybe_shape = meta.tryGetShape( "normalized_shape" );
                if ( maybe_shape.has_value() )
                {
                    normalized_shape_ = std::move( maybe_shape.value() );
                }
            }
            else if ( has_ax )
            {
                if ( auto v = meta.tryGetInt( "axis" ) )
                {
                    axis_ = *v;
                }
            }

            if ( auto v = meta.tryGetBool( "has_bias" ) )
            {
                has_bias_ = *v;
            }

            if ( auto v = meta.tryGetFloat( "epsilon" ) )
            {
                epsilon_ = *v;
            }

            if ( auto v = meta.tryGetFloat( "unit_offset" ) )
            {
                unit_offset_ = *v;
            }
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "RmsNormConfig( ";

            if ( !normalized_shape_.empty() )
            {
                oss << "normalized_shape=[ ";
                for ( size_t i = 0; i < normalized_shape_.size(); ++i )
                {
                    oss << normalized_shape_[ i ];
                    if ( i < normalized_shape_.size() - 1 )
                    {
                        oss << ", ";
                    }
                }
                oss << " ]";
            }
            else if ( axis_.has_value() )
            {
                oss << "axis=" << axis_.value();
            }

            oss << ", has_bias=" << (has_bias_ ? "true" : "false");
            oss << ", epsilon=" << epsilon_;
            oss << ", unit_offset=" << unit_offset_;
            oss << " )";

            return oss.str();
        }

    private:

        shape_t              normalized_shape_{};
        std::optional<dim_t> axis_{ std::nullopt };
        bool                 has_bias_{ true };
        float                epsilon_{ 1e-5f };
        float                unit_offset_{ 0.0f };
    };
}
