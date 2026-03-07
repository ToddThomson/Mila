/**
 * @file LayerNorm.Config.ixx
 * @brief Configuration for Layer Normalization component.
 *
 * Design principle (Mila-wide):
 *   - Constructor parameters are structurally required — no sensible default exists.
 *   - Fluent setters are reserved for optional behavioural parameters that have
 *     well-known defaults. There are no fluent overrides for constructor parameters.
 *
 * LayerNormConfig supports two mutually exclusive normalization modes selected
 * by constructor overload:
 *   - Shape mode:  LayerNormConfig( shape_t )   — normalize over a trailing shape.
 *   - Axis mode:   LayerNormConfig( int64_t )   — normalize over a single axis.
 *
 * The overloads are unambiguous — shape_t and int64_t cannot collide.
 *
 * Typical usage:
 * @code
 * // Shape mode (most common for transformers)
 * auto cfg = LayerNormConfig( shape_t{ model_dim } )
 *     .withEpsilon( 1e-5f )
 *     .withBias( false );
 *
 * // Axis mode
 * auto cfg = LayerNormConfig( int64_t{ -1 } )
 *     .withBias( false );
 * @endcode
 */

module;
#include <stdexcept>
#include <string>
#include <sstream>
#include <optional>
#include <utility>

export module Dnn.Components.LayerNorm:Config;

import Dnn.Component;
import Dnn.ComponentConfig;
import Dnn.TensorTypes;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    export class LayerNormConfig : public ComponentConfig
    {
    public:

        // ====================================================================
        // Constructors — pick exactly one normalization mode.
        // ====================================================================

        /**
         * @brief Construct in shape mode.
         *
         * Normalizes over the trailing dimensions described by @p shape.
         *
         * @param shape  Trailing dimensions to normalize over (e.g. shape_t{ model_dim }).
         */
        explicit LayerNormConfig( shape_t shape )
            : normalized_shape_( std::move( shape ) )
        {}

        /**
         * @brief Construct in axis mode.
         *
         * Normalizes over a single axis.
         *
         * @param axis  Axis along which to normalize (negative indexing supported).
         */
        explicit LayerNormConfig( int64_t axis )
            : axis_( axis )
        {
            // REVIEW: these constructors are not unambiguous. We could add static factory methods
            // if we wanted to be more explicit at the call site 
            // (e.g. LayerNormConfig::withShape( shape_t{ dim } ) and LayerNormConfig::withAxis( -1 ) ).
        }

        // ====================================================================
        // Optional fluent setters — behavioural parameters with sensible defaults.
        // No fluent overrides exist for constructor parameters.
        // ====================================================================

        /**
         * @brief Enable or disable learnable bias.
         *
         * Default: true.
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
         */
        template <typename Self>
        decltype(auto) withEpsilon( this Self&& self, float epsilon )
        {
            self.epsilon_ = epsilon;
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

        // ====================================================================
        // Validation
        // ====================================================================

        void validate() const override
        {
            if ( epsilon_ <= 0.0f )
            {
                throw std::invalid_argument( "LayerNormConfig: epsilon must be > 0" );
            }

            const bool has_shape = !normalized_shape_.empty();
            const bool has_axis = axis_.has_value();

            if ( !has_shape && !has_axis )
            {
                throw std::invalid_argument(
                    "LayerNormConfig: use LayerNormConfig( shape_t ) or LayerNormConfig( int64_t axis )" );
            }

            if ( has_shape )
            {
                for ( size_t i = 0; i < normalized_shape_.size(); ++i )
                {
                    if ( normalized_shape_[ i ] <= 0 )
                    {
                        throw std::invalid_argument(
                            "LayerNormConfig: all normalized_shape dimensions must be > 0, "
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

            meta.set( "precision", static_cast<int64_t>(precision_) )
                .set( "has_bias", has_bias_ )
                .set( "epsilon", epsilon_ );

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
            if ( auto v = meta.tryGetInt( "precision" ) )
            {
                precision_ = static_cast<decltype(precision_)>(*v);
            }

            const bool has_ns = meta.has( "normalized_shape" );
            const bool has_ax = meta.has( "axis" );

            if ( has_ns && has_ax )
            {
                throw std::invalid_argument(
                    "LayerNormConfig::fromMetadata: both normalized_shape and axis present" );
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
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "LayerNormConfig( ";

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
            oss << " )";

            return oss.str();
        }

    private:

        shape_t normalized_shape_{};
        std::optional<dim_t> axis_{ std::nullopt };
        bool has_bias_{ true };
        float epsilon_{ 1e-5f };
    };
}
