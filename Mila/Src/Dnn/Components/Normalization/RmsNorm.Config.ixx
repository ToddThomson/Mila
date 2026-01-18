/**
 * @file RmsNorm.Config.ixx
 * @brief Configuration for the RMS Normalization module.
 *
 * Provides fluent setters, validation and serialization for RMSNorm.
 */

module;
#include <stdexcept>
#include <vector>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

export module Dnn.Components.RmsNorm:Config;

import Dnn.ComponentConfig;
import Dnn.TensorTypes;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    /**
     * @brief Configuration for RMS normalization.
     *
     * Supports two modes for selecting the normalized dimensions:
     * - provide a trailing normalized_shape
     * - provide a single axis index
     *
     * Also controls whether a learnable bias is present and the epsilon
     * used for numerical stability.
     */
    export class RmsNormConfig : public ComponentConfig
    {
    public:
        RmsNormConfig() = default;

        /**
         * @brief Set the normalized shape (size of features to normalize).
         *
         * Cannot be used together with `withAxis`.
         */
        template <typename Self>
        decltype(auto) withNormalizedShape( this Self&& self, shape_t shape )
        {
            if ( self.axis_.has_value() )
            {
                throw std::invalid_argument( "Cannot specify both normalized_shape and axis. Choose one approach." );
            }

            self.normalized_shape_ = std::move( shape );

            return std::forward<Self>( self );
        }

        /**
         * @brief Set the normalization axis.
         *
         * Cannot be used together with `withNormalizedShape`.
         */
        template <typename Self>
        decltype(auto) withAxis( this Self&& self, int64_t axis )
        {
            if ( !self.normalized_shape_.empty() )
            {
                throw std::invalid_argument( "Cannot specify both axis and normalized_shape. Choose one approach." );
            }

            self.axis_ = axis;
            return std::forward<Self>( self );
        }

        /**
         * @brief Enable or disable bias.
         */
        template <typename Self>
        decltype(auto) withBias( this Self&& self, bool has_bias )
        {
            self.has_bias_ = has_bias;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set epsilon for numerical stability.
         */
        template <typename Self>
        decltype(auto) withEpsilon( this Self&& self, float epsilon )
        {
            self.epsilon_ = epsilon;
            return std::forward<Self>( self );
        }

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

        /**
         * @brief Validate configuration.
         *
         * Ensures epsilon is positive and that exactly one of normalized_shape
         * or axis is specified. Validates normalized_shape dimensions when present.
         */
        void validate() const override
        {
            if ( epsilon_ <= 0.0f )
            {
                throw std::invalid_argument( "Epsilon must be a positive value" );
            }

            bool has_shape = !normalized_shape_.empty();
            bool has_axis = axis_.has_value();

            if ( !has_shape && !has_axis )
            {
                throw std::invalid_argument(
                    "Must specify either normalized_shape (via withNormalizedShape) "
                    "or axis (via withAxis)" );
            }

            if ( has_shape && has_axis )
            {
                throw std::invalid_argument(
                    "Cannot specify both normalized_shape and axis. "
                    "Use withNormalizedShape() OR withAxis(), not both." );
            }

            if ( has_shape )
            {
                for ( size_t i = 0; i < normalized_shape_.size(); ++i )
                {
                    if ( normalized_shape_[ i ] <= 0 )
                    {
                        throw std::invalid_argument(
                            "All normalized_shape dimensions must be positive. "
                            "Found invalid dimension at index " + std::to_string( i ) +
                            ": " + std::to_string( normalized_shape_[ i ] ) );
                    }
                }
            }
        }

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
            if ( auto prec = meta.tryGetInt( "precision" ) )
            {
                precision_ = static_cast<decltype(precision_)>(*prec);
            }

            const bool has_ns = meta.has( "normalized_shape" );
            const bool has_ax = meta.has( "axis" );

            if ( has_ns && has_ax )
            {
                throw std::invalid_argument( "RmsNormConfig::fromMetadata: both normalized_shape and axis present" );
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
                if ( auto ax = meta.tryGetInt( "axis" ) )
                {
                    axis_ = *ax;
                }
            }

            if ( auto hb = meta.tryGetBool( "has_bias" ) )
            {
                has_bias_ = *hb;
            }

            if ( auto eps = meta.tryGetFloat( "epsilon" ) )
            {
                epsilon_ = *eps;
            }
        }

        std::string toString() const override
        {
            std::string repr = "RmsNormConfig(";
            if ( !normalized_shape_.empty() )
            {
                repr += "normalized_shape=[";
                for ( size_t i = 0; i < normalized_shape_.size(); ++i )
                {
                    repr += std::to_string( normalized_shape_[ i ] );
                    if ( i < normalized_shape_.size() - 1 )
                    {
                        repr += ", ";
                    }
                }
                repr += "]";
            }
            else if ( axis_.has_value() )
            {
                repr += "axis=" + std::to_string( axis_.value() );
            }

            repr += ", has_bias=" + std::string( has_bias_ ? "true" : "false" );
            repr += ", epsilon=" + std::to_string( epsilon_ );
            repr += ")";

            return repr;
        }

    private:
        shape_t normalized_shape_{};
        std::optional<dim_t> axis_{ std::nullopt };
        bool has_bias_{ true };
        float epsilon_{ 1e-5f };
    };
}