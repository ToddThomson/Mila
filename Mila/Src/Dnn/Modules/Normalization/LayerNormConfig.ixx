/**
 * @file LayerNormConfig.ixx
 * @brief Configuration interface for the Layer Normalization module in the Mila DNN framework.
 *
 * Defines the LayerNormConfig class, providing a type-safe fluent interface for configuring
 * Layer Normalization modules. Inherits from ModuleConfig CRTP base and adds LayerNorm-specific
 * options: normalization dimensions, epsilon, and bias configuration.
 *
 * Exposed as part of the LayerNorm module via module partitions.
 */

module;
#include <stdexcept>
#include <vector>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

export module Dnn.Modules.LayerNorm:Config;

import Dnn.ModuleConfig;
import Dnn.TensorTypes;
import nlohmann.json;

namespace Mila::Dnn
{
    using json = nlohmann::json;

    /**
     * @brief Configuration class for Layer Normalization module.
     *
     * Provides a type-safe fluent interface for configuring LayerNorm modules.
     */
    export class LayerNormConfig : public ModuleConfig {
    public:
        /**
         * @brief Default constructor.
         */
        LayerNormConfig() = default;

        /**
         * @brief Set the normalized shape (size of features to normalize).
         *
         * When using this method, the weight and bias tensors will be created
         * at module construction time with this exact shape.
         *
         * @param shape Vector of dimensions representing the feature shape
         * @return Self&& for method chaining
         */
        template <typename Self>
        decltype(auto) withNormalizedShape( this Self&& self, shape_t shape )
        {
            if (self.axis_.has_value())
            {
                throw std::invalid_argument(
                    "Cannot specify both normalized_shape and axis. Choose one approach." );
            }

            self.normalized_shape_ = std::move( shape );

            return std::forward<Self>( self );
        }

        /**
         * @brief Set the normalization axis.
         *
         * When using this method, the weight and bias tensors will be lazy-initialized
         * on the first forward pass based on the input tensor shape.
         *
         * @param axis The axis along which to normalize (default is -1, the last dimension)
         * @return Self&& for method chaining
         */
        template <typename Self>
        decltype(auto) withAxis( this Self&& self, int64_t axis )
        {
            if (!self.normalized_shape_.empty())
            {
                throw std::invalid_argument(
                    "Cannot specify both axis and normalized_shape. Choose one approach." );
            }

            self.axis_ = axis;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set whether the layer should use bias.
         *
         * @param has_bias Whether to include a learnable bias term
         * @return Self&& for method chaining
         */
        template <typename Self>
        decltype(auto) withBias( this Self&& self, bool has_bias )
        {
            self.has_bias_ = has_bias;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set the epsilon value for numerical stability.
         *
         * @param epsilon Small constant added to variance for numerical stability
         * @return Self&& for method chaining
         */
        template <typename Self>
        decltype(auto) withEpsilon( this Self&& self, float epsilon )
        {
            self.epsilon_ = epsilon;
            return std::forward<Self>( self );
        }

        /**
         * @brief Get the configured normalized shape.
         *
         * @return const std::vector<int64_t>& The normalized feature shape (empty if using axis mode)
         */
        const shape_t& getNormalizedShape() const
        {
            return normalized_shape_;
        }

        /**
         * @brief Check if normalized shape was specified.
         *
         * @return bool True if using normalized_shape mode, false if using axis mode
         */
        bool hasNormalizedShape() const
        {
            return !normalized_shape_.empty();
        }

        /**
         * @brief Get the configured normalization axis.
         *
         * @return std::optional<int64_t> The axis if specified, std::nullopt if using normalized_shape
         */
        std::optional<int64_t> getAxis() const
        {
            return axis_;
        }

        /**
         * @brief Check if bias is enabled.
         *
         * @return bool Whether the layer has bias enabled
         */
        bool hasBias() const { return has_bias_; }

        /**
         * @brief Get the configured epsilon value.
         *
         * @return float The epsilon value for numerical stability
         */
        float getEpsilon() const { return epsilon_; }

        /**
         * @brief Validate configuration parameters.
         *
         * @throws std::invalid_argument If validation fails
         */
        void validate() const override
        {
            //ModuleConfig::validate();

            if (epsilon_ <= 0.0f)
            {
                throw std::invalid_argument( "Epsilon must be a positive value" );
            }

            // Must specify either normalized_shape OR axis
            bool has_shape = !normalized_shape_.empty();
            bool has_axis = axis_.has_value();

            if (!has_shape && !has_axis)
            {
                throw std::invalid_argument(
                    "Must specify either normalized_shape (via withNormalizedShape) "
                    "or axis (via withAxis)" );
            }

            if (has_shape && has_axis)
            {
                throw std::invalid_argument(
                    "Cannot specify both normalized_shape and axis. "
                    "Use withNormalizedShape() OR withAxis(), not both." );
            }

            // Validate normalized_shape if provided
            if (has_shape)
            {
                for (size_t i = 0; i < normalized_shape_.size(); ++i)
                {
                    if (normalized_shape_[i] <= 0)
                    {
                        throw std::invalid_argument(
                            "All normalized_shape dimensions must be positive. "
                            "Found invalid dimension at index " + std::to_string( i ) +
                            ": " + std::to_string( normalized_shape_[i] ) );
                    }
                }
            }
        }

        /**
         * @brief Serialize configuration to JSON (ModuleConfig interface).
         *
         * Keys:
         * - "name" : string
         * - "precision" : integer (underlying value of ComputePrecision::Policy)
         * - "normalized_shape" : array<int64_t> OR "axis" : integer
         * - "has_bias" : boolean
         * - "epsilon" : float
         */
        /*json toJson() const
        {
            json j;
            j["name"] = name_;
            j["precision"] = static_cast<int>( precision_ );

            if (!normalized_shape_.empty())
            {
                j["normalized_shape"] = normalized_shape_;
            }
            else if (axis_.has_value())
            {
                j["axis"] = axis_.value();
            }

            j["has_bias"] = has_bias_;
            j["epsilon"] = epsilon_;

            return j;
        }*/

        /**
         * @brief Deserialize configuration from JSON (ModuleConfig interface).
         *
         * Missing keys leave fields at their current values. If both "normalized_shape"
         * and "axis" are provided an exception is thrown to avoid ambiguous state.
         */
        void fromJson( const json& j )
        {
            if ( j.contains( "name" ) )
            {
                name_ = j.at( "name" ).get<std::string>();
            }

            if ( j.contains( "precision" ) )
            {
                precision_ = static_cast<decltype( precision_)>( j.at( "precision" ).get<int>() );
            }

            const bool has_ns = j.contains( "normalized_shape" );
            const bool has_ax = j.contains( "axis" );

            if (has_ns && has_ax)
            {
                throw std::invalid_argument( "LayerNormConfig::fromJson: both normalized_shape and axis present" );
            }

            if ( has_ns )
            {
                normalized_shape_ = j.at( "normalized_shape" ).get<shape_t>();
            }
            else if ( has_ax )
            {
                axis_ = j.at( "axis" ).get<int64_t>();
            }

            if ( j.contains( "has_bias" ) )
            {
                has_bias_ = j.at( "has_bias" ).get<bool>();
            }

            if ( j.contains( "epsilon" ) )
            {
                epsilon_ = j.at( "epsilon" ).get<float>();
            }
        }

        std::string toString() const override
        {
            std::string repr = "LayerNormConfig(";
            if (!normalized_shape_.empty())
            {
                repr += "normalized_shape=[";
                for (size_t i = 0; i < normalized_shape_.size(); ++i)
                {
                    repr += std::to_string( normalized_shape_[i] );
                    if (i < normalized_shape_.size() - 1)
                    {
                        repr += ", ";
                    }
                }
                repr += "]";
            }
            else if (axis_.has_value())
            {
                repr += "axis=" + std::to_string( axis_.value() );
            }
            repr += ", has_bias=" + std::string( has_bias_ ? "true" : "false" );
            repr += ", epsilon=" + std::to_string( epsilon_ );
            repr += ")";
            return repr;
		}

    private:

        shape_t normalized_shape_;    ///< Shape of features to normalize (empty if using axis mode)
        std::optional<dim_t> axis_;   ///< The axis along which to normalize (nullopt if using shape mode)
        bool has_bias_{ true };       ///< Whether to include a learnable bias term
        float epsilon_{ 1e-5f };      ///< Small constant added to variance for numerical stability
    };
}