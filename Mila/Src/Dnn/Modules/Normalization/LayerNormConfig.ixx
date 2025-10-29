/**
 * @file LayerNormConfig.ixx
 * @brief Configuration interface for the Layer Normalization module in the Mila DNN framework.
 *
 * Defines the LayerNormConfig class, providing a type-safe fluent interface for configuring
 * Layer Normalization modules. Inherits from ConfigurationBase CRTP base and adds LayerNorm-specific
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

import Dnn.ConfigurationBase;
import Dnn.TensorTypes;

namespace Mila::Dnn
{
    /**
     * @brief Configuration class for Layer Normalization module.
     *
     * Provides a type-safe fluent interface for configuring LayerNorm modules.
     */
    export class LayerNormConfig : public ConfigurationBase {
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
         * @return LayerNormConfig& Reference to this for method chaining
         */
        LayerNormConfig& withNormalizedShape( shape_t shape )
        {
            if (axis_.has_value())
            {
                throw std::invalid_argument(
                    "Cannot specify both normalized_shape and axis. Choose one approach." );
            }
            normalized_shape_ = std::move( shape );
            
            return *this;
        }

        /**
         * @brief Set the normalization axis.
         *
         * When using this method, the weight and bias tensors will be lazy-initialized
         * on the first forward pass based on the input tensor shape.
         *
         * @param axis The axis along which to normalize (default is -1, the last dimension)
         * @return LayerNormConfig& Reference to this for method chaining
         */
        LayerNormConfig& withAxis( int64_t axis )
        {
            if (!normalized_shape_.empty())
            {
                throw std::invalid_argument(
                    "Cannot specify both axis and normalized_shape. Choose one approach." );
            }
            axis_ = axis;
            return *this;
        }

        /**
         * @brief Set whether the layer should use bias.
         *
         * @param has_bias Whether to include a learnable bias term
         * @return LayerNormConfig& Reference to this for method chaining
         */
        LayerNormConfig& withBias( bool has_bias ) {
            has_bias_ = has_bias;
            return *this;
        }

        /**
         * @brief Set the epsilon value for numerical stability.
         *
         * @param epsilon Small constant added to variance for numerical stability
         * @return LayerNormConfig& Reference to this for method chaining
         */
        LayerNormConfig& withEpsilon( float epsilon ) {
            epsilon_ = epsilon;
            return *this;
        }

        /**
         * @brief Get the configured normalized shape.
         *
         * @return const std::vector<int64_t>& The normalized feature shape (empty if using axis mode)
         */
        const std::vector<int64_t>& getNormalizedShape() const
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
        void validate() const
        {
            ConfigurationBase::validate();

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
        
    private:
        
        shape_t normalized_shape_;    ///< Shape of features to normalize (empty if using axis mode)
        std::optional<dim_t> axis_;              ///< The axis along which to normalize (nullopt if using shape mode)
        bool has_bias_{ true };                ///< Whether to include a learnable bias term
        float epsilon_{ 1e-5f };               ///< Small constant added to variance for numerical stability
    };
}