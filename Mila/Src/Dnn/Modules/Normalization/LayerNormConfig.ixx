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

export module Dnn.Modules.LayerNorm:Config;

import Dnn.ConfigurationBase;

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
         * @brief Constructor with normalized dimension size.
         *
         * @param normalized_dim The dimension size to normalize
         */
        explicit LayerNormConfig( size_t normalized_dim ) {
            // Store the normalized dimension as a single-element vector
            input_shape_ = { 1, 1, normalized_dim };
        }

        /**
         * @brief Set the input shape for the layer normalization.
         *
         * @param input_shape The input tensor shape [batch_size, sequence_length, channels]
         * @return LayerNormConfig& Reference to this for method chaining
         */
        LayerNormConfig& withInputShape( const std::vector<size_t>& input_shape ) {
            input_shape_ = input_shape;
            return *this;
        }

        /**
         * @brief Set the normalization axis.
         *
         * @param axis The axis along which to normalize (default is -1, the last dimension)
         * @return LayerNormConfig& Reference to this for method chaining
         */
        LayerNormConfig& withAxis( int64_t axis ) {
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
         * @brief Get the configured input shape.
         *
         * @return const std::vector<size_t>& The input tensor shape
         */
        const std::vector<size_t>& getInputShape() const { return input_shape_; }

        /**
         * @brief Get the configured normalization axis.
         *
         * @return int64_t The axis along which to normalize
         */
        int64_t getAxis() const { return axis_; }

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
        void validate() const {
            ConfigurationBase::validate();

            if ( input_shape_.empty() ) {
                throw std::invalid_argument( "Input shape cannot be empty" );
            }

            if ( epsilon_ <= 0.0f ) {
                throw std::invalid_argument( "Epsilon must be a positive value" );
            }
        }

    private:
        std::vector<size_t> input_shape_{};  ///< Shape of the input tensor [batch_size, sequence_length, channels]
        int64_t axis_{ -1 };                   ///< The axis along which to normalize (default: -1 for last dimension)
        bool has_bias_{ true };                ///< Whether to include a learnable bias term
        float epsilon_{ 1e-5f };               ///< Small constant added to variance for numerical stability
    };
}