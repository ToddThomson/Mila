/**
 * @file MLPConfig.ixx
 * @brief Configuration for the MLP block.
 *
 * Provides a fluent configuration object used to construct MLP modules.
 * Use the fluent setters to build configuration instances and call `validate()`
 * before using the config to construct runtime modules.
 */

module;
#include <stdexcept>
#include <cstdint>

export module Dnn.Blocks.MLP:Config;

import Dnn.TensorTypes;
import Dnn.ConfigurationBase;
import Dnn.ActivationType;

namespace Mila::Dnn
{
    /**
     * @brief Configuration class for the Multi-Layer Perceptron (MLP) block.
     *
     * MLPConfig specifies the architectural parameters for an MLP block:
     * - Input and hidden feature dimensions
     * - Activation function type
     * - Optional bias and layer normalization
     *
     * The MLP block structure is:
     *   Input ? Linear(in_features, hidden_size) ? [LayerNorm] ? Activation ? Linear(hidden_size, in_features) ? Output
     *
     * Usage example:
     * @code
     * MLPConfig cfg(512, 2048);  // input_features=512, hidden_size=2048
     * cfg.withBias(true)
     *    .withActivation(ActivationType::Gelu)
     *    .withLayerNorm(false)
     *    .withName("mlp_block");
     * cfg.validate();
     * @endcode
     *
     * Note: Input shape is determined at build time from the actual input tensor,
     *       not specified in the configuration.
     *
     * @see ConfigurationBase
     */
    export class MLPConfig : public ConfigurationBase
    {
    public:
        /**
         * @brief Construct MLP configuration.
         *
         * @param input_features Number of input (and output) features (must be > 0).
         * @param hidden_size Size of the intermediate hidden layer (must be > 0).
         *
         * The MLP expands from input_features to hidden_size in the first linear layer,
         * applies activation, then projects back down to input_features in the second layer.
         */
        MLPConfig( dim_t input_features, dim_t hidden_size )
            : input_features_( input_features ), hidden_size_( hidden_size )
        {
        }

        /**
         * @brief Configure whether the linear layers use bias.
         *
         * @param has_bias True to include bias terms in linear layers, false to omit.
         * @return Self forwarding reference for method chaining.
         */
        template <typename Self>
        Self&& withBias( this Self&& self, bool has_bias )
        {
            self.has_bias_ = has_bias;
            return std::forward<Self>( self );
        }

        /**
         * @brief Set the activation function type.
         *
         * @param activation Activation function to use between the two linear layers.
         * @return Reference to this config for chaining.
         */
        MLPConfig& withActivation( ActivationType activation )
        {
            activation_type_ = activation;
            return *this;
        }

        /**
         * @brief Configure whether to use layer normalization after the first linear.
         *
         * @param use_layer_norm True to enable layer normalization, false to disable.
         * @return Reference to this config for chaining.
         */
        MLPConfig& withLayerNorm( bool use_layer_norm )
        {
            use_layer_norm_ = use_layer_norm;
            return *this;
        }

        /**
         * @brief Get the configured number of input (and output) features.
         *
         * @return Number of input features.
         */
        dim_t getInputFeatures() const noexcept
        {
            return input_features_;
        }

        /**
         * @brief Get the hidden (intermediate) layer size.
         *
         * @return Hidden layer dimension.
         */
        dim_t getHiddenSize() const noexcept
        {
            return hidden_size_;
        }

        /**
         * @brief Query whether linear layers include bias terms.
         *
         * @return True if bias is enabled.
         */
        bool hasBias() const noexcept
        {
            return has_bias_;
        }

        /**
         * @brief Get the configured activation function type.
         *
         * @return ActivationType selected for the MLP.
         */
        ActivationType getActivationType() const noexcept
        {
            return activation_type_;
        }

        /**
         * @brief Query whether layer normalization is enabled.
         *
         * @return True if layer normalization is enabled.
         */
        bool useLayerNorm() const noexcept
        {
            return use_layer_norm_;
        }

        /**
         * @brief Validate configuration parameters.
         *
         * @throws std::invalid_argument If validation fails (e.g. zero-sized dimensions).
         */
        void validate() const override
        {
            ConfigurationBase::validate();

            if (input_features_ <= 0)
            {
                throw std::invalid_argument( "MLPConfig: Input features must be greater than zero" );
            }

            if (hidden_size_ <= 0)
            {
                throw std::invalid_argument( "MLPConfig: Hidden size must be greater than zero" );
            }
        }

    private:
        dim_t input_features_{ 0 };
        dim_t hidden_size_{ 0 };
        bool has_bias_{ true };
        ActivationType activation_type_{ ActivationType::Gelu };
        bool use_layer_norm_{ false };
    };
}