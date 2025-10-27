/**
 * @file MLPConfig.ixx
 * @brief Configuration for the MLP block.
 *
 * Provides a small, fluent configuration object used to construct MLP modules.
 * Use the fluent setters to build configuration instances and call `validate()`
 * before using the config to construct runtime modules.
 */

module;
#include <stdexcept>
#include <vector>

export module Dnn.Blocks.MLP:Config;

import Dnn.ConfigurationBase;
import Dnn.ActivationType;

namespace Mila::Dnn
{
    /**
     * @brief Configuration class for the Multi-Layer Perceptron (MLP) block.
     *
     * MLPConfig contains the required dimensional parameters (input shape or input
     * features and hidden size) and optional behavioral flags (bias, activation,
     * layer normalization). It implements a fluent interface (inherited from
     * ConfigurationBase) so callers can chain setter calls.
     *
     * Usage example:
     * @code
     * MLPConfig cfg({batch, seq_len, features}, hidden)
     *     .withBias(true)
     *     .withActivation(ActivationType::Gelu)
     *     .withLayerNorm(false)
     *     .withName("mlp");
     * cfg.validate(); // ensures config is consistent before construction
     * @endcode
     *
     * Semantics:
     * - If `input_shape` is provided it is stored as-is; when non-empty the last
     *   element of `input_shape` is recorded as `input_features`.
     * - Alternatively use the size-based constructor to specify `input_features`
     *   directly (creates an implicit 1-D `input_shape`).
     *
     * Threading / ownership:
     * - This is a simple value-type configuration object, safe to copy.
     *
     * @see ConfigurationBase
     */
    export class MLPConfig : public ConfigurationBase {
    public:
        /**
         * @brief Construct configuration from an input tensor shape and hidden size.
         *
         * @param input_shape Vector describing the input tensor shape (e.g. {batch, seq_len, features}).
         *                    An empty vector is permitted but then the `input_features` must be set via
         *                    the other constructor or the config will fail validation.
         * @param hidden_size Size of the intermediate hidden/activation layer (must be > 0).
         *
         * Postconditions:
         * - If `input_shape` is non-empty, `input_features_` is initialized to the last element.
         */
        MLPConfig( const std::vector<size_t>& input_shape, size_t hidden_size )
            : input_shape_( input_shape ), hidden_size_( hidden_size ) {

            if ( !input_shape.empty() ) {
                input_features_ = input_shape.back();
            }
        }

        /**
         * @brief Construct configuration by specifying input features directly.
         *
         * Convenience constructor for cases where only the feature dimension is required.
         *
         * @param input_features Number of input features (must be > 0 to pass validation).
         * @param hidden_size Size of the intermediate hidden/activation layer (must be > 0).
         *
         * Postconditions:
         * - `input_shape_` is set to a single-element vector containing `input_features`.
         */
        MLPConfig( size_t input_features, size_t hidden_size )
            : input_features_( input_features ), hidden_size_( hidden_size ) {

            input_shape_ = { input_features };
        }

        /**
         * @brief Configure whether the linear layers use bias.
         *
         * @param has_bias True to include bias terms in linear layers, false to omit.
         * @return Self forwarding reference for method chaining.
         *
         * Example:
         * @code
         * cfg.withBias(false).withLayerNorm(true);
         * @endcode
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
         * @param activation Activation function to use between the two linears.
         * @return Reference to this config for chaining.
         */
        MLPConfig& withActivation( ActivationType activation ) {
            activation_type_ = activation;
            return *this;
        }

        /**
         * @brief Configure whether to use layer normalization after the first linear.
         *
         * @param use_layer_norm True to enable layer normalization, false to disable.
         * @return Reference to this config for chaining.
         */
        MLPConfig& withLayerNorm( bool use_layer_norm ) {
            use_layer_norm_ = use_layer_norm;
            return *this;
        }

        /**
         * @brief Access the configured input shape.
         *
         * @return Const reference to the stored input shape vector.
         * @note An empty vector represents a scalar/unspecified-shape case and
         *       may fail validation depending on other fields.
         */
        const std::vector<size_t>& getInputShape() const { return input_shape_; }

        /**
         * @brief Get the configured number of input features.
         *
         * @return Number of input features inferred from input_shape or set directly.
         */
        size_t getInputFeatures() const { return input_features_; }

        /**
         * @brief Get the hidden (intermediate) layer size.
         *
         * @return Hidden layer dimension.
         */
        size_t getHiddenSize() const { return hidden_size_; }

        /**
         * @brief Query whether linear layers include bias terms.
         *
         * @return True if bias is enabled.
         */
        bool hasBias() const { return has_bias_; }

        /**
         * @brief Get the configured activation function type.
         *
         * @return ActivationType selected for the MLP.
         */
        ActivationType getActivationType() const { return activation_type_; }

        /**
         * @brief Query whether layer normalization is enabled.
         *
         * @return True if layer normalization is enabled.
         */
        bool useLayerNorm() const { return use_layer_norm_; }

        /**
         * @brief Validate configuration parameters.
         *
         * @throws std::invalid_argument If validation fails (e.g. zero-sized dimensions).
         */
        void validate() const {
            ConfigurationBase::validate();

            if ( input_features_ == 0 ) {
                throw std::invalid_argument( "Input features must be greater than zero" );
            }

            if ( hidden_size_ == 0 ) {
                throw std::invalid_argument( "Hidden size must be greater than zero" );
            }
        }

    private:
        std::vector<size_t> input_shape_;
        size_t input_features_{ 0 };
        size_t hidden_size_{ 0 };
        bool has_bias_{ true };
        ActivationType activation_type_{ ActivationType::Gelu};
        bool use_layer_norm_{ false };
    };
}