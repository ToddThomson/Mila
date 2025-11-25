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
#include <string>
#include <utility>

export module Dnn.Blocks.MLP:Config;

import Dnn.TensorTypes;
import Dnn.ComponentConfig;
import Dnn.ActivationType;
import nlohmann.json;

namespace Mila::Dnn
{
    using json = nlohmann::json;

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
     * @see ModuleConfig
     */
    export class MLPConfig : public ComponentConfig
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
         * @brief Set the activation function type (C++23 fluent style).
         */
        template <typename Self>
        decltype(auto) withActivation( this Self&& self, ActivationType activation )
        {
            self.activation_type_ = activation;
            return std::forward<Self>( self );
        }

        /**
         * @brief Configure whether to use layer normalization after the first linear (C++23 fluent style).
         */
        template <typename Self>
        decltype(auto) withLayerNorm( this Self&& self, bool use_layer_norm )
        {
            self.use_layer_norm_ = use_layer_norm;
            return std::forward<Self>( self );
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
            //ComponentConfig::validate();

            if (input_features_ <= 0)
            {
                throw std::invalid_argument( "MLPConfig: Input features must be greater than zero" );
            }

            if (hidden_size_ <= 0)
            {
                throw std::invalid_argument( "MLPConfig: Hidden size must be greater than zero" );
            }
        }

        /**
         * @brief Serialize this configuration to JSON (ModuleConfig interface).
         *
         * Keys:
         * - "name" : string
         * - "precision" : integer (underlying value of ComputePrecision::Policy)
         * - "input_features" : integer
         * - "hidden_size" : integer
         * - "has_bias" : boolean
         * - "activation" : integer (ActivationType)
         * - "use_layer_norm" : boolean
         */
        /*json toJson() const override
        {
            json j;
            j["name"] = name_;
            j["precision"] = static_cast<int>( precision_ );
            j["input_features"] = static_cast<int64_t>( input_features_ );
            j["hidden_size"] = static_cast<int64_t>( hidden_size_ );
            j["has_bias"] = has_bias_;
            j["activation"] = static_cast<int>( activation_type_ );
            j["use_layer_norm"] = use_layer_norm_;

            return j;
        }*/

        /**
         * @brief Deserialize configuration from JSON (ModuleConfig interface).
         *
         * Missing keys leave fields at their current values. Type errors are
         * propagated from nlohmann::json getters.
         */
        /*void fromJson( const json& j ) override
        {
            if ( j.contains( "name" ) )
            {
                name_ = j.at( "name" ).get<std::string>();
            }

            if ( j.contains( "precision" ) )
            {
                precision_ = static_cast<decltype( precision_)>( j.at( "precision" ).get<int>() );
            }

            if ( j.contains( "input_features" ) )
            {
                input_features_ = static_cast<dim_t>( j.at( "input_features" ).get<int64_t>() );
            }

            if ( j.contains( "hidden_size" ) )
            {
                hidden_size_ = static_cast<dim_t>( j.at( "hidden_size" ).get<int64_t>() );
            }

            if ( j.contains( "has_bias" ) )
            {
                has_bias_ = j.at( "has_bias" ).get<bool>();
            }

            if ( j.contains( "activation" ) )
            {
                activation_type_ = static_cast<ActivationType>( j.at( "activation" ).get<int>() );
            }

            if ( j.contains( "use_layer_norm" ) )
            {
                use_layer_norm_ = j.at( "use_layer_norm" ).get<bool>();
            }
        }*/

        std::string toString() const override
        {
            std::string repr = "MLPConfig(";
            repr += "input_features=" + std::to_string( input_features_ ) + ", ";
            repr += "hidden_size=" + std::to_string( hidden_size_ ) + ", ";
            repr += "has_bias=" + std::string( has_bias_ ? "true" : "false" ) + ", ";
            repr += "activation=" + std::to_string( static_cast<int>( activation_type_ ) ) + ", ";
            repr += "use_layer_norm=" + std::string( use_layer_norm_ ? "true" : "false" );
            repr += ")";
            return repr;
		}

    private:
        dim_t input_features_{ 0 };
        dim_t hidden_size_{ 0 };
        bool has_bias_{ true };
        ActivationType activation_type_{ ActivationType::Gelu };
        bool use_layer_norm_{ false };
    };
}