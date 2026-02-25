/**
 * @file MLPConfig.ixx
 * @brief Configuration for the MLP block.
 *
 * Provides a fluent configuration object used to construct MLP components.
 */

module;
#include <stdexcept>
#include <cstdint>
#include <string>
#include <utility>
#include <sstream>

export module Dnn.Components.MLP:Config;

import Dnn.TensorTypes;
import Dnn.ComponentConfig;
import Dnn.ActivationType;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    /**
     * @brief Configuration class for the Multi-Layer Perceptron (MLP) block.
     *
     * MLPConfig specifies the architectural parameters for an MLP block:
     * - Input and hidden feature dimensions
     * - Activation function type
     * - Optional bias and layer normalization
     *
     * The MLP block structure is:
     *   Input -> Linear(in_features, hidden_size) -> [LayerNorm] -> Activation -> Linear(hidden_size, in_features) -> Output
     */
    export class MLPConfig : public ComponentConfig
    {
    public:
        /**
         * @brief Construct MLP configuration.
         *
         * @param input_features Number of input (and output) features (must be > 0).
         * @param hidden_size Size of the intermediate hidden layer (must be > 0).
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
         * @brief Get the hidden layer size.
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
         * @brief Convert configuration into SerializationMetadata.
         *
         * Produces keys:
         * - "precision" : integer (underlying value of ComputePrecision::Policy)
         * - "input_features" : integer
         * - "hidden_size" : integer
         * - "has_bias" : boolean
         * - "activation" : integer (ActivationType)
         * - "use_layer_norm" : boolean
         *
         * @return SerializationMetadata Metadata representing this configuration.
         */
        SerializationMetadata toMetadata() const override
        {
            SerializationMetadata meta;
            meta.set( "precision", static_cast<int64_t>( precision_ ) )
                .set( "input_features", static_cast<int64_t>( input_features_ ) )
                .set( "hidden_size", static_cast<int64_t>( hidden_size_ ) )
                .set( "has_bias", has_bias_ )
                .set( "activation", static_cast<int64_t>( activation_type_ ) )
                .set( "use_layer_norm", use_layer_norm_ );

            return meta;
        }

        /**
         * @brief Populate configuration from provided metadata.
         *
         * Missing keys are ignored. Type-safe try-get helpers are used to avoid
         * throwing on absent fields and to preserve forward/backward compatibility.
         *
         * @param meta Metadata to read configuration values from.
         */
        void fromMetadata( const SerializationMetadata& meta ) override
        {
            if ( auto p = meta.tryGetInt( "precision" ) )
            {
                precision_ = static_cast<decltype( precision_ )>( *p );
            }

            if ( auto in = meta.tryGetInt( "input_features" ) )
            {
                input_features_ = static_cast<dim_t>( *in );
            }

            if ( auto hs = meta.tryGetInt( "hidden_size" ) )
            {
                hidden_size_ = static_cast<dim_t>( *hs );
            }

            if ( auto hb = meta.tryGetBool( "has_bias" ) )
            {
                has_bias_ = *hb;
            }

            if ( auto act = meta.tryGetInt( "activation" ) )
            {
                activation_type_ = static_cast<ActivationType>( *act );
            }

            if ( auto ln = meta.tryGetBool( "use_layer_norm" ) )
            {
                use_layer_norm_ = *ln;
            }
        }

        std::string toString() const override
        {
            std::ostringstream oss;

            oss << "MLPConfig(";
            oss << "input_features=" << input_features_ << ", ";
            oss << "hidden_size=" << hidden_size_ << ", ";
            oss << "has_bias=" << (has_bias_ ? "true" : "false") << ", ";
            oss << "activation=" << static_cast<int>(activation_type_) << ", ";
            oss << "use_layer_norm=" << (use_layer_norm_ ? "true" : "false");
            oss << ")";

            return oss.str();
        }

    private:
        dim_t input_features_{ 0 };
        dim_t hidden_size_{ 0 };
        bool has_bias_{ true };
        ActivationType activation_type_{ ActivationType::Gelu };

        // REVIEW: The use of layer normalization in an MLP block is non-standard;
        // it was a speculative addition that doesn't seem to be needed.
        // Removing it simplifies the component graph and would allow fused decode path.
        // TODO: Remove
        bool use_layer_norm_{ false };
    };
}