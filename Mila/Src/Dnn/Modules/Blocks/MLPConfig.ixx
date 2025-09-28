/**
 * @file MLPConfig.ixx
 * @brief Configuration interface for the MLP block in the Mila DNN framework.
 *
 * Defines the MLPConfig class, providing a type-safe fluent interface for configuring
 * Multi-Layer Perceptron (MLP) blocks. Inherits from ConfigurationBase CRTP base and adds
 * MLP-specific options such as input/output dimensions and activation function types.
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
     * @brief Configuration class for MLP block.
     */
    export class MLPConfig : public ConfigurationBase {
    public:
        /**
         * @brief Constructor with required parameters.
         *
         * @param input_shape The shape of the input tensor
         * @param hidden_size The size of the hidden layer (intermediate dimension)
         */
        MLPConfig( const std::vector<size_t>& input_shape, size_t hidden_size )
            : input_shape_( input_shape ), hidden_size_( hidden_size ) {

            if ( !input_shape.empty() ) {
                input_features_ = input_shape.back();
            }
        }

        /**
         * @brief Alternative constructor with direct input features specification.
         *
         * @param input_features The number of input features
         * @param hidden_size The size of the hidden layer (intermediate dimension)
         */
        MLPConfig( size_t input_features, size_t hidden_size )
            : input_features_( input_features ), hidden_size_( hidden_size ) {

            input_shape_ = { input_features };
        }

        /**
         * @brief Configure whether the linear layers use bias.
         *
         * @param has_bias Whether to include bias terms
         * @return MLPConfig& Reference to this for method chaining
         */
        MLPConfig& withBias( bool has_bias ) {
            has_bias_ = has_bias;
            return *this;
        }

        /**
         * @brief Set the activation function type.
         *
         * @param activation The activation function to use
         * @return MLPConfig& Reference to this for method chaining
         */
        MLPConfig& withActivation( ActivationType activation ) {
            activation_type_ = activation;
            return *this;
        }

        /**
         * @brief Configure whether to use layer normalization.
         *
         * @param use_layer_norm Whether to use layer normalization
         * @return MLPConfig& Reference to this for method chaining
         */
        MLPConfig& withLayerNorm( bool use_layer_norm ) {
            use_layer_norm_ = use_layer_norm;
            return *this;
        }

        const std::vector<size_t>& getInputShape() const { return input_shape_; }
        size_t getInputFeatures() const { return input_features_; }
        size_t getHiddenSize() const { return hidden_size_; }
        bool hasBias() const { return has_bias_; }
        ActivationType getActivationType() const { return activation_type_; }
        bool useLayerNorm() const { return use_layer_norm_; }

        /**
         * @brief Validate configuration parameters.
         *
         * @throws std::invalid_argument If validation fails
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