/**
 * @file MLPConfig.ixx
 * @brief Configuration interface for the MLP block in the Mila DNN framework.
 *
 * Defines the MLPConfig class, providing a type-safe fluent interface for configuring
 * Multi-Layer Perceptron (MLP) blocks. Inherits from ModuleConfig CRTP base and adds
 * MLP-specific options such as input/output dimensions and activation function types.
 */

module;
#include <stdexcept>
#include <vector>

export module Dnn.Blocks.MLP:Config;

import Dnn.Module;
import Dnn.ActivationType;

namespace Mila::Dnn
{
    /**
     * @brief Configuration class for MLP block.
     */
    export class MLPConfig : public ModuleConfig<MLPConfig> {
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
         * @brief Set the dropout rate.
         *
         * @param dropout Dropout probability (0.0 to 1.0)
         * @return MLPConfig& Reference to this for method chaining
         */
        MLPConfig& withDropout( float dropout ) {
            dropout_ = dropout;
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

        /**
         * @brief Configure whether to use a residual connection.
         *
         * @param use_residual Whether to add a residual connection
         * @return MLPConfig& Reference to this for method chaining
         */
        MLPConfig& withResidual( bool use_residual ) {
            use_residual_ = use_residual;
            return *this;
        }

        /**
         * @brief Configure whether to fuse operations for inference.
         *
         * @param fuse_ops Whether to fuse operations when possible
         * @return MLPConfig& Reference to this for method chaining
         */
        MLPConfig& withFusedOperations( bool fuse_ops ) {
            fuse_operations_ = fuse_ops;
            return *this;
        }

        const std::vector<size_t>& getInputShape() const { return input_shape_; }
        size_t getInputFeatures() const { return input_features_; }
        size_t getHiddenSize() const { return hidden_size_; }
        bool hasBias() const { return has_bias_; }
        ActivationType getActivationType() const { return activation_type_; }
        float getDropout() const { return dropout_; }
        bool useLayerNorm() const { return use_layer_norm_; }
        bool useResidual() const { return use_residual_; }
        bool useFusedOperations() const { return fuse_operations_; }

        /**
         * @brief Validate configuration parameters.
         *
         * @throws std::invalid_argument If validation fails
         */
        void validate() const {
            ModuleConfig<MLPConfig>::validate();

            if ( input_features_ == 0 ) {
                throw std::invalid_argument( "Input features must be greater than zero" );
            }

            if ( hidden_size_ == 0 ) {
                throw std::invalid_argument( "Hidden size must be greater than zero" );
            }

            if ( dropout_ < 0.0f || dropout_ >= 1.0f ) {
                throw std::invalid_argument( "Dropout probability must be in range [0, 1)" );
            }
        }

    private:
        std::vector<size_t> input_shape_;
        size_t input_features_{ 0 };
        size_t hidden_size_{ 0 };
        bool has_bias_{ true };
        ActivationType activation_type_{ ActivationType::Gelu};
        float dropout_{ 0.0f };
        bool use_layer_norm_{ false };
        bool use_residual_{ false };
        bool fuse_operations_{ false };
    };
}