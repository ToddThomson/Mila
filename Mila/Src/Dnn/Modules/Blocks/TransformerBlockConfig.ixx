/**
 * @file TransformerBlockConfig.ixx
 * @brief Configuration interface for the TransformerBlock in the Mila DNN framework.
 *
 * Defines the TransformerBlockConfig class, providing a type-safe fluent interface for configuring
 * TransformerBlock modules. Inherits from ComponentConfig CRTP base and adds TransformerBlock-specific
 * options such as input shape, number of heads, and architectural variants.
 */

module;
#include <stdexcept>
#include <vector>

export module Dnn.Blocks.TransformerBlock:Config;

import Dnn.ComponentConfig;
import Dnn.ActivationType;

namespace Mila::Dnn
{
    /**
     * @brief Configuration class for TransformerBlock.
     *
     * Provides a type-safe fluent interface for configuring TransformerBlock modules.
     */
    export class TransformerBlockConfig : public ComponentConfig {
    public:
        /**
         * @brief Constructor with required parameters.
         *
         * @param input_shape The shape of the input tensor [batch_size, sequence_length, embedding_dim]
         * @param num_heads The number of attention heads
         */
        TransformerBlockConfig( const std::vector<size_t>& input_shape, size_t num_heads )
            : input_shape_( input_shape ), num_heads_( num_heads ) {}

        /**
         * @brief Configure the hidden dimension for the feed-forward network.
         *
         * @param hidden_dim Size of the hidden layer in the feed-forward network
         * @return TransformerBlockConfig& Reference to this for method chaining
         */
        TransformerBlockConfig& withHiddenDimension( size_t hidden_dim ) {
            hidden_dim_ = hidden_dim;
            return *this;
        }

        /**
         * @brief Configure the dropout rate.
         *
         * @param dropout Dropout probability (0.0 to 1.0)
         * @return TransformerBlockConfig& Reference to this for method chaining
         */
        TransformerBlockConfig& withDropout( float dropout ) {
            dropout_ = dropout;
            return *this;
        }

        /**
         * @brief Configure whether to use pre-layer normalization architecture.
         *
         * @param use_pre_ln Whether to use pre-layer normalization
         * @return TransformerBlockConfig& Reference to this for method chaining
         */
        TransformerBlockConfig& withPreLayerNorm( bool use_pre_ln ) {
            use_pre_ln_ = use_pre_ln;
            return *this;
        }

        /**
         * @brief Configure whether to use bias in attention and feedforward layers.
         *
         * @param use_bias Whether to use bias
         * @return TransformerBlockConfig& Reference to this for method chaining
         */
        TransformerBlockConfig& withBias( bool use_bias ) {
            use_bias_ = use_bias;
            return *this;
        }

        /**
         * @brief Configure the activation function for the MLP.
         *
         * @param activation_type The activation function type
         * @return TransformerBlockConfig& Reference to this for method chaining
         */
        TransformerBlockConfig& withActivation( ActivationType activation_type ) {
            activation_type_ = activation_type;
            return *this;
        }

        /**
         * @brief Get the input shape.
         */
        const std::vector<size_t>& getInputShape() const { return input_shape_; }

        /**
         * @brief Get the number of attention heads.
         */
        size_t getNumHeads() const { return num_heads_; }

        /**
         * @brief Get the hidden dimension for the feed-forward network.
         */
        size_t getHiddenDimension() const { return hidden_dim_; }

        /**
         * @brief Get the dropout rate.
         */
        float getDropout() const { return dropout_; }

        /**
         * @brief Check if using pre-layer normalization.
         */
        bool usePreLayerNorm() const { return use_pre_ln_; }

        /**
         * @brief Check if bias is enabled.
         */
        bool useBias() const { return use_bias_; }

        /**
         * @brief Get the activation type for the MLP.
         */
        ActivationType getActivationType() const { return activation_type_; }

        /**
         * @brief Validate configuration parameters.
         *
         * @throws std::invalid_argument If validation fails
         */
        void validate() const {
            ComponentConfig::validate();

            if ( input_shape_.size() != 3 ) {
                throw std::invalid_argument( "Input shape must have rank of 3 [batch_size, sequence_length, embedding_dim]" );
            }

            if ( input_shape_[ 2 ] % num_heads_ != 0 ) {
                throw std::invalid_argument( "Embedding dimension must be divisible by number of heads" );
            }

            if ( num_heads_ == 0 ) {
                throw std::invalid_argument( "Number of attention heads must be greater than zero" );
            }

            if ( dropout_ < 0.0f || dropout_ >= 1.0f ) {
                throw std::invalid_argument( "Dropout probability must be in range [0, 1)" );
            }

            // If hidden dimension not specified, use default of 4x embedding dimension
            /*if ( hidden_dim_ == 0 ) {
                hidden_dim_ = 4 * input_shape_[ 2 ];
            }*/
        }

    private:
        std::vector<size_t> input_shape_;
        size_t num_heads_;
        size_t hidden_dim_ = 0;  // If 0, will default to 4x embedding dimension
        float dropout_ = 0.0f;
        bool use_pre_ln_ = true;
        bool use_bias_ = true;
        ActivationType activation_type_ = ActivationType::Gelu;
    };
}