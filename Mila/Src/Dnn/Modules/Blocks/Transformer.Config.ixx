/**
 * @file TransformerConfig.ixx
 * @brief Configuration for the Transformer composite module.
 *
 * Lightweight fluent configuration used to construct Transformer modules.
 */

module;
#include <stdexcept>
#include <vector>

export module Dnn.Blocks.Transformer:Config;

import Dnn.TensorTypes;
import Dnn.ConfigurationBase;
import Dnn.ActivationType;

namespace Mila::Dnn
{
    /**
     * @brief Configuration class for Transformer modules.
     *
     * Holds the embedding dimension, attention head count and MLP/attention options.
     * Instances are intended to be validated and then passed to Transformer constructors.
     */
    export class TransformerConfig : public ConfigurationBase
    {
    public:
        /**
         * @brief Construct a Transformer configuration.
         *
         * @param embedding_dim Model embedding dimension. Must be > 0.
         * @param num_heads Number of attention heads. Must be > 0 and must divide embedding_dim evenly.
         *
         * The constructor performs immediate validation of these required invariants
         * and throws std::invalid_argument if they are violated.
         */
        TransformerConfig( dim_t embedding_dim, dim_t num_heads )
			: embedding_dim_( embedding_dim ), num_heads_( num_heads )
        {
            if (embedding_dim <= 0)
            {
                throw std::invalid_argument( "TransformerConfig: embedding_dim must be > 0" );
            }

            if (num_heads <= 0)
            {
                throw std::invalid_argument( "TransformerConfig: num_heads must be > 0" );
            }

            if (embedding_dim % num_heads != 0)
            {
                throw std::invalid_argument( "TransformerConfig: embedding_dim must be divisible by num_heads" );
            }
        }

        /**
         * @brief Configure the hidden dimension for the feed-forward network.
         *
         * If not set 0 the implementation may choose a sensible default
         * commonly 4 * embedding_dim.
         *
         * @param hidden_dim Hidden layer size for the MLP.
         * @return Reference to this config for method chaining.
         */
        TransformerConfig& withHiddenDimension( dim_t hidden_dim )
        {
            hidden_dim_ = hidden_dim;
            return *this;
        }

        /**
         * @brief Enable or disable bias terms in attention and feed-forward layers.
         *
         * @param use_bias true to enable bias, false to disable.
         * @return Reference to this config for method chaining.
         */
        TransformerConfig& withBias( bool use_bias )
        {
            use_bias_ = use_bias;
            return *this;
        }

        /**
         * @brief Select activation function for the MLP.
         *
         * @param activation_type Activation type such as Gelu or Relu.
         * @return Reference to this config for method chaining.
         */
        TransformerConfig& withActivation( ActivationType activation_type )
        {
            activation_type_ = activation_type;
            return *this;
        }

        /**
         * @brief Get the embedding dimension.
         *
         * @return Embedding dimension.
         */
        dim_t getEmbeddingDim() const
        {
            return embedding_dim_;
        }

        /**
         * @brief Get the number of attention heads.
         *
         * @return Number of heads.
         */
        dim_t getNumHeads() const
        {
            return num_heads_;
        }

        /**
         * @brief Get the hidden dimension for the feed-forward network.
         *
         * @return Hidden dimension (0 if not set).
         */
        size_t getHiddenDimension() const
        {
            return hidden_dim_;
        }

        /**
         * @brief Query whether bias is enabled.
         *
         * @return True if bias is enabled for Linear/Attention layers.
         */
        bool useBias() const
        {
            return use_bias_;
        }

        /**
         * @brief Get the activation type for the MLP.
         *
         * @return ActivationType configured for the MLP.
         */
        ActivationType getActivationType() const
        {
            return activation_type_;
        }

        /**
         * @brief Validate configuration parameters.
         *
         * Ensures:
         *  - embedding_dim > 0
         *  - num_heads > 0
         *  - embedding_dim is divisible by num_heads
         *
         * Throws std::invalid_argument on invalid configuration.
         */
        void validate() const
        {
            ConfigurationBase::validate();

            if (embedding_dim_ <= 0)
            {
                throw std::invalid_argument( "Embedding dimension must be greater than zero" );
            }

            if (num_heads_ == 0)
            {
                throw std::invalid_argument( "Number of attention heads must be greater than zero" );
            }

            if (embedding_dim_ % num_heads_ != 0)
            {
                throw std::invalid_argument( "Embedding dimension must be divisible by number of heads" );
            }
        }

    private:
        dim_t embedding_dim_;
        dim_t num_heads_;
        dim_t hidden_dim_ = 0;  // If 0, implementation may default to 4x embedding_dim
        bool use_bias_ = true;
        ActivationType activation_type_ = ActivationType::Gelu;
    };
}