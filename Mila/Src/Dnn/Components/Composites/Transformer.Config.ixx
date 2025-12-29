/**
 * @file TransformerConfig.ixx
 * @brief Configuration for the Transformer composite module.
 *
 * Lightweight fluent configuration used to construct Transformer modules.
 */

module;
#include <stdexcept>
#include <vector>
#include <utility>
#include <string>
#include <sstream>

export module Dnn.Blocks.Transformer:Config;

import Dnn.TensorTypes;
import Dnn.ComponentConfig;
import Dnn.ActivationType;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    /**
     * @brief Configuration class for Transformer modules.
     *
     * Holds the embedding dimension, attention head count and MLP/attention options.
     * Instances are intended to be validated and then passed to Transformer constructors.
     */
    export class TransformerConfig : public ComponentConfig
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
            if ( embedding_dim <= 0 )
            {
                throw std::invalid_argument( "TransformerConfig: embedding_dim must be > 0" );
            }

            if ( num_heads <= 0 )
            {
                throw std::invalid_argument( "TransformerConfig: num_heads must be > 0" );
            }

            if ( embedding_dim % num_heads != 0 )
            {
                throw std::invalid_argument( "TransformerConfig: embedding_dim must be divisible by num_heads" );
            }
        }

        /**
         * @brief fluent setter for hidden dimension.
         *
         * If set to 0 the Transformer implementation will default the MLP hidden
         * dimension to 4 × embedding_dim (conventional Transformer default).
         *
         * @param hidden_dim Hidden dimension for the feed-forward network. Use 0
         * to request the implementation default (4 × embedding_dim).
         */
        template <typename Self>
        decltype(auto) withHiddenDimension( this Self&& self, dim_t hidden_dim )
        {
            self.hidden_dim_ = hidden_dim;
            return std::forward<Self>( self );
        }

        /**
         * @brief C++23-style fluent setter to enable/disable bias.
         */
        template <typename Self>
        decltype(auto) withBias( this Self&& self, bool use_bias )
        {
            self.use_bias_ = use_bias;
            return std::forward<Self>( self );
        }

        /**
         * @brief C++23-style fluent setter for activation type.
         */
        template <typename Self>
        decltype(auto) withActivation( this Self&& self, ActivationType activation_type )
        {
            self.activation_type_ = activation_type;
            return std::forward<Self>( self );
        }

        /**
         * @brief Get the embedding dimension.
         *
         * @return Embedding dimension.
         */
        dim_t getEmbeddingDim() const noexcept
        {
            return embedding_dim_;
        }

        /**
         * @brief Get the number of attention heads.
         *
         * @return Number of heads.
         */
        dim_t getNumHeads() const noexcept
        {
            return num_heads_;
        }

        /**
         * @brief Get the hidden dimension for the feed-forward network.
         *
         * @return Hidden dimension (0 if not set).
         */
        dim_t getHiddenDimension() const noexcept
        {
            return hidden_dim_;
        }

        /**
         * @brief Query whether bias is enabled.
         *
         * @return True if bias is enabled for Linear/Attention layers.
         */
        bool useBias() const noexcept
        {
            return use_bias_;
        }

        /**
         * @brief Get the activation type for the MLP.
         *
         * @return ActivationType configured for the MLP.
         */
        ActivationType getActivationType() const noexcept
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
        void validate() const override
        {
            //ComponentConfig::validate();

            if ( embedding_dim_ <= 0 )
            {
                throw std::invalid_argument( "Embedding dimension must be greater than zero" );
            }

            if ( num_heads_ == 0 )
            {
                throw std::invalid_argument( "Number of attention heads must be greater than zero" );
            }

            if ( embedding_dim_ % num_heads_ != 0 )
            {
                throw std::invalid_argument( "Embedding dimension must be divisible by number of heads" );
            }
        }

        /**
         * @brief Convert configuration to serialization metadata.
         *
         * Produces a SerializationMetadata object containing the configuration
         * fields suitable for writing into an archive by the caller.
         */
        SerializationMetadata toMetadata() const
        {
            SerializationMetadata meta;
            meta.set( "precision", static_cast<int64_t>(precision_) )
                .set( "embedding_dim", static_cast<int64_t>(embedding_dim_) )
                .set( "num_heads", static_cast<int64_t>(num_heads_) )
                .set( "hidden_dim", static_cast<int64_t>(hidden_dim_) )
                .set( "use_bias", use_bias_ )
                .set( "activation", static_cast<int64_t>(activation_type_) );

            return meta;
        }

        /**
         * @brief Deserialize this configuration from SerializationMetadata.
         *
         * Missing keys leave fields at their current values.
         */
        void fromMetadata( const SerializationMetadata& meta )
        {
            if ( auto p = meta.tryGetInt( "precision" ) )
            {
                precision_ = static_cast<decltype(precision_)>(*p);
            }

            if ( auto ed = meta.tryGetInt( "embedding_dim" ) )
            {
                embedding_dim_ = static_cast<dim_t>(*ed);
            }

            if ( auto nh = meta.tryGetInt( "num_heads" ) )
            {
                num_heads_ = static_cast<dim_t>(*nh);
            }

            if ( auto hd = meta.tryGetInt( "hidden_dim" ) )
            {
                hidden_dim_ = static_cast<dim_t>(*hd);
            }

            if ( auto ub = meta.tryGetBool( "use_bias" ) )
            {
                use_bias_ = *ub;
            }

            if ( auto act = meta.tryGetInt( "activation" ) )
            {
                activation_type_ = static_cast<ActivationType>(*act);
            }
        }

        std::string toString() const override
        {
            std::ostringstream oss;

            oss << "Transformer: " << std::endl;
            oss << "Embedding Dim: " << embedding_dim_ << std::endl;
            oss << "Num Heads: " << num_heads_ << std::endl;
            oss << "Hidden Dim: " << hidden_dim_ << std::endl;
            oss << "Use Bias: " << (use_bias_ ? "Yes" : "No") << std::endl;
            oss << "Activation: " << static_cast<int>(activation_type_) << std::endl;

            return oss.str();
        }

    private:
        dim_t embedding_dim_;
        dim_t num_heads_;
        dim_t hidden_dim_ = 0;  // Default to 4x embedding_dim
        bool use_bias_ = true;
        ActivationType activation_type_ = ActivationType::Gelu;
    };
}