/**
 * @file Gpt.Config.ixx
 * @brief Network-level configuration for GPT-style transformer networks.
 *
 * Provides only the network-level settings required for GPT networks.
 */

module;
#include <stdexcept>
#include <string>
#include <sstream>

export module Dnn.Networks.Gpt:Config;

import Dnn.TensorTypes;
import Dnn.ComponentConfig;
import Serialization.Metadata;

namespace Mila::Dnn::Networks
{
    using Serialization::SerializationMetadata;

    /**
     * @brief Network-level configuration for GPT-style transformer networks.
     *
     * Exposes the minimal network-level settings: vocabulary, number of layers,
     * embedding dimension, number of attention heads, and maximum sequence length.
     */
    export class GptConfig : public ComponentConfig
    {
    public:
        /**
         * @brief Construct a GPT network configuration.
         *
         * @param embedding_dim Model embedding dimension. Must be > 0.
         * @param num_layers Number of transformer layers. Must be > 0.
         */
        GptConfig( dim_t embedding_size, dim_t num_layers )
            : embedding_size_( embedding_size ), num_layers_( num_layers )
        {
            if ( embedding_size <= 0 )
            {
                throw std::invalid_argument( "GptConfig: embedding_dim must be > 0" );
            }

            if ( num_layers <= 0 )
            {
                throw std::invalid_argument( "GptConfig: num_layers must be > 0" );
            }
        }

        // Fluent setters

        template <typename Self>
        decltype(auto) withVocabSize( this Self&& self, dim_t vocab_size )
        {
            if ( vocab_size <= 0 )
            {
                throw std::invalid_argument( "vocab_size must be > 0" );
            }

            self.vocab_size_ = vocab_size;
            return std::forward<Self>( self );
        }

        template <typename Self>
        decltype(auto) withNumLayers( this Self&& self, dim_t num_layers )
        {
            if ( num_layers <= 0 )
            {
                throw std::invalid_argument( "num_layers must be > 0" );
            }

            self.num_layers_ = num_layers;
            return std::forward<Self>( self );
        }

        template <typename Self>
        decltype(auto) withNumHeads( this Self&& self, dim_t num_heads )
        {
            if ( num_heads <= 0 )
            {
                throw std::invalid_argument( "num_heads must be > 0" );
            }

            if ( self.embedding_size_ > 0 && (self.embedding_size_ % num_heads != 0) )
            {
                throw std::invalid_argument( "num_heads must divide embedding_dim" );
            }

            self.num_heads_ = num_heads;
            return std::forward<Self>( self );
        }

        template <typename Self>
        decltype(auto) withMaxSequenceLength( this Self&& self, dim_t max_seq_len )
        {
            if ( max_seq_len <= 0 )
            {
                throw std::invalid_argument( "max_seq_len must be > 0" );
            }

            self.max_seq_len_ = max_seq_len;
            return std::forward<Self>( self );
        }

        template <typename Self>
        decltype(auto) withHiddenSize( this Self&& self, dim_t hidden_size )
        {
            self.hidden_size_ = hidden_size;
            return std::forward<Self>( self );
        }

        template <typename Self>
        decltype(auto) withBias( this Self&& self, bool use_bias )
        {
            self.use_bias = use_bias;
            return std::forward<Self>( self );
        }

        template <typename Self>
        decltype(auto) withDropout( this Self&& self, float dropout )
        {
            if ( dropout < 0.0f || dropout > 1.0f )
            {
                throw std::invalid_argument( "dropout must be between 0.0 and 1.0" );
            }

            self.dropout = dropout;
            return std::forward<Self>( self );
        }

        // Getters

        dim_t getVocabSize() const noexcept { return vocab_size_; }
        dim_t getNumLayers() const noexcept { return num_layers_; }
        dim_t getEmbeddingSize() const noexcept { return embedding_size_; }
        dim_t getNumHeads() const noexcept { return num_heads_; }
        dim_t getMaxSequenceLength() const noexcept { return max_seq_len_; }
        
        dim_t getHiddenSize() const noexcept {
            return hidden_size_;
        }

        // Validation ensures network-level consistency
        void validate() const override
        {
            if ( vocab_size_ <= 0 )
            {
                throw std::invalid_argument( "vocab_size must be greater than zero" );
            }

            if ( num_layers_ <= 0 )
            {
                throw std::invalid_argument( "num_layers must be greater than zero" );
            }

            if ( embedding_size_ <= 0 )
            {
                throw std::invalid_argument( "embedding size must be greater than zero" );
            }

            if ( num_heads_ > 0 )
            {
                if ( embedding_size_ % num_heads_ != 0 )
                {
                    throw std::invalid_argument( "embedding size must be divisible by num_heads" );
                }
            }
        }

        // Serialization helpers

        SerializationMetadata toMetadata() const
        {
            SerializationMetadata meta;
            meta.set( "precision", static_cast<int64_t>(precision_) )
                .set( "vocab_size", static_cast<int64_t>(vocab_size_) )
                .set( "num_layers", static_cast<int64_t>(num_layers_) )
                .set( "embedding_dim", static_cast<int64_t>(embedding_size_) )
                .set( "num_heads", static_cast<int64_t>(num_heads_) )
                .set( "max_seq_len", static_cast<int64_t>(max_seq_len_) );

            return meta;
        }

        void fromMetadata( const SerializationMetadata& meta )
        {
            if ( auto p = meta.tryGetInt( "precision" ) )
                precision_ = static_cast<decltype(precision_)>(*p);
            if ( auto vs = meta.tryGetInt( "vocab_size" ) )
                vocab_size_ = static_cast<dim_t>(*vs);
            if ( auto nl = meta.tryGetInt( "num_layers" ) )
                num_layers_ = static_cast<dim_t>(*nl);
            if ( auto ed = meta.tryGetInt( "embedding_dim" ) )
                embedding_size_ = static_cast<dim_t>(*ed);
            if ( auto nh = meta.tryGetInt( "num_heads" ) )
                num_heads_ = static_cast<dim_t>(*nh);
            if ( auto msl = meta.tryGetInt( "max_seq_len" ) )
                max_seq_len_ = static_cast<dim_t>(*msl);
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "GPT Network Configuration:\n";
            oss << "  Vocab Size: " << vocab_size_ << "\n";
            oss << "  Num Layers: " << num_layers_ << "\n";
            oss << "  Embedding Size/Dim: " << embedding_size_ << "\n";
            oss << "  Num Heads: " << num_heads_ << "\n";
            oss << "  Max Seq Len: " << max_seq_len_ << "\n";

            return oss.str();
        }

    private:
        dim_t embedding_size_ = 0;
        dim_t num_layers_ = 0;

        dim_t hidden_size_ = 0;
        dim_t vocab_size_ = 0;
        dim_t num_heads_ = 0;
        dim_t max_seq_len_ = 8192;
        bool use_bias = true;
        float dropout = 0.1f;
    };
}