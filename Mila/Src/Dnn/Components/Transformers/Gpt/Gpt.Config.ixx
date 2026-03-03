/**
 * @file Gpt.Config.ixx
 * @brief Network-level configuration for GPT-style transformer networks.
 *
 * Minimal, canonical GPT network settings used to construct the model.
 */

module;
#include <stdexcept>
#include <string>
#include <sstream>

export module Dnn.Components.GptTransformer:Config;

import Dnn.TensorTypes;
import Dnn.ComponentConfig;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    /**
     * @brief Network-level configuration for GPT-style transformer networks.
     *
     * Contains only the minimal network-level settings required by GPT
     * networks: embedding dim, number of layers, heads, vocabulary and max seq len.
     */
    export class GptConfig : public ComponentConfig
    {
    public:
        GptConfig( dim_t embedding_size, dim_t num_layers)
            : embedding_size_( embedding_size ), num_layers_( num_layers )
        {}

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
            self.use_bias_ = use_bias;
            return std::forward<Self>( self );
        }

        // Getters
        dim_t getVocabSize() const noexcept { return vocab_size_; }
        dim_t getNumLayers() const noexcept { return num_layers_; }
        dim_t getEmbeddingSize() const noexcept { return embedding_size_; }
        dim_t getNumHeads() const noexcept { return num_heads_; }
        dim_t getMaxSequenceLength() const noexcept { return max_seq_len_; }
        dim_t getHiddenSize() const noexcept { return hidden_size_; }
        bool getUseBias() const noexcept { return use_bias_; }

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

        SerializationMetadata toMetadata() const
        {
            SerializationMetadata meta;
            meta.set( "vocab_size", static_cast<int64_t>(vocab_size_) )
                .set( "num_layers", static_cast<int64_t>(num_layers_) )
                .set( "embedding_dim", static_cast<int64_t>(embedding_size_) )
                .set( "num_heads", static_cast<int64_t>(num_heads_) )
                .set( "max_seq_len", static_cast<int64_t>(max_seq_len_) )
                .set( "hidden_dim", static_cast<int64_t>(hidden_size_) )
                .set( "use_bias", static_cast<bool>(use_bias_) );

            return meta;
        }

        void fromMetadata( const SerializationMetadata& meta )
        {
            // Required integer fields with common fallbacks used by exporters
            if ( auto vs = meta.tryGetInt( "vocab_size" ) )
                vocab_size_ = static_cast<dim_t>(*vs);
            else if ( auto vs2 = meta.tryGetInt( "vocab" ) )
                vocab_size_ = static_cast<dim_t>(*vs2);

            if ( auto nl = meta.tryGetInt( "num_layers" ) )
                num_layers_ = static_cast<dim_t>(*nl);
            else if ( auto nl2 = meta.tryGetInt( "n_layer" ) )
                num_layers_ = static_cast<dim_t>(*nl2);

            if ( auto ed = meta.tryGetInt( "embedding_dim" ) )
                embedding_size_ = static_cast<dim_t>(*ed);
            else if ( auto ed2 = meta.tryGetInt( "n_embd" ) )
                embedding_size_ = static_cast<dim_t>(*ed2);

            if ( auto nh = meta.tryGetInt( "num_heads" ) )
                num_heads_ = static_cast<dim_t>(*nh);
            else if ( auto nh2 = meta.tryGetInt( "n_head" ) )
                num_heads_ = static_cast<dim_t>(*nh2);

            // canonical max seq len, with common fallbacks
            if ( auto msl = meta.tryGetInt( "max_seq_len" ) )
                max_seq_len_ = static_cast<dim_t>(*msl);
            else if ( auto msl2 = meta.tryGetInt( "max_seq_length" ) )
                max_seq_len_ = static_cast<dim_t>(*msl2);
            else if ( auto msl3 = meta.tryGetInt( "max_position_embeddings" ) )
                max_seq_len_ = static_cast<dim_t>(*msl3);
            else if ( auto msl4 = meta.tryGetInt( "n_positions" ) )
                max_seq_len_ = static_cast<dim_t>(*msl4);

            // hidden size fallbacks
            if ( auto hs = meta.tryGetInt( "hidden_dim" ) )
                hidden_size_ = static_cast<dim_t>(*hs);
            else if ( auto hs2 = meta.tryGetInt( "n_inner" ) )
                hidden_size_ = static_cast<dim_t>(*hs2);

            if ( hidden_size_ == 0 && embedding_size_ > 0 )
            {
                hidden_size_ = embedding_size_ * 4;
            }

            // boolean field
            if ( auto ub = meta.tryGetBool( "use_bias" ) )
                use_bias_ = *ub;
            else if ( auto hb = meta.tryGetBool( "has_bias" ) )
                use_bias_ = *hb;
        }

        std::string toString() const override
        {
            // REVIEW: Should use getter names
            std::ostringstream oss;

            oss << "GptConfig( embedding size=" << embedding_size_
                << ", layers=" << num_layers_
                << ", heads=" << num_heads_
                << ", vocab=" << vocab_size_
                << ", max_seq_len=" << max_seq_len_
                << ", hidden_dim=" << hidden_size_
                << ", bias=" << (use_bias_ ? "yes" : "no") << ")";

            return oss.str();
        }

    private:

        dim_t embedding_size_ = 768;      // GPT-2 small default
        dim_t num_layers_ = 12;           // GPT-2 small default
        dim_t hidden_size_ = 768;         // Should match embedding_size_
        dim_t vocab_size_ = 50257;        // GPT-2 BPE vocab size
        dim_t num_heads_ = 12;            // GPT-2 small default
        dim_t max_seq_len_ = 1024;
        bool use_bias_ = true;
    };
}