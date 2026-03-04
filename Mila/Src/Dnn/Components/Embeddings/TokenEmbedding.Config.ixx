/**
 * @file TokenEmbeddingConfig.ixx
 * @brief Configuration for the TokenEmbedding component.
 *
 * Derived from LpeConfig with all positional embedding fields removed.
 * TokenEmbedding is a pure vocabulary lookup — sequence position is
 * handled downstream by a dedicated encoding component (RoPE, ALiBi,
 * or Learned).
 */

module;
#include <stdexcept>
#include <string>
#include <utility>
#include <sstream>

export module Dnn.Components.TokenEmbedding:Config;

import Dnn.ComponentConfig;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    /**
     * @brief Configuration for the TokenEmbedding component.
     *
     * Provides a type-safe fluent interface for configuring a pure token
     * embedding lookup. Positional fields are intentionally absent —
     * they belong to the model or attention configuration.
     */
    export class TokenEmbeddingConfig : public ComponentConfig
    {
    public:

        template <typename Self>
        decltype(auto) withVocabSize( this Self&& self, size_t vocab_size )
        {
            self.vocab_size_ = vocab_size;
            return std::forward<Self>( self );
        }

        template <typename Self>
        decltype(auto) withEmbeddingDim( this Self&& self, size_t embedding_dim )
        {
            self.embedding_dim_ = embedding_dim;
            return std::forward<Self>( self );
        }

        size_t getVocabSize() const
        {
            return vocab_size_;
        }

        size_t getEmbeddingDim() const
        {
            return embedding_dim_;
        }

        void validate() const override
        {
            if ( vocab_size_ == 0 )
                throw std::invalid_argument( "TokenEmbeddingConfig: vocab_size must be > 0" );

            if ( embedding_dim_ == 0 )
                throw std::invalid_argument( "TokenEmbeddingConfig: embedding_dim must be > 0" );

            if ( embedding_dim_ % 4 != 0 )
                throw std::invalid_argument( "TokenEmbeddingConfig: embedding_dim must be "
                    "divisible by 4 (float4 vectorization)" );
        }

        SerializationMetadata toMetadata() const
        {
            SerializationMetadata meta;
            meta.set( "precision", static_cast<int64_t>(precision_) )
                .set( "vocab_size", static_cast<int64_t>(vocab_size_) )
                .set( "embedding_dim", static_cast<int64_t>(embedding_dim_) );

            return meta;
        }

        void fromMetadata( const SerializationMetadata& meta )
        {
            if ( auto v = meta.tryGetInt( "precision" ) )
                precision_ = static_cast<decltype(precision_)>(*v);

            if ( auto v = meta.tryGetInt( "vocab_size" ) )
                vocab_size_ = static_cast<size_t>(*v);

            if ( auto v = meta.tryGetInt( "embedding_dim" ) )
                embedding_dim_ = static_cast<size_t>(*v);
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "vocab_size=" << vocab_size_
                << ", embedding_dim=" << embedding_dim_;
            return oss.str();
        }

    private:
        size_t vocab_size_{ 0 };
        size_t embedding_dim_{ 0 };
    };
}