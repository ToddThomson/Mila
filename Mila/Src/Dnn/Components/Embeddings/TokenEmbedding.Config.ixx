/**
 * @file TokenEmbedding.Config.ixx
 * @brief Configuration for the TokenEmbedding component.
 *
 * Derived from LpeConfig with all positional embedding fields removed.
 * TokenEmbedding is a pure vocabulary lookup -- sequence position is
 * handled downstream by a dedicated encoding component (RoPE, ALiBi,
 * or Learned).
 */

module;
#include <stdexcept>
#include <string>
#include <utility>
#include <sstream>
#include <cmath>

export module Dnn.Components.TokenEmbeddingConfig;

import Dnn.ComponentConfig;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    /**
     * @brief Configuration for the TokenEmbedding component.
     *
     * Provides a type-safe fluent interface for configuring a pure token
     * embedding lookup. Positional fields are intentionally absent --
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

        // Scalar applied to the embedding output in forward (default 1.0 = identity).
        // Gemma sets this to sqrt(embedding_dim) so the embedding table can be stored
        // raw and shared with a tied lm_head; see WeightTying.md D5.
        template <typename Self>
        decltype(auto) withEmbeddingScale( this Self&& self, float embedding_scale )
        {
            self.embedding_scale_ = embedding_scale;
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

        float getEmbeddingScale() const noexcept
        {
            return embedding_scale_;
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

            if ( !std::isfinite( embedding_scale_ ) || embedding_scale_ <= 0.0f )
                throw std::invalid_argument( "TokenEmbeddingConfig: embedding_scale must be "
                    "finite and > 0" );
        }

        SerializationMetadata toMetadata() const override
        {
            SerializationMetadata meta;

            meta.set( "vocab_size", static_cast<int64_t>(vocab_size_) )
                .set( "embedding_dim", static_cast<int64_t>(embedding_dim_) )
                .set( "embedding_scale", embedding_scale_ );

            return meta;
        }

        void fromMetadata( const SerializationMetadata& meta ) override
        {
            if ( auto v = meta.tryGetInt( "vocab_size" ) )
                vocab_size_ = static_cast<size_t>(*v);

            if ( auto v = meta.tryGetInt( "embedding_dim" ) )
                embedding_dim_ = static_cast<size_t>(*v);

            if ( auto v = meta.tryGetFloat( "embedding_scale" ) )
                embedding_scale_ = *v;
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "vocab_size=" << vocab_size_
                << ", embedding_dim=" << embedding_dim_
                << ", embedding_scale=" << embedding_scale_;
            return oss.str();
        }

    private:
        size_t vocab_size_{ 0 };
        size_t embedding_dim_{ 0 };
        float embedding_scale_{ 1.0f };
    };
}