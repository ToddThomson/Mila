/**
 * @file NgramEmbedding.Config.ixx
 * @brief Configuration for the hashed n-gram embedding: n-gram orders, heads, the hash constants and the table.
 *
 * Qwen 4's per-layer embedding reads one of these (Specifications/Qwen4.md section 3.4). The hash constants ship
 * in the checkpoint and arrive through the model's metadata; nothing here derives them.
 */

module;
#include <cstdint>
#include <format>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

export module Dnn.Components.NgramEmbeddingConfig;

import Dnn.TensorTypes;
import Dnn.ComponentConfig;
import Serialization.Metadata;

namespace Mila::Dnn
{
    using Serialization::SerializationMetadata;

    export class NgramEmbeddingConfig : public ComponentConfig
    {
    public:
        /**
         * @param ngram_size       The largest order: orders 2 .. ngram_size are hashed (Qwen 4: 3).
         * @param heads_per_ngram  Independently hashed heads per order (Qwen 4: 8).
         * @param embedding_dim    The concatenated output width, heads * row width (Qwen 4: ple_embed_dim).
         * @param eos_token_id     Pads the history and ends a segment: no n-gram reaches back across it.
         */
        NgramEmbeddingConfig( dim_t ngram_size, dim_t heads_per_ngram, dim_t embedding_dim, int64_t eos_token_id )
            : ngram_size_( ngram_size ), heads_per_ngram_( heads_per_ngram ), embedding_dim_( embedding_dim ),
              eos_token_id_( eos_token_id )
        {
        }

        /**
         * @brief The checkpoint's hash constants.
         *
         * @param multipliers      One per position of the largest n-gram, odd int64 values.
         * @param head_vocab_sizes One prime per head, orders ascending, heads within an order ascending.
         * @param head_offsets     Each head's first row in the shared table: the running sum of the sizes.
         * @param table_rows       The table's row count, the sizes' sum padded up by the checkpoint.
         */
        template<typename Self>
        decltype(auto) withHashConstants( this Self&& self, std::vector<int64_t> multipliers,
            std::vector<int64_t> head_vocab_sizes, std::vector<int64_t> head_offsets, dim_t table_rows )
        {
            self.multipliers_ = std::move( multipliers );
            self.head_vocab_sizes_ = std::move( head_vocab_sizes );
            self.head_offsets_ = std::move( head_offsets );
            self.table_rows_ = table_rows;

            return std::forward<Self>( self );
        }

        dim_t getNgramSize() const noexcept { return ngram_size_; }
        dim_t getHeadsPerNgram() const noexcept { return heads_per_ngram_; }
        dim_t getEmbeddingDim() const noexcept { return embedding_dim_; }
        int64_t getEosTokenId() const noexcept { return eos_token_id_; }
        dim_t getTableRows() const noexcept { return table_rows_; }

        const std::vector<int64_t>& getMultipliers() const noexcept { return multipliers_; }
        const std::vector<int64_t>& getHeadVocabSizes() const noexcept { return head_vocab_sizes_; }
        const std::vector<int64_t>& getHeadOffsets() const noexcept { return head_offsets_; }

        /// Every head of every order: ( ngram_size - 1 ) * heads_per_ngram.
        dim_t getHeads() const noexcept { return ( ngram_size_ - 1 ) * heads_per_ngram_; }

        /// One table row: embedding_dim / heads.
        dim_t getRowWidth() const noexcept { return embedding_dim_ / getHeads(); }

        /// Token ids the hash looks back over: ngram_size - 1.
        dim_t getHistoryLength() const noexcept { return ngram_size_ - 1; }

        void validate() const override
        {
            if ( ngram_size_ < 2 )
            {
                throw std::invalid_argument( "NgramEmbeddingConfig: ngram_size must be >= 2" );
            }

            if ( heads_per_ngram_ <= 0 || embedding_dim_ <= 0 || embedding_dim_ % getHeads() != 0 )
            {
                throw std::invalid_argument( std::format(
                    "NgramEmbeddingConfig: embedding_dim {} must be a positive multiple of the {} heads", embedding_dim_, getHeads() ) );
            }

            if ( eos_token_id_ < 0 )
            {
                throw std::invalid_argument( "NgramEmbeddingConfig: eos_token_id must be >= 0" );
            }

            if ( static_cast<dim_t>( multipliers_.size() ) != ngram_size_ )
            {
                throw std::invalid_argument( std::format(
                    "NgramEmbeddingConfig: {} multipliers for n-gram size {}", multipliers_.size(), ngram_size_ ) );
            }

            if ( static_cast<dim_t>( head_vocab_sizes_.size() ) != getHeads() || static_cast<dim_t>( head_offsets_.size() ) != getHeads() )
            {
                throw std::invalid_argument( std::format(
                    "NgramEmbeddingConfig: the head constants must have {} entries", getHeads() ) );
            }

            int64_t running = 0;

            for ( size_t head = 0; head < head_vocab_sizes_.size(); ++head )
            {
                if ( head_vocab_sizes_[ head ] <= 0 || head_offsets_[ head ] != running )
                {
                    throw std::invalid_argument( std::format(
                        "NgramEmbeddingConfig: head {} is not laid out after the heads before it", head ) );
                }

                running += head_vocab_sizes_[ head ];
            }

            if ( table_rows_ < running )
            {
                throw std::invalid_argument( std::format(
                    "NgramEmbeddingConfig: {} table rows cannot hold the heads' {}", table_rows_, running ) );
            }
        }

        SerializationMetadata toMetadata() const override
        {
            SerializationMetadata meta;

            // Decimal strings: a multiplier can exceed what a double holds exactly.
            meta.set( "ngram_size", static_cast<int64_t>( ngram_size_ ) )
                .set( "heads_per_ngram", static_cast<int64_t>( heads_per_ngram_ ) )
                .set( "embedding_dim", static_cast<int64_t>( embedding_dim_ ) )
                .set( "eos_token_id", eos_token_id_ )
                .set( "table_rows", static_cast<int64_t>( table_rows_ ) )
                .set( "multipliers", joinIntegers( multipliers_ ) )
                .set( "head_vocab_sizes", joinIntegers( head_vocab_sizes_ ) )
                .set( "head_offsets", joinIntegers( head_offsets_ ) );

            return meta;
        }

        void fromMetadata( const SerializationMetadata& meta ) override
        {
            if ( auto v = meta.tryGetInt( "ngram_size" ) )
            {
                ngram_size_ = static_cast<dim_t>( *v );
            }

            if ( auto v = meta.tryGetInt( "heads_per_ngram" ) )
            {
                heads_per_ngram_ = static_cast<dim_t>( *v );
            }

            if ( auto v = meta.tryGetInt( "embedding_dim" ) )
            {
                embedding_dim_ = static_cast<dim_t>( *v );
            }

            if ( auto v = meta.tryGetInt( "eos_token_id" ) )
            {
                eos_token_id_ = *v;
            }

            if ( auto v = meta.tryGetInt( "table_rows" ) )
            {
                table_rows_ = static_cast<dim_t>( *v );
            }

            if ( auto v = meta.tryGetString( "multipliers" ) )
            {
                multipliers_ = splitIntegers( *v );
            }

            if ( auto v = meta.tryGetString( "head_vocab_sizes" ) )
            {
                head_vocab_sizes_ = splitIntegers( *v );
            }

            if ( auto v = meta.tryGetString( "head_offsets" ) )
            {
                head_offsets_ = splitIntegers( *v );
            }
        }

        std::string toString() const override
        {
            std::ostringstream oss;
            oss << "NgramEmbeddingConfig( ngram_size=" << ngram_size_ << ", heads_per_ngram=" << heads_per_ngram_
                << ", embedding_dim=" << embedding_dim_ << ", eos_token_id=" << eos_token_id_
                << ", table_rows=" << table_rows_ << " )";

            return oss.str();
        }

        /// "a,b,c" -> { a, b, c }, the converter's encoding of an int64 list.
        static std::vector<int64_t> splitIntegers( const std::string& text )
        {
            std::vector<int64_t> values;
            std::istringstream stream( text );
            std::string item;

            while ( std::getline( stream, item, ',' ) )
            {
                if ( !item.empty() )
                {
                    values.push_back( std::stoll( item ) );
                }
            }

            return values;
        }

    private:
        dim_t ngram_size_{ 0 };
        dim_t heads_per_ngram_{ 0 };
        dim_t embedding_dim_{ 0 };
        int64_t eos_token_id_{ -1 };
        dim_t table_rows_{ 0 };

        std::vector<int64_t> multipliers_{};
        std::vector<int64_t> head_vocab_sizes_{};
        std::vector<int64_t> head_offsets_{};

        static std::string joinIntegers( const std::vector<int64_t>& values )
        {
            std::string text;

            for ( size_t i = 0; i < values.size(); ++i )
            {
                if ( i > 0 )
                {
                    text += ",";
                }

                text += std::to_string( values[ i ] );
            }

            return text;
        }
    };
}
