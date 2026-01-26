/**
 * @file CharVocabulary.ixx
 * @brief Character vocabulary management for language modeling.
 *
 * Provides character-to-index and index-to-character mappings with
 * serialization support for preprocessing pipelines.
 */

module;
#include <vector>
#include <unordered_map>
#include <string>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
#include <filesystem>
#include <optional>

export module Data.CharVocabulary;

import Data.TokenizerVocabulary;
import Data.SpecialTokens;

namespace Mila::Data
{
    using Mila::Dnn::Data::TokenizerVocabulary;
    namespace fs = std::filesystem;

    /**
     * @brief Character vocabulary for tokenization.
     *
     * Implements the generic TokenizerVocabulary interface so the character
     * vocabulary can be used anywhere a TokenizerVocabulary is required.
     *
     * Notes:
     * - Tokens are single bytes (char). Callers should document UTF-8 usage
     *   if they expect multi-byte characters; this implementation operates on
     *   raw bytes and normalizes CRLF to LF during buildFromText().
     * - Special token ids (pad/unk) are optional; when present their ids are
     *   returned by tokenToId() for unknown tokens if configured.
     */
    export class CharVocabulary : public TokenizerVocabulary
    {
    public:

        /**
         * @brief Load vocabulary state from disk.
         *
         * Replaces the in-memory state with the contents of the file.
         * Throws std::runtime_error on I/O or format errors.
         *
         * @param path Filesystem path to read the vocabulary from.
         * @return Number of tokens loaded.
         */
        static CharVocabulary load( const std::filesystem::path& path )
        {
            std::ifstream file( path.string(), std::ios::binary );

            if ( !file )
            {
                throw std::runtime_error( "Cannot open vocabulary file: " + path.string() );
            }

            CharVocabulary vocab;

            vocab.char_to_idx_.clear();
            vocab.idx_to_char_.clear();

            size_t vocab_size;
            file.read( reinterpret_cast<char*>(&vocab_size), sizeof( vocab_size ) );

            bool has_special;
            file.read( reinterpret_cast<char*>(&has_special), sizeof( has_special ) );

            if ( has_special )
            {
                file.read( reinterpret_cast<char*>(&vocab.pad_token_id_), sizeof( vocab.pad_token_id_ ) );
                file.read( reinterpret_cast<char*>(&vocab.unk_token_id_), sizeof( vocab.unk_token_id_ ) );
            }
            else
            {
                vocab.pad_token_id_ = -1;
                vocab.unk_token_id_ = -1;
            }

            vocab.idx_to_char_.resize( vocab_size );
            for ( size_t i = 0; i < vocab_size; ++i )
            {
                char c;
                file.read( &c, sizeof( char ) );
                vocab.idx_to_char_[ i ] = c;
                vocab.char_to_idx_[ c ] = static_cast<int>( i );
            }
        }

        /**
         * @brief Builds vocabulary from text corpus.
         *
         * Extracts unique bytes and creates sorted, deterministic mappings.
         * Optionally adds special tokens for padding and unknown values.
         *
         * @param text Source text for vocabulary extraction.
         * @param add_special_tokens Whether to include PAD and UNK at the start.
         * @return Number of tokens in vocabulary after build.
         */
        size_t buildFromText(
            const std::string& text,
            const SpecialTokens& special_tokens = SpecialTokens{},
            bool case_sensitive = true,
            bool normalize_unicode = false,
            bool byte_level = false )
        {
            char_to_idx_.clear();
            idx_to_char_.clear();

            pad_token_id_ = -1;
            unk_token_id_ = -1;

            std::string norm;
            norm.reserve( text.size() );

            for ( size_t i = 0; i < text.size(); ++i )
            {
                char c = text[ i ];

                if ( c == '\r' )
                {
                    if ( i + 1 < text.size() && text[ i + 1 ] == '\n' )
                    {
                        continue;
                    }

                    norm.push_back( '\n' );
                }
                else
                {
                    norm.push_back( c );
                }
            }

            std::unordered_map<unsigned char, bool> unique_bytes;
            for ( char c : norm )
            {
                unique_bytes[ static_cast<unsigned char>( c ) ] = true;
            }

            std::vector<unsigned char> sorted_bytes;
            sorted_bytes.reserve( unique_bytes.size() );

            for ( const auto &kv : unique_bytes )
            {
                sorted_bytes.push_back( kv.first );
            }

            std::sort( sorted_bytes.begin(), sorted_bytes.end() );

            int idx = 0;

            if ( special_tokens.enabled )
            {
                pad_token_id_ = idx++;
                unk_token_id_ = idx++;

                idx_to_char_.push_back( '\0' );  // PAD token
                idx_to_char_.push_back( '\1' );  // UNK token

                char_to_idx_['\0'] = pad_token_id_;
                char_to_idx_['\1'] = unk_token_id_;
            }

            for ( unsigned char ub : sorted_bytes )
            {
                char c = static_cast<char>( ub );

                if ( !special_tokens.enabled || ( c != '\0' && c != '\1' ) )
                {
                    idx_to_char_.push_back( c );
                    char_to_idx_[ c ] = idx++;
                }
            }

            return idx_to_char_.size();
        }

        /**
         * @brief Serialize the vocabulary to disk.
         *
         * Produces the same binary format as the previous string-based save().
         * Throws std::runtime_error on I/O errors.
         *
         * @param path Filesystem path to write the vocabulary to.
         */
        void save( const std::filesystem::path& path ) const override
        {
            std::ofstream file( path.string(), std::ios::binary );
            if (!file)
            {
                throw std::runtime_error( "Cannot open vocabulary file for writing: " + path.string() );
            }

            size_t vocab_size = idx_to_char_.size();
            file.write( reinterpret_cast<const char*>(&vocab_size), sizeof( vocab_size ) );

            bool has_special = (pad_token_id_ >= 0);
            file.write( reinterpret_cast<const char*>(&has_special), sizeof( has_special ) );

            if (has_special)
            {
                file.write( reinterpret_cast<const char*>(&pad_token_id_), sizeof( pad_token_id_ ) );
                file.write( reinterpret_cast<const char*>(&unk_token_id_), sizeof( unk_token_id_ ) );
            }

            for (char c : idx_to_char_)
            {
                file.write( &c, sizeof( char ) );
            }

            if (!file)
            {
                throw std::runtime_error( "Error writing vocabulary file: " + path.string() );
            }
        }

        

        /**
         * @brief Returns vocabulary size.
         *
         * Implements TokenizerVocabulary::getSize().
         *
         * @return Number of tokens.
         */
        size_t getSize() const override
        {
            return idx_to_char_.size();
        }

        /**
         * @brief Map a token string to its numeric id.
         *
         * For character vocabulary the token string is interpreted by its first
         * byte. If the token is not present and an unknown token id is configured
         * the unknown id is returned. If no unknown token exists, an empty optional
         * is returned.
         *
         * @param token Token string to look up.
         * @return optional id if available, or empty optional if not found and no UNK.
         */
        std::optional<uint32_t> tokenToId( const std::string& token ) const override
        {
            if ( token.empty() )
            {
                return std::nullopt;
            }

            unsigned char c = static_cast<unsigned char>( token[0] );
            auto it = char_to_idx_.find( static_cast<char>( c ) );

            if ( it != char_to_idx_.end() )
            {
                return static_cast<uint32_t>( it->second );
            }

            if ( unk_token_id_ >= 0 )
            {
                return static_cast<uint32_t>( unk_token_id_ );
            }

            return std::nullopt;
        }

        /**
         * @brief Map a numeric id back to its token string.
         *
         * Returns empty optional if id is out of range.
         *
         * @param id Token id to convert.
         * @return optional token string.
         */
        std::optional<std::string> idToToken( uint32_t id ) const override
        {
            if ( id < idx_to_char_.size() )
            {
                return std::string( 1, idx_to_char_[ static_cast<size_t>( id ) ] );
            }
            return std::nullopt;
        }

        /**
         * @name Backwards-compatible char-level API
         *
         * These helpers preserve the existing char-specific API used elsewhere
         * in the codebase.
         */
        ///@{
        int charToIndex( char c ) const
        {
            auto it = char_to_idx_.find( c );
            if ( it != char_to_idx_.end() )
            {
                return it->second;
            }
            return unk_token_id_ >= 0 ? unk_token_id_ : 0;
        }

        char indexToChar( int idx ) const
        {
            if ( idx >= 0 && idx < static_cast<int>( idx_to_char_.size() ) )
            {
                return idx_to_char_[ static_cast<size_t>( idx ) ];
            }
            return '?';
        }

        size_t size() const
        {
            return getSize();
        }

        int padTokenId() const
        {
            return pad_token_id_;
        }

        int unkTokenId() const
        {
            return unk_token_id_;
        }

        bool hasSpecialTokens() const
        {
            return pad_token_id_ >= 0;
        }
        ///@}

    private:

        friend class CharTrainer;
        friend class CharTokenizer;

        CharVocabulary() = default;

        std::unordered_map<char, int> char_to_idx_;
        std::vector<char> idx_to_char_;
        int pad_token_id_{ -1 };
        int unk_token_id_{ -1 };
    };
}