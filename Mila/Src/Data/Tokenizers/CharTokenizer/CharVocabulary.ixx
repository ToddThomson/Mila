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
#include <cctype>

export module Data.CharVocabulary;

import Data.TokenizerVocabulary;
import Data.CharTrainerConfig;

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

            return vocab;
        }

        /**
         * @brief Builds vocabulary from text corpus.
         *
         * Extracts unique bytes and creates sorted, deterministic mappings.
         * Respects the training configuration provided by `CharTrainerConfig`.
         *
         * Behavior details:
         * - CRLF sequences ("\r\n") are normalized to a single '\n'.
         * - If `config.isCaseSensitive()` is false input bytes are lower-cased
         *   using `std::tolower` on each byte (this is byte-level lowercasing).
         * - Special tokens are reserved in a deterministic order:
         *     PAD -> '\0', UNK -> '\1', BOS -> '\2', EOS -> '\3', MASK -> '\4'
         *   Only PAD and UNK are surfaced by this class via `padTokenId()` and
         *   `unkTokenId()`. Other reserved entries are present in the index
         *   table for deterministic ordering but do not have dedicated getters.
         *
         * @param text Source text for vocabulary extraction.
         * @param config Trainer configuration that describes special token usage
         *               and other flags (case sensitivity, byte-level, etc.)
         * @return Number of tokens in vocabulary after build.
         */
        size_t buildFromText(
            const std::string& text,
            const CharTrainerConfig& config )
        {
            char_to_idx_.clear();
            idx_to_char_.clear();

            pad_token_id_ = -1;
            unk_token_id_ = -1;

            std::string norm;
            norm.reserve( text.size() );

            // Normalize CRLF -> LF and optionally lowercase depending on config
            if ( config.isCaseSensitive() )
            {
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
            }
            else
            {
                for ( size_t i = 0; i < text.size(); ++i )
                {
                    unsigned char cu = static_cast<unsigned char>( text[ i ] );

                    if ( cu == '\r' )
                    {
                        if ( i + 1 < text.size() && static_cast<unsigned char>( text[ i + 1 ] ) == '\n' )
                        {
                            continue;
                        }

                        norm.push_back( '\n' );
                    }
                    else
                    {
                        // Lowercase the byte (byte-level lowercasing)
                        char lowered = static_cast<char>( std::tolower( cu ) );
                        norm.push_back( lowered );
                    }
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

            // Reserve special tokens in deterministic order if requested by config.
            // We map them to low-valued control bytes so they do not collide with
            // printable characters when sorting the remainder.
            std::vector<unsigned char> reserved_bytes;

            const CharSpecialTokens& st = config.getSpecialTokens();

            if ( st.use_pad )
            {
                pad_token_id_ = idx++;
                idx_to_char_.push_back( '\0' );  // PAD token placeholder byte
                char_to_idx_['\0'] = pad_token_id_;
                reserved_bytes.push_back( static_cast<unsigned char>('\0') );
            }

            if ( st.use_unk )
            {
                unk_token_id_ = idx++;
                idx_to_char_.push_back( '\1' );  // UNK token placeholder byte
                char_to_idx_['\1'] = unk_token_id_;
                reserved_bytes.push_back( static_cast<unsigned char>('\1') );
            }

            if ( st.use_bos )
            {
                idx_to_char_.push_back( '\2' );  // BOS placeholder byte
                char_to_idx_['\2'] = idx++;
                reserved_bytes.push_back( static_cast<unsigned char>('\2') );
            }

            if ( st.use_eos )
            {
                idx_to_char_.push_back( '\3' );  // EOS placeholder byte
                char_to_idx_['\3'] = idx++;
                reserved_bytes.push_back( static_cast<unsigned char>('\3') );
            }

            if ( st.use_mask )
            {
                idx_to_char_.push_back( '\4' );  // MASK placeholder byte
                char_to_idx_['\4'] = idx++;
                reserved_bytes.push_back( static_cast<unsigned char>('\4') );
            }

            // Append all sorted bytes that are not reserved by special tokens.
            for ( unsigned char ub : sorted_bytes )
            {
                char c = static_cast<char>( ub );

                bool is_reserved = false;
                for ( unsigned char rb : reserved_bytes )
                {
                    if ( rb == ub )
                    {
                        is_reserved = true;
                        break;
                    }
                }

                if ( !is_reserved )
                {
                    idx_to_char_.push_back( c );
                    char_to_idx_[ c ] = idx++;
                }
            }

            // Final vocabulary size
            // (leave a blank line before return per coding policy)
            
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