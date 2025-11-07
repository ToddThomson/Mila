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

export module CharLM.Vocabulary;

namespace Mila::CharLM
{
    /**
     * @brief Character vocabulary for tokenization.
     *
     * Manages bidirectional mappings between characters and token indices.
     * Supports serialization to disk for preprocessing pipelines.
     *
     * Features:
     * - Deterministic vocabulary ordering (sorted by character code)
     * - Special token support (padding, unknown, etc.)
     * - Save/load functionality for caching
     * - Fast lookups via hash maps
     */
    export class CharVocabulary
    {
    public:
        /**
         * @brief Constructs empty vocabulary.
         */
        CharVocabulary() = default;

        /**
         * @brief Builds vocabulary from text corpus.
         *
         * Extracts unique characters and creates sorted, deterministic mappings.
         * Optionally adds special tokens for padding, unknown characters, etc.
         *
         * @param text Source text for vocabulary extraction
         * @param add_special_tokens Whether to add special tokens (pad, unk)
         * @return Number of tokens in vocabulary
         */
        size_t buildFromText( const std::string& text, bool add_special_tokens = false )
        {
            char_to_idx_.clear();
            idx_to_char_.clear();

            // Collect unique characters
            std::unordered_map<char, bool> unique_chars;
            for (char c : text)
            {
                unique_chars[c] = true;
            }

            // Sort for deterministic ordering
            std::vector<char> sorted_chars;
            sorted_chars.reserve( unique_chars.size() );
            for (const auto& [c, _] : unique_chars)
            {
                sorted_chars.push_back( c );
            }
            std::sort( sorted_chars.begin(), sorted_chars.end() );

            // Add special tokens at the beginning if requested
            int idx = 0;
            if (add_special_tokens)
            {
                // Reserve indices for special tokens
                pad_token_id_ = idx++;
                unk_token_id_ = idx++;
                
                idx_to_char_.push_back( '\0' );  // PAD token
                idx_to_char_.push_back( '\1' );  // UNK token
                
                char_to_idx_['\0'] = pad_token_id_;
                char_to_idx_['\1'] = unk_token_id_;
            }

            // Add regular characters
            for (char c : sorted_chars)
            {
                if (!add_special_tokens || (c != '\0' && c != '\1'))
                {
                    idx_to_char_.push_back( c );
                    char_to_idx_[c] = idx++;
                }
            }

            return idx_to_char_.size();
        }

        /**
         * @brief Saves vocabulary to file.
         *
         * Format:
         *   Line 1: vocab_size
         *   Line 2: has_special_tokens (0 or 1)
         *   Lines 3+: character_code (one per line)
         *
         * @param filename Path to output vocabulary file
         */
        void save( const std::string& filename ) const
        {
            std::ofstream file( filename, std::ios::binary );
            if (!file)
            {
                throw std::runtime_error( "Cannot open vocabulary file for writing: " + filename );
            }

            // Write header
            size_t vocab_size = idx_to_char_.size();
            file.write( reinterpret_cast<const char*>(&vocab_size), sizeof( vocab_size ) );

            bool has_special = (pad_token_id_ >= 0);
            file.write( reinterpret_cast<const char*>(&has_special), sizeof( has_special ) );

            if (has_special)
            {
                file.write( reinterpret_cast<const char*>(&pad_token_id_), sizeof( pad_token_id_ ) );
                file.write( reinterpret_cast<const char*>(&unk_token_id_), sizeof( unk_token_id_ ) );
            }

            // Write characters
            for (char c : idx_to_char_)
            {
                file.write( &c, sizeof( char ) );
            }

            if (!file)
            {
                throw std::runtime_error( "Error writing vocabulary file: " + filename );
            }
        }

        /**
         * @brief Loads vocabulary from file.
         *
         * @param filename Path to vocabulary file
         * @return Number of tokens loaded
         */
        size_t load( const std::string& filename )
        {
            std::ifstream file( filename, std::ios::binary );
            if (!file)
            {
                throw std::runtime_error( "Cannot open vocabulary file: " + filename );
            }

            char_to_idx_.clear();
            idx_to_char_.clear();

            // Read header
            size_t vocab_size;
            file.read( reinterpret_cast<char*>(&vocab_size), sizeof( vocab_size ) );

            bool has_special;
            file.read( reinterpret_cast<char*>(&has_special), sizeof( has_special ) );

            if (has_special)
            {
                file.read( reinterpret_cast<char*>(&pad_token_id_), sizeof( pad_token_id_ ) );
                file.read( reinterpret_cast<char*>(&unk_token_id_), sizeof( unk_token_id_ ) );
            }
            else
            {
                pad_token_id_ = -1;
                unk_token_id_ = -1;
            }

            // Read characters
            idx_to_char_.resize( vocab_size );
            for (size_t i = 0; i < vocab_size; ++i)
            {
                char c;
                file.read( &c, sizeof( char ) );
                idx_to_char_[i] = c;
                char_to_idx_[c] = static_cast<int>( i );
            }

            if (!file)
            {
                throw std::runtime_error( "Error reading vocabulary file: " + filename );
            }

            return vocab_size;
        }

        /**
         * @brief Converts character to token index.
         *
         * @param c Character to convert
         * @return Token index, or unk_token_id if character not in vocabulary
         */
        int charToIndex( char c ) const
        {
            auto it = char_to_idx_.find( c );
            if (it != char_to_idx_.end())
            {
                return it->second;
            }
            return unk_token_id_ >= 0 ? unk_token_id_ : 0;
        }

        /**
         * @brief Converts token index to character.
         *
         * @param idx Token index
         * @return Character, or '?' if index out of range
         */
        char indexToChar( int idx ) const
        {
            if (idx >= 0 && idx < static_cast<int>( idx_to_char_.size() ))
            {
                return idx_to_char_[idx];
            }
            return '?';
        }

        /**
         * @brief Gets vocabulary size.
         */
        size_t size() const
        {
            return idx_to_char_.size();
        }

        /**
         * @brief Gets padding token ID.
         */
        int padTokenId() const
        {
            return pad_token_id_;
        }

        /**
         * @brief Gets unknown token ID.
         */
        int unkTokenId() const
        {
            return unk_token_id_;
        }

        /**
         * @brief Checks if vocabulary has special tokens.
         */
        bool hasSpecialTokens() const
        {
            return pad_token_id_ >= 0;
        }

    private:
        std::unordered_map<char, int> char_to_idx_;
        std::vector<char> idx_to_char_;
        int pad_token_id_{ -1 };
        int unk_token_id_{ -1 };
    };
}
