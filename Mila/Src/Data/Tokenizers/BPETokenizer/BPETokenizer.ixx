/*!
 * \file
 * \brief BPE tokenizer implementing the Tokenizer API.
 *
 * Greedy longest-match tokenizer that uses a `TokenizerVocabulary`
 * containing multi-byte token strings (BPE pieces).
 */

module;
#include <string>
#include <string_view>
#include <vector>
#include <span>
#include <memory>
#include <optional>
#include <unordered_map>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include <iostream>
#include <regex>
#include <limits>

export module Data.BpeTokenizer;
import Data.BpeVocabulary;
import Data.Tokenizer;
import Data.TokenizerVocabulary;

namespace Mila::Data
{
    using Mila::Dnn::Data::TokenId;
    using Mila::Dnn::Data::Tokenizer;
    using Mila::Dnn::Data::TokenizerVocabulary;

    //// GPT-2 pre-tokenization patterns
    //constexpr const char* GPT2_PRETOKENIZATION_PATTERN =
    //    R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";

    //constexpr const char* GPT2_PRETOKENIZATION_PATTERN_ASCII_FALLBACK =
    //    R"('s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+)";

    /**
     * @brief Byte-Pair Encoding (BPE) style tokenizer.
     *
     * Implements a BPE tokenizer with optional pre-tokenization (e.g., GPT-2 style).
     * Supports both byte-level and character-level encoding.
     *
     * Encoding strategy:
     *  1. Pre-tokenize text using regex pattern (if specified)
     *  2. Convert each word to byte/character tokens
     *  3. Apply BPE merges greedily (lowest priority first)
     *  4. Convert final tokens to IDs
     *
     * Decoding strategy:
     *  - Concatenate token strings and decode bytes back to text
     */
    export class BpeTokenizer : public Tokenizer
    {
    public:

        explicit BpeTokenizer( BpeVocabulary vocab )
            : vocab_( std::move( vocab ) )
        {
            // Initialize regex if pre-tokenization pattern is specified
            initializePreTokenization();
        }

        // ========================================================================
        // Convenience Methods - Shortcuts for common workflows
        // ========================================================================

        // Load custom-trained vocabulary and create tokenizer in one step
        static BpeTokenizer load( const std::filesystem::path& path )
        {
            return BpeTokenizer( BpeVocabulary::load( path ) );
        }

        // Load pre-trained models and create tokenizer in one step
        static std::shared_ptr<BpeTokenizer> loadGpt2( const std::filesystem::path& vocab_path )
        {
            return std::make_shared<BpeTokenizer>( BpeVocabulary::loadGpt2( vocab_path ) );
        }

        static BpeTokenizer loadLlama( const std::filesystem::path& modelPath )
        {
            return BpeTokenizer( BpeVocabulary::loadLlama( modelPath ) );
        }

        static BpeTokenizer loadMistral(
            const std::filesystem::path& vocab_path,
            const std::filesystem::path& mergesPath )
        {
            return BpeTokenizer( BpeVocabulary::loadMistral( vocab_path, mergesPath ) );
        }

        // ========================================================================
        // Tokenizer Interface Implementation
        // ========================================================================

        std::vector<TokenId> encode( const std::string& text ) override
        {
            auto start_time = std::chrono::high_resolution_clock::now();

            // Step 1: Pre-tokenize based on pattern (if any)
            std::vector<std::string> words = preTokenize( text );

            /*std::cout << "Pre-tokenized into " << words.size() << " words:\n";
            for ( size_t i = 0; i < words.size(); ++i )
            {
                std::cout << "  [" << i << "]: '" << words[ i ] << "'\n";
            }*/

            /*std::cout << "Tokenizing " << text.size() << " characters into "
                << words.size() << " word(s)...\n";*/

            std::vector<std::string> all_tokens;
            size_t pass = 0;

            // Step 2: Process each word separately
            for ( const auto& word : words )
            {
                // Convert word to byte/char tokens
                std::vector<std::string> tokens;

                if ( vocab_.isByteLevel() )
                {
                    const auto& byte_encoder = vocab_.getByteEncoder();
                    for ( unsigned char byte : word )
                    {
                        tokens.push_back( byte_encoder.at( byte ) );
                    }
                }
                else
                {
                    for ( char c : word )
                    {
                        tokens.push_back( std::string( 1, c ) );
                    }
                }

                // Step 3: Apply BPE merges to this word
                while ( tokens.size() > 1 )
                {
                    // Find the best merge (lowest priority value)
                    int best_idx = -1;
                    int best_priority = std::numeric_limits<int>::max();

                    for ( size_t i = 0; i < tokens.size() - 1; ++i )
                    {
                        auto priority = vocab_.getMergePriority( tokens[ i ], tokens[ i + 1 ] );
                        if ( priority && *priority < best_priority )
                        {
                            best_priority = *priority;
                            best_idx = static_cast<int>( i );
                        }
                    }

                    // No more merges available
                    if ( best_idx == -1 ) break;

                    // Apply the merge
                    tokens[ best_idx ] = tokens[ best_idx ] + tokens[ best_idx + 1 ];
                    tokens.erase( tokens.begin() + best_idx + 1 );
                }

                // Add this word's tokens to result
                all_tokens.insert( all_tokens.end(), tokens.begin(), tokens.end() );

                ++pass;
                if ( pass % 100 == 0 )
                {
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start_time
                    ).count();
                    std::cout << "\r[" << elapsed << "ms] Words: " << pass
                        << " | Tokens: " << all_tokens.size()
                        << "          " << std::flush;
                }
            }

            // Step 4: Convert tokens to IDs
            std::vector<TokenId> out;
            out.reserve( all_tokens.size() );
            for ( const auto& token : all_tokens )
            {
                auto id = vocab_.tokenToId( token );
                out.push_back( id ? *id : 0 );
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            std::cout << "\n Encoding completed in " << ms << "ms"
                << " (" << text.size() << " chars -> " << out.size() << " tokens)\n";

            return out;
        }

        std::string decode( std::span<const TokenId> tokens ) override
        {
            std::string out;
            out.reserve( tokens.size() * 4 );

            for ( TokenId id : tokens )
            {
                auto tok_opt = vocab_.idToToken( id );
                if ( tok_opt )
                {
                    decodeToken( *tok_opt, out );
                }
                else
                {
                    out.push_back( '?' );
                }
            }

            return out;
        }

        size_t getVocabSize() const override
        {
            return vocab_.getSize();
        }

        std::optional<TokenId> getBosTokenId() const override
        {
            return std::nullopt;
        }

        std::optional<TokenId> getEosTokenId() const override
        {
            return std::nullopt;
        }

        std::optional<TokenId> getPadTokenId() const override
        {
            return std::nullopt;
        }

        std::string tokenToString( TokenId tokenId ) const override
        {
            auto tok = vocab_.idToToken( tokenId );
            return tok ? *tok : std::string();
        }

        bool isValidToken( TokenId tokenId ) const override
        {
            return static_cast<bool>(vocab_.idToToken( tokenId ));
        }

        // Access to underlying vocabulary
        const BpeVocabulary& getVocab() const
        {
            return vocab_;
        }

    private:
        BpeVocabulary vocab_;
        // DEPRECATED: std::unordered_map<std::string, TokenId> token_map_;
        // DEPRECATED: size_t max_token_len_{ 0 };
        std::optional<std::regex> pre_tokenization_regex_;

        /**
         * @brief Initialize pre-tokenization regex from vocabulary config.
         */
        void initializePreTokenization()
        {
            const auto& pattern = vocab_.getConfig().getPreTokenizationPattern();
            if ( pattern.empty() )
            {
                return;  // No pre-tokenization
            }

            try
            {
                pre_tokenization_regex_ = std::regex( pattern );
            }
            catch ( const std::regex_error& )
            {
                // Try ASCII fallback for GPT-2 pattern
                if ( pattern == GPT2_PRETOKENIZATION_PATTERN )
                {
                    std::cerr << "Warning: Unicode regex not supported, using ASCII fallback for GPT-2\n";
                    pre_tokenization_regex_ = std::regex( GPT2_PRETOKENIZATION_PATTERN_ASCII_FALLBACK );
                }
                else
                {
                    throw std::runtime_error( "Invalid pre-tokenization pattern: " + pattern );
                }
            }
        }

        /**
         * @brief Pre-tokenize text using regex pattern (if configured).
         * @param text Input text
         * @return Vector of words/tokens to process separately
         */
        std::vector<std::string> preTokenize( const std::string& text )
        {
            if ( !pre_tokenization_regex_ )
            {
                // No pre-tokenization, return entire text as single unit
                return { text };
            }

            std::vector<std::string> words;
            std::sregex_iterator iter( text.begin(), text.end(), *pre_tokenization_regex_ );
            std::sregex_iterator end;

            for ( ; iter != end; ++iter )
            {
                words.push_back( iter->str() );
            }

            return words.empty() ? std::vector<std::string>{ text } : words;
        }

        /**
         * @brief Decode a single token string to output.
         * @param token Token string (potentially byte-encoded)
         * @param out Output string to append to
         */
        void decodeToken( const std::string& token, std::string& out )
        {
            if ( !vocab_.isByteLevel() )
            {
                // Character-level: direct concatenation
                out.append( token );
                return;
            }

            // Byte-level: decode escaped bytes
            const auto& byte_decoder = vocab_.getByteDecoder();
            size_t i = 0;
            while ( i < token.size() )
            {
                size_t char_len = utf8CharLength( token[ i ] );
                std::string utf8_char = token.substr( i, char_len );

                auto it = byte_decoder.find( utf8_char );
                out.push_back( it != byte_decoder.end() ?
                    static_cast<char>( it->second ) : '?' );
                i += char_len;
            }
        }

        /**
         * @brief Get UTF-8 character length from first byte.
         */
        static size_t utf8CharLength( unsigned char first_byte )
        {
            if ( (first_byte & 0x80) == 0 ) return 1;
            if ( (first_byte & 0xE0) == 0xC0 ) return 2;
            if ( (first_byte & 0xF0) == 0xE0 ) return 3;
            if ( (first_byte & 0xF8) == 0xF0 ) return 4;
            return 1;
        }
    };
}