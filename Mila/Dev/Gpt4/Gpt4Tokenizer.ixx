/*!
 * \file Gpt4Tokenizer.ixx
 * \brief GPT-4 style BPE tokenizer for Llama 3.x models.
 *
 * Mirrors Gpt2Tokenizer but adds a special token pre-pass in encode()
 * required by Llama 3.x's large special token set.
 *
 * Encode pipeline:
 *   1. Special token pre-pass: scan input for registered special token strings
 *      (e.g. "<|begin_of_text|>") and split them out as direct ID lookups,
 *      bypassing BPE entirely. Matching is longest-first to handle any prefix
 *      collisions in the special token set.
 *   2. Pre-tokenize remaining text segments using the Llama3Regex pattern.
 *   3. Byte-encode each pre-token.
 *   4. Apply BPE merges greedily (lowest priority first).
 *   5. Convert final tokens to IDs.
 *
 * Decode pipeline:
 *   - Concatenate token strings and decode GPT-2 style byte encoding back to UTF-8.
 */

module;
#include <string>
#include <string_view>
#include <vector>
#include <span>
#include <memory>
#include <optional>
#include <variant>
#include <filesystem>
#include <chrono>
#include <iostream>
#include <regex>
#include <limits>
#include <stdexcept>

export module Data.Tokenizers.Bpe.Gpt4Tokenizer;

import Data.Tokenizers.Bpe.Gpt4Vocabulary;
import Data.Tokenizers.Bpe.PreTokenizationMode;
import Data.Tokenizer;
import Data.TokenizerVocabulary;

namespace Mila::Data
{
    using Mila::Dnn::Data::TokenId;
    using Mila::Dnn::Data::Tokenizer;
    using Mila::Dnn::Data::TokenizerVocabulary;

    /**
     * @brief GPT-4 style BPE tokenizer targeting Llama 3.x models.
     *
     * Use Gpt4Vocabulary::loadLlama32() to construct the vocabulary,
     * then wrap it in this tokenizer:
     *
     * @code
     * auto vocab = Gpt4Vocabulary::loadLlama32("llama32_tokenizer.bin");
     * auto tokenizer = std::make_shared<Gpt4BpeTokenizer>(std::move(vocab));
     * auto ids = tokenizer->encode("Hello, world!");
     * @endcode
     */
    export class Gpt4Tokenizer : public Tokenizer
    {
    public:

        explicit Gpt4Tokenizer( Gpt4Vocabulary vocab )
            : vocab_( std::move( vocab ) )
        {
            initializePreTokenization();
        }

        // ====================================================================
        // Convenience factory
        // ====================================================================

        static std::shared_ptr<Gpt4Tokenizer> loadLlama32( const std::filesystem::path& path )
        {
            return std::make_shared<Gpt4Tokenizer>( Gpt4Vocabulary::loadLlama32( path ) );
        }

        // ====================================================================
        // Tokenizer Interface
        // ====================================================================

        /**
         * @brief Encode text to token IDs.
         *
         * Special tokens (e.g. "<|begin_of_text|>") are matched first and
         * returned as direct IDs without going through BPE. All remaining
         * text segments are pre-tokenized with the Llama3 regex and then
         * BPE-merged.
         */
        std::vector<TokenId> encode( const std::string& text ) override
        {
            const auto start_time = std::chrono::high_resolution_clock::now();

            // Step 1: Special token pre-pass
            // Split text into alternating (special | plain) segments.
            // Special tokens are resolved directly to IDs; plain segments
            // proceed through pre-tokenization and BPE.
            std::vector<TokenId> out;
            const auto& special_list = vocab_.getSpecialTokenList();

            if ( special_list.empty() )
            {
                // Fast path: no special tokens registered, skip pre-pass
                encodeSegment( text, out );
            }
            else
            {
                size_t pos = 0;
                while ( pos < text.size() )
                {
                    // Try to match any special token at current position (longest first)
                    bool matched = false;
                    for ( const auto& [token_str, token_id] : special_list )
                    {
                        if ( text.compare( pos, token_str.size(), token_str ) == 0 )
                        {
                            out.push_back( token_id );
                            pos += token_str.size();
                            matched = true;
                            break;
                        }
                    }

                    if ( !matched )
                    {
                        // Find the start of the next special token (or end of string)
                        size_t next_special = text.size();
                        for ( const auto& [token_str, token_id] : special_list )
                        {
                            size_t found = text.find( token_str, pos );
                            if ( found != std::string::npos && found < next_special )
                            {
                                next_special = found;
                            }
                        }

                        // Encode the plain text segment between pos and next_special
                        encodeSegment( text.substr( pos, next_special - pos ), out );
                        pos = next_special;
                    }
                }
            }

            const auto end_time = std::chrono::high_resolution_clock::now();
            const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time).count();
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
            return vocab_.getSpecialTokenId(
                vocab_.getConfig().getSpecialTokens().bos_token );
        }

        std::optional<TokenId> getEosTokenId() const override
        {
            return vocab_.getSpecialTokenId(
                vocab_.getConfig().getSpecialTokens().eos_token );
        }

        std::optional<TokenId> getPadTokenId() const override
        {
            if ( !vocab_.getConfig().getSpecialTokens().use_pad )
            {
                return std::nullopt;
            }
            return vocab_.getSpecialTokenId(
                vocab_.getConfig().getSpecialTokens().pad_token );
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

        const Gpt4Vocabulary& getVocab() const
        {
            return vocab_;
        }

    private:
        Gpt4Vocabulary vocab_;
        std::optional<std::regex> pre_tokenization_regex_;

        // ====================================================================
        // Initialization
        // ====================================================================

        void initializePreTokenization()
        {
            const auto& pattern = vocab_.getConfig().getPreTokenizationPattern();
            if ( pattern.empty() )
            {
                return;
            }

            try
            {
                pre_tokenization_regex_ = std::regex( pattern, std::regex::ECMAScript );
            }
            catch ( const std::regex_error& )
            {
                // \p{L} / \p{N} not supported by MSVC std::regex — use ASCII fallback.
                // Known gap for alpha.2: non-ASCII text will tokenize differently from HF.
                // Post-alpha: consider RE2 or ICU for full Unicode property support.
                std::cerr << "Warning: Unicode regex not supported by std::regex, "
                    "using ASCII fallback for Llama3 pre-tokenization.\n"
                    "Non-ASCII tokenization may differ from HuggingFace reference.\n";

                pre_tokenization_regex_ = std::regex(
                    LLAMA3_PRETOKENIZATION_PATTERN_ASCII_FALLBACK,
                    std::regex::ECMAScript );
            }
        }

        // ====================================================================
        // Encode helpers
        // ====================================================================

        /**
         * @brief Encode a plain text segment (no special tokens) to IDs.
         *
         * Applies pre-tokenization regex, byte encoding, and BPE merges.
         */
        void encodeSegment( const std::string& text, std::vector<TokenId>& out )
        {
            if ( text.empty() ) return;

            // Pre-tokenize into words
            const std::vector<std::string> words = preTokenize( text );

            size_t pass = 0;
            for ( const auto& word : words )
            {
                // Byte-encode the word
                std::vector<std::string> tokens;
                tokens.reserve( word.size() );

                if ( vocab_.isByteLevel() )
                {
                    const auto& byte_encoder = Gpt4Vocabulary::getByteEncoder();
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

                // Apply BPE merges (greedy lowest-priority-first)
                while ( tokens.size() > 1 )
                {
                    int best_idx = -1;
                    size_t best_priority = std::numeric_limits<size_t>::max();

                    for ( size_t i = 0; i < tokens.size() - 1; ++i )
                    {
                        auto priority = vocab_.getMergePriority( tokens[ i ], tokens[ i + 1 ] );
                        if ( priority && *priority < best_priority )
                        {
                            best_priority = *priority;
                            best_idx = static_cast<int>( i );
                        }
                    }

                    if ( best_idx == -1 ) break;

                    tokens[ best_idx ] = tokens[ best_idx ] + tokens[ best_idx + 1 ];
                    tokens.erase( tokens.begin() + best_idx + 1 );
                }

                // Convert tokens to IDs
                for ( const auto& token : tokens )
                {
                    auto id = vocab_.tokenToId( token );
                    out.push_back( id ? *id : 0 );
                }

                ++pass;
                if ( pass % 100 == 0 )
                {
                    std::cout << "\r[BPE] Words: " << pass
                        << " | Tokens: " << out.size()
                        << "          " << std::flush;
                }
            }
        }

        /**
         * @brief Pre-tokenize a plain text segment using the configured regex.
         */
        std::vector<std::string> preTokenize( const std::string& text )
        {
            if ( !pre_tokenization_regex_ )
            {
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
         * @brief Decode a single token string to output bytes.
         */
        void decodeToken( const std::string& token, std::string& out )
        {
            if ( !vocab_.isByteLevel() )
            {
                out.append( token );
                return;
            }

            const auto& byte_decoder = Gpt4Vocabulary::getByteDecoder();
            size_t i = 0;
            while ( i < token.size() )
            {
                const size_t char_len = utf8CharLength( token[ i ] );
                const std::string utf8_char = token.substr( i, char_len );

                auto it = byte_decoder.find( utf8_char );
                out.push_back( it != byte_decoder.end() ?
                    static_cast<char>( it->second ) : '?' );
                i += char_len;
            }
        }

        static size_t utf8CharLength( unsigned char first_byte )
        {
            if ( (first_byte & 0x80) == 0 )   return 1;
            if ( (first_byte & 0xE0) == 0xC0 ) return 2;
            if ( (first_byte & 0xF0) == 0xE0 ) return 3;
            if ( (first_byte & 0xF8) == 0xF0 ) return 4;
            return 1;
        }
    };
}
