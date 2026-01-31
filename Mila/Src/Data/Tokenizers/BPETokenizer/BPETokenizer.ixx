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

export module Data.BpeTokenizer;
import Data.BpeVocabulary;
import Data.Tokenizer;
import Data.TokenizerVocabulary;

namespace Mila::Data
{
    using Mila::Dnn::Data::TokenId;
    using Mila::Dnn::Data::Tokenizer;
    using Mila::Dnn::Data::TokenizerVocabulary;

    /**
     * @brief Byte-Pair Encoding (BPE) style tokenizer.
     *
     * Implements a greedy longest-match tokenizer using the provided
     * `TokenizerVocabulary`. On construction the tokenizer caches the
     * vocabulary token strings and their ids to enable efficient matching.
     *
     * Encoding strategy:
     *  - At each text position try to match the longest token present in the
     *    vocabulary; if no match is found emit the UNK id (or 0u fallback).
     *
     * Decoding strategy:
     *  - Concatenate token strings returned by the vocabulary for each id.
     */
    export class BpeTokenizer : public Tokenizer
    {
    public:

        explicit BpeTokenizer( BpeVocabulary vocab )
            : vocab_( std::move( vocab ) )
        {
            size_t n = vocab_.getSize();
            max_token_len_ = 0;

            for ( size_t i = 0; i < n; ++i )
            {
                auto tok = vocab_.idToToken( static_cast<TokenId>( i ) );
                if ( tok && !tok->empty() )
                {
                    token_map_.emplace( *tok, static_cast<TokenId>( i ) );
                    if ( tok->size() > max_token_len_ )
                    {
                        max_token_len_ = tok->size();
                    }
                }
            }
        }

        // ========================================================================
        // Convenience Methods - Shortcuts for common workflows
        // ========================================================================

        // Load custom-trained vocabulary and create tokenizer in one step
        static BpeTokenizer load( const std::filesystem::path& path ) {
            return BpeTokenizer( BpeVocabulary::load( path ) );
        }

        // Load pre-trained models and create tokenizer in one step
        static BpeTokenizer loadGpt2( const std::filesystem::path& vocabPath )
        {
            return BpeTokenizer( BpeVocabulary::loadGpt2( vocabPath ) );
        }

        static BpeTokenizer loadLlama( const std::filesystem::path& modelPath ) {
            return BpeTokenizer( BpeVocabulary::loadLlama( modelPath ) );
        }

        static BpeTokenizer loadMistral(
            const std::filesystem::path& vocabPath,
            const std::filesystem::path& mergesPath ) {
            return BpeTokenizer( BpeVocabulary::loadMistral( vocabPath, mergesPath ) );
        }

        // ========================================================================
        // Tokenizer Interface Implementation
        // ========================================================================

        std::vector<TokenId> encode( const std::string& text ) override
        {
            auto start_time = std::chrono::high_resolution_clock::now();

            // Step 1: Split into byte-level tokens using GPT-2's encoding
            std::vector<std::string> tokens;

            if ( vocab_.isByteLevel() ) {
                // Byte-level: use unicode mapping
                const auto& byte_encoder = vocab_.getByteEncoder();
                for ( unsigned char byte : text ) {
                    tokens.push_back( byte_encoder.at( byte ) );
                }
            }
            else {
                // Character-level: use plain characters
                for ( char c : text ) {
                    tokens.push_back( std::string( 1, c ) );
                }
            }

            std::cout << "Tokenizing " << text.size() << " characters...\n";
            size_t initial_tokens = tokens.size();

            // Step 2: Apply merges efficiently - multiple passes
            bool changed = true;
            size_t pass = 0;
            
            while ( changed ) {
                changed = false;
                std::vector<std::string> new_tokens;
                new_tokens.reserve( tokens.size() );

                for ( size_t i = 0; i < tokens.size(); ) {
                    // Try to merge with next token
                    if ( i + 1 < tokens.size() ) {
                        auto priority = vocab_.getMergePriority( tokens[ i ], tokens[ i + 1 ] );
                        if ( priority ) {
                            // Merge found
                            std::string merged;
                            merged.reserve( tokens[ i ].size() + tokens[ i + 1 ].size() );
                            merged = tokens[ i ];
                            merged += tokens[ i + 1 ];
                            new_tokens.push_back( std::move( merged ) );
                            i += 2;
                            changed = true;
                            continue;
                        }
                    }
                    // No merge, keep token as-is
                    new_tokens.push_back( std::move( tokens[ i ] ) );
                    ++i;
                }

                tokens = std::move( new_tokens );

                ++pass;

                // Progress update every 5 passes or when done
                if ( pass % 5 == 0 || !changed ) {
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start_time
                    ).count();

                    std::cout << "\r[" << elapsed << "ms] Pass " << pass
                        << " | Tokens: " << tokens.size()
                        << "          " << std::flush;
                }
            }

            // Step 3: Convert to IDs
            std::vector<TokenId> out;
            out.reserve( tokens.size() );
            for ( const auto& token : tokens ) {
                auto id = vocab_.tokenToId( token );
                out.push_back( id ? *id : 0 );
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            std::cout << "\n Encoding completed in " << ms << "ms"
                << " (" << initial_tokens << " -> " << tokens.size() << " tokens)\n";

            return out;
        }

        //std::vector<TokenId> encode_old( const std::string& text )
        //{
        //    std::vector<TokenId> out;

        //    const size_t text_len = text.size();
        //    out.reserve( std::max<size_t>( 16, text_len / 2 ) );

        //    size_t i = 0;
        //    while ( i < text_len )
        //    {
        //        // Determine search upper bound
        //        size_t max_len = std::min( max_token_len_, text_len - i );

        //        bool matched = false;

        //        for ( size_t len = max_len; len > 0; --len )
        //        {
        //            std::string_view sv( text.data() + i, len );
        //            std::string key( sv );
        //            auto it = token_map_.find( key );
        //            
        //            if ( it != token_map_.end() )
        //            {
        //                out.push_back( it->second );
        //                i += len;
        //                matched = true;
        //                break;
        //            }
        //        }

        //        if ( !matched )
        //        {
        //            // No token matched. Emit UNK if available, otherwise 0u.
        //            // tokenToId may return unk id if vocabulary implements it.
        //            auto unk_opt = vocab_.tokenToId( std::string( 1, '\1' ) ); // try explicit UNK marker
        //            if ( !unk_opt )
        //            {
        //                // Ask vocabulary for the single-byte character at this position
        //                std::string single( 1, text[ i ] );
        //                auto id_opt = vocab_.tokenToId( single );
        //                if ( id_opt )
        //                {
        //                    out.push_back( *id_opt );
        //                }
        //                else
        //                {
        //                    out.push_back( static_cast<TokenId>(0u) );
        //                }
        //            }
        //            else
        //            {
        //                out.push_back( *unk_opt );
        //            }

        //            ++i;
        //        }
        //    }

        //    return out;
        //}

        std::string decode( std::span<const TokenId> tokens ) override
        {
            // TODO: Update for byte-level decoding setting
            
            const auto& byte_decoder = vocab_.getByteDecoder();

            std::string out;
            out.reserve( tokens.size() * std::max<size_t>( 1, max_token_len_ ) );

            for ( TokenId id : tokens ) {
                auto tok_opt = vocab_.idToToken( id );
                if ( tok_opt ) {
                    // Decode each UTF-8 character in the token back to its byte
                    const std::string& token = *tok_opt;
                    size_t i = 0;
                    while ( i < token.size() ) {
                        // Extract one UTF-8 character
                        size_t char_len = 1;
                        if ( (token[ i ] & 0x80) == 0 ) {
                            char_len = 1;
                        }
                        else if ( (token[ i ] & 0xE0) == 0xC0 ) {
                            char_len = 2;
                        }
                        else if ( (token[ i ] & 0xF0) == 0xE0 ) {
                            char_len = 3;
                        }
                        else if ( (token[ i ] & 0xF8) == 0xF0 ) {
                            char_len = 4;
                        }

                        std::string utf8_char = token.substr( i, char_len );
                        auto it = byte_decoder.find( utf8_char );
                        if ( it != byte_decoder.end() ) {
                            out.push_back( static_cast<char>(it->second) );
                        }
                        else {
                            out.push_back( '?' );  // Unknown character
                        }
                        i += char_len;
                    }
                }
                else {
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
            //if ( !vocab_ ) return std::string();
            auto tok = vocab_.idToToken( tokenId );
            
            return tok ? *tok : std::string();
        }

        bool isValidToken( TokenId tokenId ) const override
        {
            return static_cast<bool>(vocab_.idToToken( tokenId ));
        }

    private:
        BpeVocabulary vocab_;
        std::unordered_map<std::string, TokenId> token_map_;
        size_t max_token_len_{ 0 };
    };
}