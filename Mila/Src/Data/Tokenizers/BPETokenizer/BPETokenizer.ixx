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
            // Build token -> id map and compute max token length.
            /*if ( vocab_ )
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
            }*/
        }

        // ========================================================================
        // Convenience Methods - Shortcuts for common workflows
        // ========================================================================

        // Load custom-trained vocabulary and create tokenizer in one step
        static BpeTokenizer load( const std::filesystem::path& path ) {
            return BpeTokenizer( BpeVocabulary::load( path ) );
        }

        // Load pre-trained models and create tokenizer in one step
        static BpeTokenizer loadGpt2(
            const std::filesystem::path& vocabPath,
            const std::filesystem::path& mergesPath ) {
            
            return BpeTokenizer( BpeVocabulary::loadGpt2( vocabPath, mergesPath ) );
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
            std::vector<TokenId> out;
            // REVIEW: Not needed?
            /*if ( !vocab_ )
            {
                return out;
            }*/

            const size_t text_len = text.size();
            out.reserve( std::max<size_t>( 16, text_len / 2 ) );

            size_t i = 0;
            while ( i < text_len )
            {
                // Determine search upper bound
                size_t max_len = std::min( max_token_len_, text_len - i );

                bool matched = false;

                // Try longest-first greedy match
                for ( size_t len = max_len; len >= 1; --len )
                {
                    // Construct token candidate from substring
                    std::string_view sv( text.data() + i, len );
                    std::string key( sv ); // map keys are std::string

                    auto it = token_map_.find( key );
                    if ( it != token_map_.end() )
                    {
                        out.push_back( it->second );
                        i += len;
                        matched = true;
                        break;
                    }

                    if ( len == 1 )
                    {
                        break; // avoid underflow of size_t
                    }
                }

                if ( !matched )
                {
                    // No token matched. Emit UNK if available, otherwise 0u.
                    // tokenToId may return unk id if vocabulary implements it.
                    auto unk_opt = vocab_.tokenToId( std::string( 1, '\1' ) ); // try explicit UNK marker
                    if ( !unk_opt )
                    {
                        // Ask vocabulary for the single-byte character at this position
                        std::string single( 1, text[ i ] );
                        auto id_opt = vocab_.tokenToId( single );
                        if ( id_opt )
                        {
                            out.push_back( *id_opt );
                        }
                        else
                        {
                            out.push_back( static_cast<TokenId>(0u) );
                        }
                    }
                    else
                    {
                        out.push_back( *unk_opt );
                    }

                    ++i;
                }
            }

            return out;
        }

        std::string decode( std::span<const TokenId> tokens ) override
        {
            std::string out;
            /*if ( !vocab_ )
            {
                return out;
            }*/

            // Reserve some space to reduce reallocations
            out.reserve( tokens.size() * std::max<size_t>( 1, max_token_len_ ) );

            for ( TokenId id : tokens )
            {
                auto tok_opt = vocab_.idToToken( id );
                if ( tok_opt )
                {
                    out.append( *tok_opt );
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