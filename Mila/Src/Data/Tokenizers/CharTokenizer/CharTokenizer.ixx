/*!
 * \file
 * \brief Character-level tokenizer implementing the Tokenizer API.
 *
 * Provides a simple byte/char tokenizer that maps single-byte characters
 * to token ids via a TokenizerVocabulary.
 */

module;
#include <string>
#include <string_view>
#include <vector>
#include <span>
#include <memory>
#include <optional>
#include <filesystem>

export module Data.CharTokenizer;

import Data.CharVocabulary;
import Data.Tokenizer;
import Data.TokenizerVocabulary;

namespace Mila::Data
{
    using Mila::Dnn::Data::TokenId;
    using Mila::Dnn::Data::Tokenizer;
    using Mila::Dnn::Data::TokenizerVocabulary;

    /**
     * @brief Character-level tokenizer.
     *
     * This tokenizer treats tokens as single bytes (single-character strings).
     * It delegates all token <-> id mapping and persistence to a
     * `TokenizerVocabulary` implementation.
     *
     * Ownership:
     * - The tokenizer holds a shared pointer to the vocabulary so the same
     *   vocabulary instance may be shared between tokenizers or other users.
     *
     * Encoding/decoding semantics:
     * - `encode()` produces a TokenId for each byte in the input string. If a
     *   token is not found in the vocabulary the encoder emits `0u` as a
     *   fallback id.
     * - `decode()` converts each TokenId back to the first byte of the token
     *   string returned by the vocabulary; missing ids produce a '?' character.
     *
     * Note: This implementation does not add or interpret BOS/EOS tokens;
     * `encodeWithSpecial()` ignores the `addBos`/`addEos` flags because the
     * generic `TokenizerVocabulary` interface does not expose special-token ids.
     */
    export class CharTokenizer : public Tokenizer
    {
    public:

        /**
         * @brief Construct a CharTokenizer with a vocabulary.
         *
         * @param vocab Shared pointer to a TokenizerVocabulary implementation.
         *              Must remain valid for the lifetime of this tokenizer.
         */
        explicit CharTokenizer( CharVocabulary vocab )
            : vocab_( std::move( vocab ) )
        {}

        // ========================================================================
        // Convenience Loading Method(s)
        // ========================================================================

        static CharTokenizer load( const std::filesystem::path& path )
        {
            return CharTokenizer( CharVocabulary::load( path ) );
        }

        /**
         * @brief Encode text into token ids (one id per input byte).
         *
         * @param text UTF-8 encoded text to encode. Each input byte is treated
         *             as a separate token; callers should handle multi-byte
         *             characters if needed.
         * @return std::vector<TokenId> Vector of token ids; missing tokens map to 0u.
         */
        std::vector<TokenId> encode( const std::string& text ) override
        {
            std::vector<TokenId> out;
            out.reserve( text.size() );

            for ( unsigned char c : text )
            {
                std::string tk( 1, static_cast<char>( c ) );
                auto id_opt = vocab_.tokenToId( tk );
                out.push_back( id_opt ? *id_opt : static_cast<TokenId>( 0u ) );
            }

            return out;
        }

        /**
         * @brief Decode token ids back to text.
         *
         * Each token id is converted to its token string via the vocabulary,
         * and the first byte of that token is appended to the result. If an id
         * is missing the character '?' is appended.
         *
         * @param tokens Span of token ids to decode.
         * @return Decoded text string.
         */
        std::string decode( std::span<const TokenId> tokens ) override
        {
            std::string out;
            out.reserve( tokens.size() );

            for ( TokenId id : tokens )
            {
                auto tok_opt = vocab_.idToToken( id );
                if ( tok_opt && !tok_opt->empty() )
                {
                    out.push_back( (*tok_opt)[0] );
                }
                else
                {
                    out.push_back( '?' );
                }
            }

            return out;
        }

        /**
         * @brief Number of tokens in the underlying vocabulary.
         */
        size_t getVocabSize() const override
        {
            return vocab_.getSize();
        }

        /**
         * @brief BOS id query - not supported for char-level tokenizer.
         *
         * Returns empty optional because the generic vocabulary does not expose
         * special-token ids in the interface.
         */
        std::optional<TokenId> getBosTokenId() const override
        {
            return std::nullopt;
        }

        /**
         * @brief EOS id query - not supported for char-level tokenizer.
         */
        std::optional<TokenId> getEosTokenId() const override
        {
            return std::nullopt;
        }

        /**
         * @brief PAD id query - not supported for char-level tokenizer.
         */
        std::optional<TokenId> getPadTokenId() const override
        {
            return std::nullopt;
        }

        /**
         * @brief Convert a token id to a debug string.
         *
         * Returns the token string from the vocabulary or an empty string if not found.
         */
        std::string tokenToString( TokenId tokenId ) const override
        {
            auto tok = vocab_.idToToken( tokenId );
            return tok ? *tok : std::string();
        }

        /**
         * @brief Check if token id is valid in the vocabulary.
         */
        bool isValidToken( TokenId tokenId ) const override
        {
            return static_cast<bool>( vocab_.idToToken( tokenId ) );
        }

    private:
        CharVocabulary vocab_;
    };
}