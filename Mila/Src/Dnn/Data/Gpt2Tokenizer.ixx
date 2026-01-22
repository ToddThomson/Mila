/**
 * @file Gpt2Tokenizer.ixx
 * @brief GPT-style BPE tokenizer and binary loader used by Mila.
 *
 * Loads a compact binary tokenizer format and provides encode/decode functionality.
 */

module;
#include <string>
#include <vector>
#include <unordered_map>
#include <span>
#include <optional>
#include <iostream>
#include <fstream>
#include <memory>
#include <regex>
#include <functional>
#include <limits>

export module Data.Gpt2Tokenizer;

import Data.Tokenizer;

namespace Mila::Dnn::Data
{
    /**
     * @brief Hash functor for std::pair<std::string, std::string>.
     *
     * Allows using string pairs as keys in unordered_map.
     */
    struct PairHash {
        size_t operator()( std::pair<std::string, std::string> const& p ) const noexcept {
            auto h1 = std::hash<std::string>{}(p.first);
            auto h2 = std::hash<std::string>{}(p.second);
            return h1 ^ (h2 << 1);
        }
    };

    /**
     * @class GptTokenizer
     * @brief GPT-style tokenizer implementing BPE encode/decode operations.
     *
     * The tokenizer expects a repository-specific compact binary format produced
     * by the conversion utility (for example: convert_gpt2_tokenizer.py).
     *
     * Thread-safety: instances are not guaranteed thread-safe for concurrent
     * mutations. Concurrent read-only encode/decode usage is acceptable when no
     * writer modifies state.
     */
    export class Gpt2Tokenizer : public Tokenizer {
    public:
        /**
         * @brief Create a tokenizer by loading the binary file at `path`.
         * @param path Filesystem path to the binary tokenizer file.
         * @return unique_ptr to a valid GptTokenizer on success, nullptr on failure.
         *
         * Preconditions: `path` points to a file produced by the repository
         * conversion utility or another producer that follows the same layout.
         */
        static std::unique_ptr<Gpt2Tokenizer> fromFile( const std::string& path ) {
            auto tokenizer = std::unique_ptr<Gpt2Tokenizer>( new Gpt2Tokenizer() );
            if ( !tokenizer->loadFromBinary( path ) ) {
                return nullptr;
            }

            return tokenizer;
        }

        /**
         * @brief Encode UTF-8 input text into a sequence of token IDs.
         * @param text Input UTF-8 string to tokenize.
         * @return Vector of token IDs representing the encoded input.
         *
         * This function applies the GPT-2 regex-based split then performs
         * ranked BPE merges according to the loaded merge table.
         */
        std::vector<TokenId> encode( const std::string& text ) override {
            std::vector<TokenId> result;

            auto words = splitIntoWords( text );

            for ( const auto& word : words ) {
                auto tokens = bpeEncode( word );
                result.insert( result.end(), tokens.begin(), tokens.end() );
            }

            return result;
        }

        /**
         * @brief Decode a sequence of token IDs back to a UTF-8 string.
         * @param tokens Span of token IDs to decode.
         * @return Concatenated string of token bytes.
         *
         * Unknown token IDs are skipped silently.
         */
        std::string decode( std::span<const TokenId> tokens ) override {
            std::string result;

            for ( auto tokenId : tokens ) {
                auto it = decoder_.find( tokenId );
                if ( it != decoder_.end() ) {
                    result += it->second;
                }
            }

            return result;
        }

        /**
         * @brief Encode and optionally add special tokens (BOS/EOS).
         * @param text Input text to encode.
         * @param addBos When true and a BOS token is available, insert at front.
         * @param addEos When true and an EOS token is available, append at end.
         * @return Token id vector including requested special tokens.
         */
        std::vector<TokenId> encodeWithSpecial(
            const std::string& text,
            bool addBos = true,
            bool addEos = true
        ) override {
            auto tokens = encode( text );

            if ( addBos && bosTokenId_ ) {
                tokens.insert( tokens.begin(), *bosTokenId_ );
            }

            if ( addEos && eosTokenId_ ) {
                tokens.push_back( *eosTokenId_ );
            }

            return tokens;
        }

        /**
         * @brief Get vocabulary size.
         * @return Number of tokens in the loaded vocabulary.
         */
        size_t getVocabSize() const override {
            return vocabSize_;
        }

        /**
         * @brief Get optional BOS token id.
         * @return Optional token id value for BOS, or std::nullopt if absent.
         */
        std::optional<TokenId> getBosTokenId() const override {
            return bosTokenId_;
        }

        /**
         * @brief Get optional EOS token id.
         * @return Optional token id value for EOS, or std::nullopt if absent.
         */
        std::optional<TokenId> getEosTokenId() const override {
            return eosTokenId_;
        }

        /**
         * @brief Get optional PAD token id.
         * @return Optional token id for PAD, or std::nullopt if absent.
         */
        std::optional<TokenId> getPadTokenId() const override {
            return padTokenId_;
        }

        /**
         * @brief Convert a token id to its string/bytes representation.
         * @param tokenId Token id to translate.
         * @return Token bytes as std::string or "<UNK>" if unknown.
         */
        std::string tokenToString( TokenId tokenId ) const override {
            auto it = decoder_.find( tokenId );
            return it != decoder_.end() ? it->second : "<UNK>";
        }

        /**
         * @brief Check whether a token id exists in the vocabulary.
         * @param tokenId Token id to check.
         * @return true when the id maps to a known token.
         */
        bool isValidToken( TokenId tokenId ) const override {
            return decoder_.contains( tokenId );
        }

    private:
        Gpt2Tokenizer() = default;

        /**
         * @brief Load the tokenizer from the repository binary layout.
         * @param path Binary file path.
         * @return true on success, false on I/O or format errors.
         *
         * Layout (uint32 are 4-byte little-endian integers):
         *  - vocab_size (uint32)
         *  - num_merges (uint32)
         *  - For each vocabulary entry:
         *      - token_len (uint32)
         *      - token_bytes (token_len bytes)
         *      - token_id (uint32)
         *  - For each merge:
         *      - len1 (uint32), bytes, len2 (uint32), bytes
         *  - Special token flags/ids: has_eos, eos_id, has_bos, bos_id, has_pad, pad_id (each uint32 when present)
         */
        bool loadFromBinary( const std::string& path ) {
            std::ifstream file( path, std::ios::binary );
            if ( !file ) {
                return false;
            }

            uint32_t vocabSize = 0;
            uint32_t numMerges = 0;
            file.read( reinterpret_cast<char*>(&vocabSize), sizeof( uint32_t ) );
            file.read( reinterpret_cast<char*>(&numMerges), sizeof( uint32_t ) );

            vocabSize_ = vocabSize;

            for ( uint32_t i = 0; i < vocabSize; ++i ) {
                uint32_t tokenLen = 0;
                file.read( reinterpret_cast<char*>( &tokenLen ), sizeof( uint32_t ) );

                std::string tokenStr( tokenLen, '\0' );
                file.read( tokenStr.data(), tokenLen );

                uint32_t tokenId = 0;
                file.read( reinterpret_cast<char*>( &tokenId ), sizeof( uint32_t ) );

                encoder_[ tokenStr ] = tokenId;
                decoder_[ tokenId ] = tokenStr;
            }

            for ( uint32_t i = 0; i < numMerges; ++i ) {
                uint32_t len1 = 0, len2 = 0;
                file.read( reinterpret_cast<char*>( &len1 ), sizeof( uint32_t ) );
                std::string token1( len1, '\0' );
                file.read( token1.data(), len1 );

                file.read( reinterpret_cast<char*>( &len2 ), sizeof( uint32_t ) );
                std::string token2( len2, '\0' );
                file.read( token2.data(), len2 );

                bpeMerges_[ {token1, token2} ] = static_cast<int>( i );
            }

            uint32_t hasEos = 0, hasBos = 0, hasPad = 0;
            file.read( reinterpret_cast<char*>(&hasEos), sizeof( uint32_t ) );
            if ( hasEos ) {
                uint32_t eosId = 0;
                file.read( reinterpret_cast<char*>(&eosId), sizeof( uint32_t ) );
                eosTokenId_ = eosId;
            }

            file.read( reinterpret_cast<char*>(&hasBos), sizeof( uint32_t ) );
            if ( hasBos ) {
                uint32_t bosId = 0;
                file.read( reinterpret_cast<char*>(&bosId), sizeof( uint32_t ) );
                bosTokenId_ = bosId;
            }

            file.read( reinterpret_cast<char*>(&hasPad), sizeof( uint32_t ) );
            if ( hasPad ) {
                uint32_t padId = 0;
                file.read( reinterpret_cast<char*>(&padId), sizeof( uint32_t ) );
                padTokenId_ = padId;
            }

            // Original GPT-2 Unicode-aware regex pattern:
            //pattern_ = std::regex(
            //    R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)"
            //);

            // ASCII-only fallback pattern (uncomment to use):
            //pattern_splitter_ = std::regex(
            //    "'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\\s\\w]+|\\s+(?!\\S)|\\s+" );
            // TJT: slight changes
            //  R"('s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+)"

            // Try to use the original GPT-2 Unicode-aware regex first.
            // If the platform's std::regex does not support \p{L}/\p{N} it will throw
            // std::regex_error and we fall back to an ASCII-compatible pattern.
            try {
                pattern_ = std::regex(
                    R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)"
                );
            }
            catch ( const std::regex_error& ) {
                // ASCII fallback: mirrors GPT-2 splitting behavior for ASCII text.
                // This fallback will be used on MSVC where \p escapes are unsupported.
                std::cerr << "Warning: std::regex does not support \\p Unicode properties on this platform; using ASCII fallback tokenizer splitting.\n";
                pattern_ = std::regex(
                    R"('s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+)"
                );
            }

            return true;
        }

        /**
         * @brief Split text into GPT-2 token pieces using the internal regex.
         * @param text UTF-8 input string.
         * @return Vector of matched pieces in left-to-right order.
         *
         * This uses std::regex iterators; large inputs may allocate intermediate strings.
         */
        std::vector<std::string> splitIntoWords( const std::string& text ) const {
            std::vector<std::string> words;
            std::sregex_iterator iter( text.begin(), text.end(), pattern_ );
            std::sregex_iterator end;

            for ( ; iter != end; ++iter ) {
                words.push_back( iter->str() );
            }

            return words;
        }

        /**
         * @brief Apply BPE merges to a single text piece and return token ids.
         * @param token Input piece to encode.
         * @return Vector of TokenId after BPE merges and lookup.
         *
         * Algorithm:
         *  - Start with single-character segmentation.
         *  - Repeatedly find the lowest-rank adjacent merge and apply it.
         *  - After no merges apply, map pieces to token ids using encoder_.
         */
        std::vector<TokenId> bpeEncode( const std::string& token ) const {
            std::vector<std::string> word;
            for ( char c : token ) {
                word.push_back( std::string( 1, c ) );
            }

            while ( word.size() > 1 ) {
                auto [i, rank] = findBestMerge( word );
                if ( i == -1 ) break;

                word[ i ] = word[ i ] + word[ i + 1 ];
                word.erase( word.begin() + i + 1 );
            }

            std::vector<TokenId> result;
            for ( const auto& w : word ) {
                auto it = encoder_.find( w );
                if ( it != encoder_.end() ) {
                    result.push_back( it->second );
                }
            }

            return result;
        }

        /**
         * @brief Find the best adjacent merge in `word`.
         * @param word Current vector of pieces.
         * @return Pair (index, rank) where index is merge position or -1 when no merge applies.
         */
        std::pair<int, int> findBestMerge(
            const std::vector<std::string>& word
        ) const {
            int bestIdx = -1;
            int bestRank = std::numeric_limits<int>::max();

            for ( size_t i = 0; i < word.size() - 1; ++i ) {
                auto it = bpeMerges_.find( { word[ i ], word[ i + 1 ] } );
                if ( it != bpeMerges_.end() && it->second < bestRank ) {
                    bestRank = it->second;
                    bestIdx = static_cast<int>( i );
                }
            }

            return { bestIdx, bestRank };
        }

        // token_id -> token bytes
        std::unordered_map<TokenId, std::string> decoder_;

        // token bytes -> token_id
        std::unordered_map<std::string, TokenId> encoder_;

        // BPE merge rules, mapped to rank (lower is applied earlier)
        std::unordered_map<std::pair<std::string, std::string>, int, PairHash> bpeMerges_;

        // Optional special tokens
        std::optional<TokenId> bosTokenId_;
        std::optional<TokenId> eosTokenId_;
        std::optional<TokenId> padTokenId_;

        // Vocabulary size
        size_t vocabSize_{ 0 };

        // GPT-2 splitting regex
        std::regex pattern_;
    };
}