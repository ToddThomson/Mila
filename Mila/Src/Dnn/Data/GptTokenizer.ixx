module;
#include <string>
#include <vector>
#include <unordered_map>
#include <span>
#include <optional>
#include <fstream>
#include <memory>
#include <regex>
#include <functional>

export module Data.GptTokenizer;

import Data.Tokenizer;

namespace Mila::Data
{
    // Hash for std::pair<std::string,std::string> so unordered_map works on MSVC
    struct PairHash {
        size_t operator()( std::pair<std::string, std::string> const& p ) const noexcept {
            // combine two string hashes (simple, fast, acceptable for tokenizer rules)
            auto h1 = std::hash<std::string>{}( p.first );
            auto h2 = std::hash<std::string>{}( p.second );
            return h1 ^ ( h2 << 1 );
        }
    };

    export class GptTokenizer : public Tokenizer {
    public:
        // Load tokenizer from binary file
        static std::unique_ptr<GptTokenizer> fromFile( const std::string& path );

        std::vector<TokenId> encode( const std::string& text ) override;
        std::string decode( std::span<const TokenId> tokens ) override;

        std::vector<TokenId> encodeWithSpecial(
            const std::string& text,
            bool addBos = true,
            bool addEos = true
        ) override;

        size_t getVocabSize() const override {
            return vocabSize_;
        }

        std::optional<TokenId> getBosTokenId() const override {
            return bosTokenId_;
        }
        std::optional<TokenId> getEosTokenId() const override {
            return eosTokenId_;
        }
        std::optional<TokenId> getPadTokenId() const override {
            return padTokenId_;
        }

        std::string tokenToString( TokenId tokenId ) const override;
        bool isValidToken( TokenId tokenId ) const override;

    private:
        GptTokenizer() = default;

        // Load from binary format
        bool loadFromBinary( const std::string& path );

        // BPE encoding helpers
        std::vector<std::string> splitIntoWords( const std::string& text ) const;
        std::vector<TokenId> bpeEncode( const std::string& token ) const;
        std::pair<int, int> findBestMerge( const std::vector<std::string>& word ) const;

        // Vocabulary: token_id -> token_bytes
        std::unordered_map<TokenId, std::string> decoder_;

        // Reverse: token_bytes -> token_id
        std::unordered_map<std::string, TokenId> encoder_;

        // BPE merge rules: (token1, token2) -> rank
        std::unordered_map<std::pair<std::string, std::string>, int, PairHash> bpeMerges_;

        // Special tokens
        std::optional<TokenId> bosTokenId_;
        std::optional<TokenId> eosTokenId_;
        std::optional<TokenId> padTokenId_;

        size_t vocabSize_{ 0 };

        // Regex pattern for splitting text (GPT-2 pattern)
        std::regex pattern_;
    };

    // Implementation
    std::unique_ptr<GptTokenizer> GptTokenizer::fromFile( const std::string& path ) {
        auto tokenizer = std::unique_ptr<GptTokenizer>( new GptTokenizer() );
        if ( !tokenizer->loadFromBinary( path ) ) {
            return nullptr;
        }
        return tokenizer;
    }

    bool GptTokenizer::loadFromBinary( const std::string& path ) {
        std::ifstream file( path, std::ios::binary );
        if ( !file ) {
            return false;
        }

        // Read header
        uint32_t vocabSize, numMerges;
        file.read( reinterpret_cast<char*>(&vocabSize), sizeof( uint32_t ) );
        file.read( reinterpret_cast<char*>(&numMerges), sizeof( uint32_t ) );

        vocabSize_ = vocabSize;

        // Read vocabulary
        for ( uint32_t i = 0; i < vocabSize; ++i ) {
            uint32_t tokenLen;
            file.read( reinterpret_cast<char*>( &tokenLen ), sizeof( uint32_t ) );

            std::string tokenStr( tokenLen, '\0' );
            file.read( tokenStr.data(), tokenLen );

            uint32_t tokenId;
            file.read( reinterpret_cast<char*>( &tokenId ), sizeof( uint32_t ) );

            encoder_[ tokenStr ] = tokenId;
            decoder_[ tokenId ] = tokenStr;
        }

        // Read BPE merges
        for ( uint32_t i = 0; i < numMerges; ++i ) {
            uint32_t len1, len2;
            file.read( reinterpret_cast<char*>( &len1 ), sizeof( uint32_t ) );
            std::string token1( len1, '\0' );
            file.read( token1.data(), len1 );

            file.read( reinterpret_cast<char*>( &len2 ), sizeof( uint32_t ) );
            std::string token2( len2, '\0' );
            file.read( token2.data(), len2 );

            bpeMerges_[ {token1, token2} ] = static_cast<int>( i );
        }

        // Read special tokens
        uint32_t hasEos, hasBos, hasPad;
        file.read( reinterpret_cast<char*>(&hasEos), sizeof( uint32_t ) );
        if ( hasEos ) {
            uint32_t eosId;
            file.read( reinterpret_cast<char*>(&eosId), sizeof( uint32_t ) );
            eosTokenId_ = eosId;
        }

        file.read( reinterpret_cast<char*>(&hasBos), sizeof( uint32_t ) );
        if ( hasBos ) {
            uint32_t bosId;
            file.read( reinterpret_cast<char*>(&bosId), sizeof( uint32_t ) );
            bosTokenId_ = bosId;
        }

        file.read( reinterpret_cast<char*>(&hasPad), sizeof( uint32_t ) );
        if ( hasPad ) {
            uint32_t padId;
            file.read( reinterpret_cast<char*>(&padId), sizeof( uint32_t ) );
            padTokenId_ = padId;
        }

        // Initialize regex pattern for GPT-2 tokenization
        pattern_ = std::regex(
            R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)"
        );

        return true;
    }

    std::vector<TokenId> GptTokenizer::encode( const std::string& text ) {
        std::vector<TokenId> result;

        // Split text into words using GPT-2 pattern
        auto words = splitIntoWords( text );

        // Encode each word using BPE
        for ( const auto& word : words ) {
            auto tokens = bpeEncode( word );
            result.insert( result.end(), tokens.begin(), tokens.end() );
        }

        return result;
    }

    std::vector<TokenId> GptTokenizer::encodeWithSpecial(
        const std::string& text,
        bool addBos,
        bool addEos
    ) {
        auto tokens = encode( text );

        if ( addBos && bosTokenId_ ) {
            tokens.insert( tokens.begin(), *bosTokenId_ );
        }

        if ( addEos && eosTokenId_ ) {
            tokens.push_back( *eosTokenId_ );
        }

        return tokens;
    }

    std::string GptTokenizer::decode( std::span<const TokenId> tokens ) {
        std::string result;

        for ( auto tokenId : tokens ) {
            auto it = decoder_.find( tokenId );
            if ( it != decoder_.end() ) {
                result += it->second;
            }
        }

        return result;
    }

    std::string GptTokenizer::tokenToString( TokenId tokenId ) const {
        auto it = decoder_.find( tokenId );
        return it != decoder_.end() ? it->second : "<UNK>";
    }

    bool GptTokenizer::isValidToken( TokenId tokenId ) const {
        return decoder_.contains( tokenId );
    }

    std::vector<std::string> GptTokenizer::splitIntoWords( const std::string& text ) const {
        std::vector<std::string> words;
        std::sregex_iterator iter( text.begin(), text.end(), pattern_ );
        std::sregex_iterator end;

        for ( ; iter != end; ++iter ) {
            words.push_back( iter->str() );
        }

        return words;
    }

    std::vector<TokenId> GptTokenizer::bpeEncode( const std::string& token ) const {
        // Convert to individual characters
        std::vector<std::string> word;
        for ( char c : token ) {
            word.push_back( std::string( 1, c ) );
        }

        // Apply BPE merges
        while ( word.size() > 1 ) {
            auto [i, j] = findBestMerge( word );
            if ( i == -1 ) break;

            // Merge tokens at positions i and i+1
            word[ i ] = word[ i ] + word[ i + 1 ];
            word.erase( word.begin() + i + 1 );
        }

        // Convert to token IDs
        std::vector<TokenId> result;
        for ( const auto& w : word ) {
            auto it = encoder_.find( w );
            if ( it != encoder_.end() ) {
                result.push_back( it->second );
            }
        }

        return result;
    }

    std::pair<int, int> GptTokenizer::findBestMerge(
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
}