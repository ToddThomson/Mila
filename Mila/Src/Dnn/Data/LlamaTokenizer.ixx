module;
#include <string>
#include <vector>
#include <unordered_map>
#include <span>
#include <optional>
#include <fstream>
#include <memory>
#include <algorithm>

export module Data.LlamaTokenizer;

import Data.Tokenizer;

namespace Mila::Dnn::Data
{
    export class LlamaTokenizer : public Tokenizer {
    public:
        // Load tokenizer from binary file
        static std::unique_ptr<LlamaTokenizer> fromFile( const std::string& path );

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
        LlamaTokenizer() = default;

        // Load from binary format
        bool loadFromBinary( const std::string& path );

        // SentencePiece encoding helpers
        std::vector<TokenId> sentencePieceEncode( const std::string& text ) const;
        std::string normalizeText( const std::string& text ) const;

        struct SpmPiece {
            std::string piece;
            float score;
            TokenId id;
        };

        // Vocabulary pieces sorted by score
        std::vector<SpmPiece> pieces_;

        // Quick lookup: piece -> id
        std::unordered_map<std::string, TokenId> pieceToId_;

        // Reverse lookup: id -> piece
        std::unordered_map<TokenId, std::string> idToPiece_;

        // Special tokens
        std::optional<TokenId> bosTokenId_;
        std::optional<TokenId> eosTokenId_;
        std::optional<TokenId> padTokenId_;
        std::optional<TokenId> unkTokenId_;

        size_t vocabSize_{ 0 };

        // SentencePiece byte fallback for unknown bytes
        bool useByteFallback_{ true };
    };

    // Implementation
    std::unique_ptr<LlamaTokenizer> LlamaTokenizer::fromFile( const std::string& path ) {
        auto tokenizer = std::unique_ptr<LlamaTokenizer>( new LlamaTokenizer() );
        if ( !tokenizer->loadFromBinary( path ) ) {
            return nullptr;
        }
        return tokenizer;
    }

    bool LlamaTokenizer::loadFromBinary( const std::string& path ) {
        std::ifstream file( path, std::ios::binary );
        if ( !file ) {
            return false;
        }

        // Read header
        uint32_t vocabSize;
        file.read( reinterpret_cast<char*>(&vocabSize), sizeof( uint32_t ) );
        vocabSize_ = vocabSize;

        uint8_t useByteFallback;
        file.read( reinterpret_cast<char*>(&useByteFallback), sizeof( uint8_t ) );
        useByteFallback_ = useByteFallback != 0;

        // Read vocabulary pieces
        pieces_.reserve( vocabSize );

        for ( uint32_t i = 0; i < vocabSize; ++i ) {
            uint32_t pieceLen;
            file.read( reinterpret_cast<char*>( &pieceLen ), sizeof( uint32_t ) );

            std::string piece( pieceLen, '\0' );
            file.read( piece.data(), pieceLen );

            float score;
            file.read( reinterpret_cast<char*>( &score ), sizeof( float ) );

            uint32_t tokenId;
            file.read( reinterpret_cast<char*>(&tokenId), sizeof( uint32_t ) );

            pieces_.push_back( { piece, score, tokenId } );
            pieceToId_[ piece ] = tokenId;
            idToPiece_[ tokenId ] = piece;
        }

        // Sort pieces by score (higher scores = higher priority)
        std::sort( pieces_.begin(), pieces_.end(),
            []( const SpmPiece& a, const SpmPiece& b ) {
                return a.score > b.score;
            } );

        // Read special tokens
        uint32_t hasBos, hasEos, hasPad, hasUnk;

        file.read( reinterpret_cast<char*>(&hasBos), sizeof( uint32_t ) );
        if ( hasBos ) {
            uint32_t bosId;
            file.read( reinterpret_cast<char*>(&bosId), sizeof( uint32_t ) );
            bosTokenId_ = bosId;
        }

        file.read( reinterpret_cast<char*>(&hasEos), sizeof( uint32_t ) );
        if ( hasEos ) {
            uint32_t eosId;
            file.read( reinterpret_cast<char*>(&eosId), sizeof( uint32_t ) );
            eosTokenId_ = eosId;
        }

        file.read( reinterpret_cast<char*>(&hasPad), sizeof( uint32_t ) );
        if ( hasPad ) {
            uint32_t padId;
            file.read( reinterpret_cast<char*>(&padId), sizeof( uint32_t ) );
            padTokenId_ = padId;
        }

        file.read( reinterpret_cast<char*>(&hasUnk), sizeof( uint32_t ) );
        if ( hasUnk ) {
            uint32_t unkId;
            file.read( reinterpret_cast<char*>(&unkId), sizeof( uint32_t ) );
            unkTokenId_ = unkId;
        }

        return true;
    }

    std::string LlamaTokenizer::normalizeText( const std::string& text ) const {
        // Llama adds a space prefix for proper tokenization
        return " " + text;
    }

    std::vector<TokenId> LlamaTokenizer::encode( const std::string& text ) {
        std::string normalized = normalizeText( text );
        return sentencePieceEncode( normalized );
    }

    std::vector<TokenId> LlamaTokenizer::encodeWithSpecial(
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

    std::vector<TokenId> LlamaTokenizer::sentencePieceEncode( const std::string& text ) const {
        std::vector<TokenId> result;
        size_t pos = 0;

        while ( pos < text.length() ) {
            // Greedy longest match
            bool found = false;

            for ( const auto& piece : pieces_ ) {
                if ( pos + piece.piece.length() <= text.length() &&
                    text.substr( pos, piece.piece.length() ) == piece.piece ) {
                    result.push_back( piece.id );
                    pos += piece.piece.length();
                    found = true;
                    break;
                }
            }

            if ( !found ) {
                // Byte fallback for unknown character
                if ( useByteFallback_ && pos < text.length() ) {
                    unsigned char byte = static_cast<unsigned char>( text[ pos ] );
                    // Byte tokens are typically at the end of vocab
                    // Format: <0xHH> where HH is hex
                    std::string byteToken = "<0x" +
                        std::to_string( byte / 16 ) +
                        std::to_string( byte % 16 ) + ">";

                    auto it = pieceToId_.find( byteToken );
                    if ( it != pieceToId_.end() ) {
                        result.push_back( it->second );
                    }
                    else if ( unkTokenId_ ) {
                        result.push_back( *unkTokenId_ );
                    }
                    pos++;
                }
                else {
                    // Unknown token
                    if ( unkTokenId_ ) {
                        result.push_back( *unkTokenId_ );
                    }
                    pos++;
                }
            }
        }

        return result;
    }

    std::string LlamaTokenizer::decode( std::span<const TokenId> tokens ) {
        std::string result;

        for ( auto tokenId : tokens ) {
            // Skip special tokens in decoding
            if ( tokenId == bosTokenId_ || tokenId == eosTokenId_ ||
                tokenId == padTokenId_ ) {
                continue;
            }

            auto it = idToPiece_.find( tokenId );
            if ( it != idToPiece_.end() ) {
                std::string piece = it->second;

                // Handle byte tokens
                if ( piece.starts_with( "<0x" ) && piece.ends_with( ">" ) ) {
                    // Convert hex byte back to character
                    std::string hexStr = piece.substr( 3, piece.length() - 4 );
                    int byte = std::stoi( hexStr, nullptr, 16 );
                    result += static_cast<char>(byte);
                }
                else {
                    // Replace ? (U+2581) with space for SentencePiece
                    if ( piece.starts_with( "?" ) ) {
                        result += ' ' + piece.substr( 3 );  // UTF-8 char is 3 bytes
                    }
                    else {
                        result += piece;
                    }
                }
            }
        }

        return result;
    }

    std::string LlamaTokenizer::tokenToString( TokenId tokenId ) const {
        auto it = idToPiece_.find( tokenId );
        return it != idToPiece_.end() ? it->second : "<UNK>";
    }

    bool LlamaTokenizer::isValidToken( TokenId tokenId ) const {
        return idToPiece_.contains( tokenId );
    }
}