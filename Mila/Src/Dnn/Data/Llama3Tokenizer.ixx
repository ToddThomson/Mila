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
        static std::unique_ptr<LlamaTokenizer> fromFile( const std::string& path );

        std::vector<TokenId> encode( const std::string& text ) override;
        std::string decode( std::span<const TokenId> tokens ) override;

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

        bool loadFromBinary( const std::string& path );

        std::vector<TokenId> sentencePieceEncode( const std::string& text ) const;
        std::string normalizeText( const std::string& text ) const;

        struct SpmPiece {
            std::string piece;
            float score;
            TokenId id;
        };

        std::vector<SpmPiece> pieces_;
        std::unordered_map<std::string, TokenId> pieceToId_;
        std::unordered_map<TokenId, std::string> idToPiece_;

        std::optional<TokenId> bosTokenId_;
        std::optional<TokenId> eosTokenId_;
        std::optional<TokenId> padTokenId_;
        std::optional<TokenId> unkTokenId_;

        size_t vocabSize_{ 0 };
        bool useByteFallback_{ true };
    };

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

        uint32_t vocabSize;
        file.read( reinterpret_cast<char*>(&vocabSize), sizeof( uint32_t ) );
        vocabSize_ = vocabSize;

        uint8_t useByteFallback;
        file.read( reinterpret_cast<char*>(&useByteFallback), sizeof( uint8_t ) );
        useByteFallback_ = useByteFallback != 0;

        pieces_.reserve( vocabSize );

        for ( uint32_t i = 0; i < vocabSize; ++i ) {
            uint32_t pieceLen;
            file.read( reinterpret_cast<char*>( &pieceLen ), sizeof( uint32_t ) );

            std::string piece( pieceLen, '\0' );
            file.read( piece.data(), pieceLen );

            float score;
            file.read( reinterpret_cast<char*>( &score ), sizeof( float ) );

            int32_t tokenId;
            file.read( reinterpret_cast<char*>(&tokenId), sizeof( int32_t ) );

            pieces_.push_back( { piece, score, tokenId } );
            pieceToId_[ piece ] = tokenId;
            idToPiece_[ tokenId ] = piece;
        }

        std::sort( pieces_.begin(), pieces_.end(),
            []( const SpmPiece& a, const SpmPiece& b ) {
                return a.score > b.score;
            } );

        uint32_t hasBos, hasEos, hasPad, hasUnk;

        file.read( reinterpret_cast<char*>(&hasBos), sizeof( uint32_t ) );
        if ( hasBos ) {
            int32_t bosId;
            file.read( reinterpret_cast<char*>(&bosId), sizeof( int32_t ) );
            bosTokenId_ = bosId;
        }

        file.read( reinterpret_cast<char*>(&hasEos), sizeof( uint32_t ) );
        if ( hasEos ) {
            int32_t eosId;
            file.read( reinterpret_cast<char*>(&eosId), sizeof( int32_t ) );
            eosTokenId_ = eosId;
        }

        file.read( reinterpret_cast<char*>(&hasPad), sizeof( uint32_t ) );
        if ( hasPad ) {
            int32_t padId;
            file.read( reinterpret_cast<char*>(&padId), sizeof( int32_t ) );
            padTokenId_ = padId;
        }

        file.read( reinterpret_cast<char*>(&hasUnk), sizeof( uint32_t ) );
        if ( hasUnk ) {
            int32_t unkId;
            file.read( reinterpret_cast<char*>(&unkId), sizeof( int32_t ) );
            unkTokenId_ = unkId;
        }

        return true;
    }

    std::string LlamaTokenizer::normalizeText( const std::string& text ) const {
        return " " + text;
    }

    std::vector<TokenId> LlamaTokenizer::encode( const std::string& text ) {
        std::string normalized = normalizeText( text );
        return sentencePieceEncode( normalized );
    }

    std::vector<TokenId> LlamaTokenizer::sentencePieceEncode( const std::string& text ) const {
        std::vector<TokenId> result;
        size_t pos = 0;

        while ( pos < text.length() ) {
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
                if ( useByteFallback_ && pos < text.length() ) {
                    unsigned char byte = static_cast<unsigned char>( text[ pos ] );

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
            if ( tokenId == bosTokenId_ || tokenId == eosTokenId_ ||
                tokenId == padTokenId_ ) {
                continue;
            }

            auto it = idToPiece_.find( tokenId );
            if ( it != idToPiece_.end() ) {
                std::string piece = it->second;

                if ( piece.starts_with( "<0x" ) && piece.ends_with( ">" ) ) {
                    std::string hexStr = piece.substr( 3, piece.length() - 4 );
                    int byte = std::stoi( hexStr, nullptr, 16 );
                    result += static_cast<char>(byte);
                }
                else {
                    if ( piece.starts_with( "?" ) ) {
                        result += ' ' + piece.substr( 3 );
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