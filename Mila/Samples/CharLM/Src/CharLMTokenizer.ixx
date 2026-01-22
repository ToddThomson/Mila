module;
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <cstdint>
#include <stdexcept>
#include <optional>
#include <span>

export module CharLM.Tokenizer;

import Mila;

namespace CharLM
{
    using Mila::Dnn::Data::Tokenizer;
    using Mila::Dnn::Data::TokenId;

    /**
     * @file CharLMTokenizer.ixx
     * @brief Character-level Tokenizer implementation for CharLM sample.
     *
     * Loads vocabulary from a binary .vocab file (header + length-prefixed token strings).
     */

    export class CharLMTokenizer : public Tokenizer
    {
    public:
        explicit CharLMTokenizer( const std::string& vocab_file )
        {
            std::ifstream file( vocab_file, std::ios::binary );
            if ( !file )
            {
                throw std::runtime_error( "Cannot open vocabulary file: " + vocab_file );
            }

            // Read vocab size
            size_t vs = 0;
            file.read( reinterpret_cast<char*>(&vs), sizeof( vs ) );
            if ( !file || vs == 0 )
            {
                throw std::runtime_error( "Invalid or empty vocabulary file: " + vocab_file );
            }

            vocab_size_ = vs;

            // Read has_special flag
            bool has_special = false;
            file.read( reinterpret_cast<char*>(&has_special), sizeof( has_special ) );
            if ( !file )
            {
                // If flag not present treat as no special tokens
                file.clear();
                has_special = false;
            }

            if ( has_special )
            {
                int pad = -1;
                int unk = -1;
                file.read( reinterpret_cast<char*>(&pad), sizeof( pad ) );
                if ( !file ) throw std::runtime_error( "Error reading pad token id from: " + vocab_file );
                file.read( reinterpret_cast<char*>(&unk), sizeof( unk ) );
                if ( !file ) throw std::runtime_error( "Error reading unk token id from: " + vocab_file );

                pad_token_id_ = (pad >= 0) ? std::optional<TokenId>( static_cast<TokenId>(pad) ) : std::nullopt;
                unk_token_id_ = (unk >= 0) ? std::optional<TokenId>( static_cast<TokenId>(unk) ) : std::nullopt;
            }
            else
            {
                pad_token_id_.reset();
                unk_token_id_.reset();
            }

            // Read token strings: each token is stored as uint32_t length + bytes (no null terminator)
            id_to_token_.reserve( vocab_size_ );
            token_to_id_.reserve( vocab_size_ * 2 );

            for ( size_t i = 0; i < vocab_size_; ++i )
            {
                uint32_t len = 0;
                file.read( reinterpret_cast<char*>( &len ), sizeof( len ) );
                if ( !file )
                {
                    throw std::runtime_error( "Error reading token length from vocab file: " + vocab_file );
                }

                std::string token;
                token.resize( len );
                if ( len > 0 )
                {
                    file.read( token.data(), len );
                    if ( !file )
                    {
                        throw std::runtime_error( "Error reading token bytes from vocab file: " + vocab_file );
                    }
                }

                id_to_token_.push_back( token );
                token_to_id_.emplace( token, static_cast<TokenId>(i) );
            }
        }

        ~CharLMTokenizer() override = default;

        std::vector<TokenId> encode( const std::string& text ) override
        {
            std::vector<TokenId> out;
            out.reserve( text.size() );

            for ( char c : text )
            {
                std::string key( 1, c );
                auto it = token_to_id_.find( key );
                if ( it != token_to_id_.end() )
                {
                    out.push_back( it->second );
                }
                else if ( unk_token_id_.has_value() )
                {
                    out.push_back( unk_token_id_.value() );
                }
                else
                {
                    throw std::runtime_error( "CharLMTokenizer::encode - unknown token and no UNK defined: " + key );
                }
            }

            return out;
        }

        std::string decode( std::span<const TokenId> tokens ) override
        {
            std::string out;
            out.reserve( tokens.size() );

            for ( TokenId t : tokens )
            {
                if ( t < id_to_token_.size() )
                {
                    out += id_to_token_[ t ];
                }
                else
                {
                    // Out-of-range token: represent as '?'
                    out += '?';
                }
            }

            return out;
        }

        std::vector<TokenId> encodeWithSpecial(
            const std::string& text,
            bool addBos = true,
            bool addEos = true ) override
        {
            std::vector<TokenId> encoded;
            if ( addBos && bos_token_id_.has_value() )
            {
                encoded.push_back( bos_token_id_.value() );
            }

            auto core = encode( text );
            encoded.insert( encoded.end(), core.begin(), core.end() );

            if ( addEos && eos_token_id_.has_value() )
            {
                encoded.push_back( eos_token_id_.value() );
            }

            return encoded;
        }

        size_t getVocabSize() const override
        {
            return vocab_size_;
        }

        std::optional<TokenId> getBosTokenId() const override
        {
            return bos_token_id_;
        }

        std::optional<TokenId> getEosTokenId() const override
        {
            return eos_token_id_;
        }

        std::optional<TokenId> getPadTokenId() const override
        {
            return pad_token_id_;
        }

        std::string tokenToString( TokenId tokenId ) const override
        {
            if ( tokenId < id_to_token_.size() )
            {
                return id_to_token_[ tokenId ];
            }

            return std::string( "<?>" );
        }

        bool isValidToken( TokenId tokenId ) const override
        {
            return tokenId < static_cast<TokenId>( vocab_size_ );
        }

        // Optionally allow setting BOS/EOS post-construction if the preprocessor stores them separately
        void setBosTokenId( std::optional<TokenId> id ) {
            bos_token_id_ = id;
        }
        void setEosTokenId( std::optional<TokenId> id ) {
            eos_token_id_ = id;
        }

        // Expose token-file loader so callers (DataLoader / preprocessor) can reuse tokenizer logic.
        std::vector<TokenId> loadTokensFromFile( const std::string& tokensFile ) const
        {
            std::ifstream file( tokensFile, std::ios::binary );
            if ( !file ) throw std::runtime_error( "Cannot open tokens file: " + tokensFile );

            size_t num_tokens = 0;
            file.read( reinterpret_cast<char*>(&num_tokens), sizeof( num_tokens ) );
            if ( num_tokens == 0 ) throw std::runtime_error( "Empty tokens file: " + tokensFile );

            std::vector<int32_t> tmp( num_tokens );
            file.read( reinterpret_cast<char*>(tmp.data()), num_tokens * sizeof( int32_t ) );
            if ( !file ) throw std::runtime_error( "Error reading tokens file: " + tokensFile );

            std::vector<TokenId> out;
            out.reserve( num_tokens );
            for ( size_t i = 0; i < tmp.size(); ++i )
            {
                int32_t s = tmp[ i ];
                if ( s < 0 ) throw std::runtime_error( "Negative token in file at pos " + std::to_string( i ) );
                TokenId t = static_cast<TokenId>( s );
                
                if ( !isValidToken( t ) ) 
                    throw std::runtime_error( "Invalid token " + std::to_string( t ) + " at pos " + std::to_string( i ) );
                
                out.push_back( t );
            }
            return out;
        }

    private:
        size_t vocab_size_{ 0 };
        std::vector<std::string> id_to_token_;
        std::unordered_map<std::string, TokenId> token_to_id_;

        std::optional<TokenId> bos_token_id_;
        std::optional<TokenId> eos_token_id_;
        std::optional<TokenId> pad_token_id_;
        std::optional<TokenId> unk_token_id_;
    };
}