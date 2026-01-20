module;
#include <string>
#include <vector>
#include <cstdint>
#include <optional>
#include <span>

export module Data.Tokenizer;


namespace Mila::Data
{
    // Token ID type
    export using TokenId = uint32_t;

    export class Tokenizer {
    public:
        virtual ~Tokenizer() = default;

        // Encode text to token IDs
        virtual std::vector<TokenId> encode( const std::string& text ) = 0;

        // Decode token IDs back to text
        virtual std::string decode( std::span<const TokenId> tokens ) = 0;

        // Encode with special tokens (BOS/EOS)
        virtual std::vector<TokenId> encodeWithSpecial(
            const std::string& text,
            bool addBos = true,
            bool addEos = true
        ) = 0;

        // Get vocabulary size
        virtual size_t getVocabSize() const = 0;

        // Get special token IDs
        virtual std::optional<TokenId> getBosTokenId() const = 0;
        virtual std::optional<TokenId> getEosTokenId() const = 0;
        virtual std::optional<TokenId> getPadTokenId() const = 0;

        // Convert token ID to string (for debugging)
        virtual std::string tokenToString( TokenId tokenId ) const = 0;

        // Check if token ID is valid
        virtual bool isValidToken( TokenId tokenId ) const = 0;
    };

}