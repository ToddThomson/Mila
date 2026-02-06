module;
#include <string>
#include <vector>
#include <cstdint>
#include <optional>
#include <span>
#include <filesystem>

export module Data.Tokenizer;

namespace Mila::Dnn::Data
{
    // Factory function - loads any tokenizer type
    //std::unique_ptr<Tokenizer> loadTokenizer( const std::filesystem::path& vocabulary_path );

    // Semantically, Token Ids are unsigned. However, we use int* in my CUDA Encoder kenels
    // For now we 'll use int32_t as the TokenId type
    export using TokenId = int32_t;

    export class Tokenizer {
    public:
        virtual ~Tokenizer() = default;

        // Encode text to token IDs
        virtual std::vector<TokenId> encode( const std::string& text ) = 0;

        // Decode token IDs back to text
        virtual std::string decode( std::span<const TokenId> tokens ) = 0;

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