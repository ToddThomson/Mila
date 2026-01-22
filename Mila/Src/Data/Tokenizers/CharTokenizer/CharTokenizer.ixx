module;
#include <string>
#include <string_view>
#include <vector>
#include <span>

export module Data.CharTokenizer;

import Data.Tokenizer;
import Data.Vocabulary;

namespace Mila::Data
{
    // Factory function - loads any tokenizer type
    //std::unique_ptr<Tokenizer> loadTokenizer( const std::filesystem::path& vocab_path );

    export class CharTokenizer : public Tokenizer {
    public:
        explicit CharTokenizer( Vocabulary vocab );

        std::vector<uint32_t> encode( std::string_view text ) const override;
        std::string decode( std::span<const uint32_t> tokens ) const override;

    private:
        Vocabulary vocab_;
    };