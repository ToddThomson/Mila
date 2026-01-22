module;
#include <cstdint>
#include <string_view>
#include <string>

export module Data.TokenizerType;

namespace Mila::Dnn::Data
{
    /** Tokenizer type discriminator used across tokenizer and vocabulary types. */
    export enum class TokenizerType : uint8_t {
        Unknown = 0,
        Char,
        Bpe,
        SentencePiece,
        Word,
        Unigram
    };

    /** Convert TokenizerType to a stable string representation. */
    export inline std::string_view to_string( TokenizerType t ) noexcept
    {
        switch ( t ) {
            case TokenizerType::Char: return "char";
            case TokenizerType::Bpe: return "bpe";
            case TokenizerType::SentencePiece: return "sentencepiece";
            case TokenizerType::Word: return "word";
            case TokenizerType::Unigram: return "unigram";
            default: return "unknown";
        }
    }

    /** Parse a string into TokenizerType. Comparison is case-insensitive for the ASCII range. */
    export inline TokenizerType from_string( std::string_view s ) noexcept
    {
        if ( s.empty() ) return TokenizerType::Unknown;

        // Normalize ASCII lower-case on the fly for common strings
        auto eq_icase = []( std::string_view a, std::string_view b ) noexcept {
            if ( a.size() != b.size() ) return false;
            for ( size_t i = 0; i < a.size(); ++i ) {
                char ca = a[ i ];
                if ( ca >= 'A' && ca <= 'Z' ) ca = static_cast<char>( ca - 'A' + 'a' );
                if ( ca != b[ i ] ) return false;
            }
            return true;
            };

        if ( eq_icase( s, "char" ) ) return TokenizerType::Char;
        if ( eq_icase( s, "bpe" ) ) return TokenizerType::Bpe;
        if ( eq_icase( s, "sentencepiece" ) ) return TokenizerType::SentencePiece;
        if ( eq_icase( s, "word" ) ) return TokenizerType::Word;
        if ( eq_icase( s, "unigram" ) ) return TokenizerType::Unigram;
        return TokenizerType::Unknown;
    }
}