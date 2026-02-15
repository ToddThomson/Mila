module;

export module Data.PreTokenizationMode;

namespace Mila::Data
{
    export enum class PreTokenizationMode
    {
        None,           // No pre-tokenization (byte/char level only)
        Whitespace,     // Simple whitespace splitting
        Gpt2Regex       // GPT-2 style regex pattern
    };

    // GPT-2 pre-tokenization patterns
    export constexpr const char* GPT2_PRETOKENIZATION_PATTERN =
        R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";

    export constexpr const char* GPT2_PRETOKENIZATION_PATTERN_ASCII_FALLBACK =
        R"('s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+)";
}