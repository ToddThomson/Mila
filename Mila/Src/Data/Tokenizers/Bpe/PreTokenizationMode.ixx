/**
 * @file PreTokenizationMode.ixx
 * @brief Pre-tokenization mode enumeration for GPT-4 style BPE tokenizers.
 *
 * Extends the GPT-2 BPE pre-tokenization modes with the GPT-4 / Llama 3.x
 * regex pattern. This module is intentionally separate from the BpeTokenizer
 * variant to avoid any risk of breaking validated GPT-2 tokenization.
 */

module;

export module Data.Tokenizers.Bpe.PreTokenizationMode;

namespace Mila::Data
{
    /**
     * @brief Pre-tokenization strategies for GPT-4 style BPE tokenizers.
     */
    export enum class PreTokenizationMode
    {
        None,           // No pre-tokenization (byte level only)
        Whitespace,     // Simple whitespace splitting
        Gpt2Regex,      // GPT-2 style regex pattern
        Llama3Regex     // GPT-4 / TikToken pattern used by Llama 3.x
    };

    // =========================================================================
    // GPT-2 patterns (reproduced here for completeness — Gpt2Regex mode)
    // =========================================================================

    export constexpr const char* GPT2_PRETOKENIZATION_PATTERN =
        R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";

    export constexpr const char* GPT2_PRETOKENIZATION_PATTERN_ASCII_FALLBACK =
        R"('s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+)";

    // =========================================================================
    // Llama 3.x / GPT-4 patterns (Llama3Regex mode)
    //
    // Key differences from GPT-2:
    //  - Case-insensitive contractions: (?i:'[sdmt]|'ll|'ve|'re)
    //  - Newlines handled as a separate split category
    //  - Numbers chunked to max 3 digits: \p{N}{1,3}
    //    (prevents very long number strings from becoming single tokens)
    //  - Trailing whitespace before newlines is split out separately
    //
    // Note on std::regex and Unicode properties:
    //  \p{L} and \p{N} are NOT supported by MSVC's std::regex (ECMAScript mode).
    //  The ASCII fallback is used automatically when the Unicode pattern fails
    //  to compile. This is acceptable for Mila alpha.2 but means non-ASCII
    //  text (CJK, accented chars, etc.) may tokenize differently from HuggingFace.
    //  Track as a known gap for post-alpha work (consider RE2 or ICU regex).
    // =========================================================================

    export constexpr const char* LLAMA3_PRETOKENIZATION_PATTERN =
        R"((?i:'[sdmt]|'ll|'ve|'re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

    export constexpr const char* LLAMA3_PRETOKENIZATION_PATTERN_ASCII_FALLBACK =
        R"((?:'[sdmt]|'ll|'ve|'re)|[^\r\nA-Za-z0-9]?[A-Za-z]+|[0-9]{1,3}| ?[^\sA-Za-z0-9]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
}
