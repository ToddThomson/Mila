/**
 * @file BpePreTokenizationMode.ixx
 * @brief Pre-tokenization mode enumeration for GPT-4 style BPE tokenizers.
 *
 * Extends the GPT-2 BPE pre-tokenization modes with the GPT-4 / Llama 3.x
 * regex pattern. This module is intentionally separate from the BpeTokenizer
 * variant to avoid any risk of breaking validated GPT-2 tokenization.
 */

module;

export module Data.BpePreTokenizationMode;

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
        Llama3Regex,    // GPT-4 / TikToken pattern used by Llama 3.x
        Qwen3Regex,     // Qwen 3.x pattern: single-digit numbers, marks join the letter run
        SentencePiece   // SentencePiece Metaspace: space -> U+2581, split at marks (Gemma)
    };

    // =========================================================================
    // GPT-2 patterns (reproduced here for completeness -- Gpt2Regex mode)
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

    // =========================================================================
    // Qwen 3.x patterns (Qwen3Regex mode)
    //
    // Verbatim from the Qwen 3.8 checkpoint's tokenizer.json. Two differences
    // from Llama 3, of which only the second changes any output:
    //  - \p{N} matches ONE digit, where Llama 3 chunks up to three. Inert for
    //    this vocabulary: it holds no multi-digit piece and no digit-digit merge
    //    rule, so the merge loop cannot join digits however they were grouped.
    //  - [\p{L}\p{M}]+ admits combining marks into the letter run (and the
    //    punctuation class excludes them to match). The vocabulary contains
    //    base+mark pieces for Devanagari, Thai and Arabic that only this form
    //    can produce, so Llama's pattern shortens those scripts to roughly half
    //    the pieces per token -- silently, since both decode back to the input.
    //
    // Note: `transformers` 5.12.1 rebuilds Qwen2/3 tokenizers from a hardcoded
    // Qwen2-era pattern lacking \p{M} (tokenization_qwen2.py:33), so ids taken
    // from AutoTokenizer disagree with the checkpoint on those scripts. The
    // checkpoint is the reference here; see Specifications/Qwen3.8.md.
    // =========================================================================

    export constexpr const char* QWEN3_PRETOKENIZATION_PATTERN =
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

    // \p{M} has no ASCII counterpart, so the fallback drops it entirely -- as it
    // drops every other Unicode property. Which means that under the fallback this
    // pattern and Llama 3's produce identical pretokens on ASCII input: the only
    // remaining difference is the digit rule, and that one is inert here. The
    // divergence between the two families is reachable ONLY through the Unicode
    // form, so an ASCII-only test cannot tell them apart.
    export constexpr const char* QWEN3_PRETOKENIZATION_PATTERN_ASCII_FALLBACK =
        R"((?:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\nA-Za-z0-9]?[A-Za-z]+|[0-9]| ?[^\sA-Za-z0-9]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
}
