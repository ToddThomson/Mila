/**
 * @file SpecialTokens.ixx
 * @brief Configuration for special tokens used across all tokenizer types.
 */

module;
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <cstdint>

export module Data.SpecialTokens;

namespace Mila::Data
{
    /**
     * @brief Special token types for tokenization.
     */
    export enum class SpecialToken : uint8_t
    {
        PAD = 0,
        UNK,
        BOS,
        EOS,
        MASK,
        SEP,
        CLS
    };

    /**
     * @brief Configuration for special tokens across all tokenizer types.
     *
     * Used by CharTokenizer, BpeTokenizer, Gpt4BpeTokenizer, WordPieceTokenizer, etc.
     * Token strings are customizable to support different model conventions.
     *
     * The seven named slots (PAD, UNK, BOS, EOS, MASK, SEP, CLS) cover the
     * common case for all known model families. For models with additional
     * special tokens beyond these seven (e.g. Llama 3.2's 256 reserved tokens),
     * use the extended_special_tokens map.
     */
    export struct SpecialTokens
    {
        bool use_pad = true;
        bool use_unk = true;
        bool use_bos = true;
        bool use_eos = true;
        bool use_mask = false;
        bool use_sep = false;
        bool use_cls = false;

        std::string pad_token = "<PAD>";
        std::string unk_token = "<UNK>";
        std::string bos_token = "<BOS>";
        std::string eos_token = "<EOS>";
        std::string mask_token = "<MASK>";
        std::string sep_token = "<SEP>";
        std::string cls_token = "<CLS>";

        /**
         * @brief Extended special tokens beyond the seven named slots.
         *
         * Used for model families with large special token sets, such as
         * Llama 3.2's reserved tokens (IDs 128002-128255). These are matched
         * during the encode pre-pass before BPE merges are applied.
         *
         * Key: token string (e.g. "<|reserved_special_token_0|>")
         * Value: token ID
         */
        std::unordered_map<std::string, int32_t> extended_special_tokens;

        /**
         * @brief Get string representation of a named special token.
         */
        std::string_view getString( SpecialToken token ) const
        {
            switch ( token )
            {
                case SpecialToken::PAD:  return pad_token;
                case SpecialToken::UNK:  return unk_token;
                case SpecialToken::BOS:  return bos_token;
                case SpecialToken::EOS:  return eos_token;
                case SpecialToken::MASK: return mask_token;
                case SpecialToken::SEP:  return sep_token;
                case SpecialToken::CLS:  return cls_token;
            }
            return "";
        }

        /**
         * @brief Check if a named token type is enabled.
         */
        constexpr bool isEnabled( SpecialToken token ) const
        {
            switch ( token )
            {
                case SpecialToken::PAD:  return use_pad;
                case SpecialToken::UNK:  return use_unk;
                case SpecialToken::BOS:  return use_bos;
                case SpecialToken::EOS:  return use_eos;
                case SpecialToken::MASK: return use_mask;
                case SpecialToken::SEP:  return use_sep;
                case SpecialToken::CLS:  return use_cls;
            }
            return false;
        }

        /**
         * @brief Get all enabled named tokens in priority order.
         */
        std::vector<SpecialToken> getEnabledTokens() const
        {
            std::vector<SpecialToken> tokens;
            tokens.reserve( 7 );

            constexpr SpecialToken all[] = {
                SpecialToken::PAD, SpecialToken::UNK, SpecialToken::BOS, SpecialToken::EOS,
                SpecialToken::MASK, SpecialToken::SEP, SpecialToken::CLS
            };

            for ( auto token : all )
            {
                if ( isEnabled( token ) )
                {
                    tokens.push_back( token );
                }
            }

            return tokens;
        }

        /**
         * @brief Count enabled named special tokens.
         * Does not include extended_special_tokens.
         */
        constexpr size_t count() const
        {
            return (use_pad ? 1 : 0) + (use_unk ? 1 : 0) + (use_bos ? 1 : 0) +
                (use_eos ? 1 : 0) + (use_mask ? 1 : 0) + (use_sep ? 1 : 0) +
                (use_cls ? 1 : 0);
        }

        /**
         * @brief Count all special tokens including extended set.
         */
        size_t countAll() const
        {
            return count() + extended_special_tokens.size();
        }

        /**
         * @brief Check if a string matches any enabled special token (named or extended).
         */
        bool isSpecialToken( std::string_view str ) const
        {
            if ( use_pad && str == pad_token ) return true;
            if ( use_unk && str == unk_token ) return true;
            if ( use_bos && str == bos_token ) return true;
            if ( use_eos && str == eos_token ) return true;
            if ( use_mask && str == mask_token ) return true;
            if ( use_sep && str == sep_token ) return true;
            if ( use_cls && str == cls_token ) return true;

            // Check extended tokens
            return extended_special_tokens.find( std::string( str ) ) != extended_special_tokens.end();
        }

        /**
         * @brief Get the ID offset for regular tokens.
         *
         * Named special tokens occupy IDs 0 to (count()-1), so regular tokens
         * start at this offset. Extended tokens have explicit IDs and do not
         * contribute to this offset.
         */
        constexpr size_t getIdOffset() const
        {
            return count();
        }

        // ====================================================================
        // Factory methods
        // ====================================================================

        /**
         * @brief Standard configuration (PAD, UNK, BOS, EOS).
         */
        static SpecialTokens standard()
        {
            return SpecialTokens{
                .use_pad = true, .use_unk = true, .use_bos = true, .use_eos = true
            };
        }

        /**
         * @brief Minimal configuration (PAD, UNK only).
         */
        static SpecialTokens minimal()
        {
            return SpecialTokens{ .use_pad = true, .use_unk = true };
        }

        /**
         * @brief Configuration for masked language modeling.
         */
        static SpecialTokens forMLM()
        {
            return SpecialTokens{
                .use_pad = true, .use_unk = true, .use_bos = true,
                .use_eos = true, .use_mask = true
            };
        }

        /**
         * @brief Configuration for sequence classification (BERT-style).
         */
        static SpecialTokens forClassification()
        {
            return SpecialTokens{
                .use_pad = true, .use_unk = true, .use_bos = true,
                .use_eos = true, .use_sep = true, .use_cls = true
            };
        }

        /**
         * @brief GPT-2 style configuration.
         *
         * Uses <|endoftext|> for PAD, UNK, BOS, and EOS — GPT-2 uses one
         * token string for all roles.
         */
        static SpecialTokens gptStyle()
        {
            return SpecialTokens{
                .use_pad = true, .use_unk = true, .use_bos = true, .use_eos = true,
                .use_mask = false, .use_sep = false, .use_cls = false,
                .pad_token = "<|endoftext|>",
                .unk_token = "<|endoftext|>",
                .bos_token = "<|endoftext|>",
                .eos_token = "<|endoftext|>"
            };
        }

        /**
         * @brief Llama 3.x style configuration.
         *
         * BOS: <|begin_of_text|> (ID 128000)
         * EOS: <|end_of_text|>   (ID 128001)
         *
         * No PAD or UNK — Llama 3.x uses byte fallback for unknown bytes
         * and does not use a dedicated padding token.
         *
         * The 254 reserved special tokens (IDs 128002-128255) are not
         * populated here as they are unused in standard inference. Add them
         * to extended_special_tokens if fine-tuning control tokens are needed.
         */
        static SpecialTokens llamaStyle()
        {
            return SpecialTokens{
                .use_pad = false, .use_unk = false, .use_bos = true, .use_eos = true,
                .use_mask = false, .use_sep = false, .use_cls = false,
                .bos_token = "<|begin_of_text|>",
                .eos_token = "<|end_of_text|>"
            };
        }

        /**
         * @brief Configuration with no special tokens.
         */
        static SpecialTokens none()
        {
            return SpecialTokens{
                .use_pad = false, .use_unk = false, .use_bos = false,
                .use_eos = false, .use_mask = false, .use_sep = false, .use_cls = false
            };
        }
    };
}
