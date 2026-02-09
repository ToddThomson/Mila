/**
 * @file SpecialTokens.ixx
 * @brief Configuration for special tokens used across all tokenizer types.
 */

module;
#include <string>
#include <string_view>
#include <vector>
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
     * Used by CharTokenizer, BpeTokenizer, WordPieceTokenizer, etc.
     * Token strings are customizable to support different model conventions.
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
         * @brief Get string representation of a special token.
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
         * @brief Check if a token type is enabled.
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
         * @brief Get all enabled special tokens in priority order.
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
         * @brief Count enabled special tokens.
         */
        constexpr size_t count() const
        {
            return (use_pad ? 1 : 0) + (use_unk ? 1 : 0) + (use_bos ? 1 : 0) +
                (use_eos ? 1 : 0) + (use_mask ? 1 : 0) + (use_sep ? 1 : 0) +
                (use_cls ? 1 : 0);
        }

        /**
         * @brief Check if a string matches an enabled special token.
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
            
            return false;
        }

        /**
         * @brief Get the ID offset for regular tokens.
         *
         * Special tokens occupy IDs 0 to (count()-1), so regular tokens
         * start at this offset.
         */
        constexpr size_t getIdOffset() const
        {
            return count();
        }

        // Factory methods

        /**
         * @brief Standard configuration (PAD, UNK, BOS, EOS).
         *
         * Common for most sequence-to-sequence tasks.
         */
        static constexpr SpecialTokens standard()
        {
            return SpecialTokens{
                .use_pad = true, .use_unk = true, .use_bos = true, .use_eos = true
            };
        }

        /**
         * @brief Minimal configuration (PAD, UNK only).
         *
         * Common for inference or when sequence markers aren't needed.
         */
        static constexpr SpecialTokens minimal()
        {
            return SpecialTokens{ .use_pad = true, .use_unk = true };
        }

        /**
         * @brief Configuration for masked language modeling.
         *
         * Includes standard tokens plus MASK.
         */
        static constexpr SpecialTokens forMLM()
        {
            return SpecialTokens{
                .use_pad = true, .use_unk = true, .use_bos = true,
                .use_eos = true, .use_mask = true
            };
        }

        /**
         * @brief Configuration for sequence classification (BERT-style).
         *
         * Includes standard tokens plus SEP and CLS.
         */
        static constexpr SpecialTokens forClassification()
        {
            return SpecialTokens{
                .use_pad = true, .use_unk = true, .use_bos = true,
                .use_eos = true, .use_sep = true, .use_cls = true
            };
        }

        /**
         * @brief GPT-style configuration using <|endoftext|> for multiple purposes.
         *
         * GPT models use the same token string for PAD, UNK, BOS, and EOS.
         */
        static SpecialTokens gptStyle()
        {
            return SpecialTokens{
                .use_pad = true, .use_unk = true, .use_bos = true, .use_eos = true,
                .pad_token = "<|endoftext|>",
                .unk_token = "<|endoftext|>",
                .bos_token = "<|endoftext|>",
                .eos_token = "<|endoftext|>"
            };
        }

        /**
         * @brief Configuration with no special tokens.
         *
         * Rare, but useful for pre-tokenized data or specific research tasks.
         */
        static constexpr SpecialTokens none()
        {
            return SpecialTokens{
                .use_pad = false, .use_unk = false, .use_bos = false,
                .use_eos = false, .use_mask = false, .use_sep = false, .use_cls = false
            };
        }
    };
}