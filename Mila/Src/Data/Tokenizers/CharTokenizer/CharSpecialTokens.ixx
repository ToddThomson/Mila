/**
 * @file CharSpecialTokens.ixx
 * @brief Configuration for special tokens used in character-level tokenization.
 *
 * Provides individual token control, string definitions, validation, and factory methods.
 */

module;
#include <string>
#include <vector>
#include <unordered_set>
#include <stdexcept>

export module Data.CharSpecialTokens;

namespace Mila::Data
{
    /**
     * @brief Configuration for special tokens used in character-level tokenization.
     *
     * Individual flags control which tokens are included, with configurable string
     * representations for each token type.
     */
    export struct CharSpecialTokens
    {
        bool use_pad = true;
        bool use_unk = true;
        bool use_bos = true;
        bool use_eos = true;
        bool use_mask = false;
        bool use_sep = false;
        bool use_cls = false;

        std::string pad_token = "<PAD>";   ///< Padding token string
        std::string unk_token = "<UNK>";   ///< Unknown token string
        std::string bos_token = "<BOS>";   ///< Beginning of sequence string
        std::string eos_token = "<EOS>";   ///< End of sequence string
        std::string mask_token = "<MASK>"; ///< Mask token string (MLM)
        std::string sep_token = "<SEP>";   ///< Separator token string
        std::string cls_token = "<CLS>";   ///< Classification token string

        /**
         * @brief Get all enabled special tokens in priority order.
         *
         * @return Vector of special token strings for enabled tokens only.
         */
        std::vector<std::string> getAllTokens() const
        {
            std::vector<std::string> tokens;

            if ( use_pad ) tokens.push_back( pad_token );
            if ( use_unk ) tokens.push_back( unk_token );
            if ( use_bos ) tokens.push_back( bos_token );
            if ( use_eos ) tokens.push_back( eos_token );
            if ( use_mask ) tokens.push_back( mask_token );
            if ( use_sep ) tokens.push_back( sep_token );
            if ( use_cls ) tokens.push_back( cls_token );

            return tokens;
        }

        /**
         * @brief Count enabled special tokens.
         *
         * @return Number of special tokens that will be added to vocabulary.
         */
        size_t count() const
        {
            return (use_pad ? 1 : 0) + (use_unk ? 1 : 0) +
                   (use_bos ? 1 : 0) + (use_eos ? 1 : 0) +
                   (use_mask ? 1 : 0) + (use_sep ? 1 : 0) +
                   (use_cls ? 1 : 0);
        }

        /**
         * @brief Validate special tokens configuration.
         *
         * @throws std::invalid_argument if validation fails.
         */
        void validate() const
        {
            auto tokens = getAllTokens();

            if ( tokens.empty() )
            {
                return;
            }

            if ( use_pad && pad_token.empty() )
            {
                throw std::invalid_argument( "pad_token cannot be empty when use_pad=true" );
            }
            if ( use_unk && unk_token.empty() )
            {
                throw std::invalid_argument( "unk_token cannot be empty when use_unk=true" );
            }
            if ( use_bos && bos_token.empty() )
            {
                throw std::invalid_argument( "bos_token cannot be empty when use_bos=true" );
            }
            if ( use_eos && eos_token.empty() )
            {
                throw std::invalid_argument( "eos_token cannot be empty when use_eos=true" );
            }
            if ( use_mask && mask_token.empty() )
            {
                throw std::invalid_argument( "mask_token cannot be empty when use_mask=true" );
            }
            if ( use_sep && sep_token.empty() )
            {
                throw std::invalid_argument( "sep_token cannot be empty when use_sep=true" );
            }
            if ( use_cls && cls_token.empty() )
            {
                throw std::invalid_argument( "cls_token cannot be empty when use_cls=true" );
            }

            std::unordered_set<std::string> unique_tokens( tokens.begin(), tokens.end() );
            if ( unique_tokens.size() != tokens.size() )
            {
                throw std::invalid_argument( "Special tokens must be unique" );
            }
        }

        /**
         * @brief Check if a string is an enabled special token.
         *
         * @param token String to check.
         * @return true if token matches an enabled special token.
         */
        bool isSpecialToken( const std::string& token ) const
        {
            return (use_pad && token == pad_token) ||
                   (use_unk && token == unk_token) ||
                   (use_bos && token == bos_token) ||
                   (use_eos && token == eos_token) ||
                   (use_mask && token == mask_token) ||
                   (use_sep && token == sep_token) ||
                   (use_cls && token == cls_token);
        }

        /**
         * @brief Get the ID offset for regular tokens.
         *
         * Special tokens occupy IDs 0 to (count()-1), so regular tokens
         * start at this offset.
         *
         * @return Number of special tokens.
         */
        size_t getIdOffset() const
        {
            return count();
        }

        /**
         * @brief Create standard character-level special tokens.
         *
         * Includes PAD, UNK, BOS, EOS.
         */
        static CharSpecialTokens standard()
        {
            return CharSpecialTokens{
                .use_pad = true,
                .use_unk = true,
                .use_bos = true,
                .use_eos = true
            };
        }

        /**
         * @brief Create minimal special tokens (PAD and UNK only).
         *
         * Common for inference or when sequence markers aren't needed.
         */
        static CharSpecialTokens minimal()
        {
            return CharSpecialTokens{
                .use_pad = true,
                .use_unk = true,
                .use_bos = false,
                .use_eos = false
            };
        }

        /**
         * @brief Create special tokens for masked language modeling.
         *
         * Includes standard tokens plus MASK.
         */
        static CharSpecialTokens forMLM()
        {
            return CharSpecialTokens{
                .use_pad = true,
                .use_unk = true,
                .use_bos = true,
                .use_eos = true,
                .use_mask = true
            };
        }

        /**
         * @brief Create special tokens for sequence classification.
         *
         * Includes standard tokens plus SEP and CLS.
         */
        static CharSpecialTokens forClassification()
        {
            return CharSpecialTokens{
                .use_pad = true,
                .use_unk = true,
                .use_bos = true,
                .use_eos = true,
                .use_sep = true,
                .use_cls = true
            };
        }

        /**
         * @brief Create configuration with no special tokens.
         */
        static CharSpecialTokens none()
        {
            return CharSpecialTokens{
                .use_pad = false,
                .use_unk = false,
                .use_bos = false,
                .use_eos = false
            };
        }
    };
}