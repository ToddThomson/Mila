module;
#include <string>
#include <vector>
#include <unordered_set>
#include <stdexcept>

export module Data.BpeSpecialTokens;

namespace Mila::Data
{
    /**
     * @brief Configuration for special tokens used in BPE tokenization.
     *
     * Special tokens are reserved vocabulary entries with specific semantic meaning
     * used during training, inference, or data processing. Unlike character-level
     * tokenizers, BPE works with subword strings where the actual token strings matter.
     */
    export struct BpeSpecialTokens {
        bool enabled = true;

        std::string pad_token = "<|pad|>";   ///< Padding token for sequence alignment
        std::string unk_token = "<|unk|>";   ///< Unknown token for OOV words
        std::string bos_token = "<|bos|>";   ///< Beginning of sequence marker
        std::string eos_token = "<|eos|>";   ///< End of sequence marker

        // Optional extended tokens (empty string = disabled)
        std::string mask_token = "";         ///< Mask token for MLM (e.g., "<|mask|>")
        std::string sep_token = "";          ///< Separator for multi-sequence (e.g., "<|sep|>")
        std::string cls_token = "";          ///< Classification token (e.g., "<|cls|>")

        /**
         * @brief Get all enabled special tokens in priority order.
         * @return Vector of special token strings (empty strings are excluded)
         */
        std::vector<std::string> getAllTokens() const {
            std::vector<std::string> tokens;

            if ( !enabled ) {
                return tokens;
            }

            // Core tokens (always included when enabled=true)
            if ( !pad_token.empty() ) tokens.push_back( pad_token );
            if ( !unk_token.empty() ) tokens.push_back( unk_token );
            if ( !bos_token.empty() ) tokens.push_back( bos_token );
            if ( !eos_token.empty() ) tokens.push_back( eos_token );

            // Extended tokens (only if not empty)
            if ( !mask_token.empty() ) tokens.push_back( mask_token );
            if ( !sep_token.empty() ) tokens.push_back( sep_token );
            if ( !cls_token.empty() ) tokens.push_back( cls_token );

            return tokens;
        }

        /**
         * @brief Count enabled special tokens.
         * @return Number of non-empty special tokens when enabled=true, 0 otherwise
         */
        size_t count() const {
            if ( !enabled ) {
                return 0;
            }
            return getAllTokens().size();
        }

        /**
         * @brief Validate special tokens configuration.
         * @throws std::invalid_argument if validation fails
         */
        void validate() const {
            if ( !enabled ) {
                return;
            }

            auto tokens = getAllTokens();

            // Check for empty core tokens
            if ( pad_token.empty() ) {
                throw std::invalid_argument( "pad_token cannot be empty when enabled=true" );
            }
            if ( unk_token.empty() ) {
                throw std::invalid_argument( "unk_token cannot be empty when enabled=true" );
            }
            if ( bos_token.empty() ) {
                throw std::invalid_argument( "bos_token cannot be empty when enabled=true" );
            }
            if ( eos_token.empty() ) {
                throw std::invalid_argument( "eos_token cannot be empty when enabled=true" );
            }

            // Check for duplicates
            std::unordered_set<std::string> unique_tokens( tokens.begin(), tokens.end() );
            if ( unique_tokens.size() != tokens.size() ) {
                throw std::invalid_argument( "Special tokens must be unique" );
            }
        }

        /**
         * @brief Check if a string is a special token.
         * @param token String to check
         * @return true if token is a special token, false otherwise
         */
        bool isSpecialToken( const std::string& token ) const {
            if ( !enabled ) {
                return false;
            }

            return token == pad_token || token == unk_token ||
                token == bos_token || token == eos_token ||
                (!mask_token.empty() && token == mask_token) ||
                (!sep_token.empty() && token == sep_token) ||
                (!cls_token.empty() && token == cls_token);
        }

        /**
         * @brief Get the ID offset for regular tokens.
         * @return Number of special tokens that will occupy the first N IDs
         */
        size_t getIdOffset() const {
            return count();
        }

        // Factory methods for common configurations

        /**
         * @brief Create standard BPE special tokens (PAD, UNK, BOS, EOS).
         */
        static BpeSpecialTokens standard() {
            return BpeSpecialTokens{
                .enabled = true,
                .pad_token = "<|pad|>",
                .unk_token = "<|unk|>",
                .bos_token = "<|bos|>",
                .eos_token = "<|eos|>"
            };
        }

        /**
         * @brief Create BPE special tokens for masked language modeling.
         */
        static BpeSpecialTokens forMLM() {
            return BpeSpecialTokens{
                .enabled = true,
                .pad_token = "<|pad|>",
                .unk_token = "<|unk|>",
                .bos_token = "<|bos|>",
                .eos_token = "<|eos|>",
                .mask_token = "<|mask|>"
            };
        }

        /**
         * @brief Create BPE special tokens for sequence classification.
         */
        static BpeSpecialTokens forClassification() {
            return BpeSpecialTokens{
                .enabled = true,
                .pad_token = "<|pad|>",
                .unk_token = "<|unk|>",
                .bos_token = "<|bos|>",
                .eos_token = "<|eos|>",
                .sep_token = "<|sep|>",
                .cls_token = "<|cls|>"
            };
        }

        /**
         * @brief Create configuration with no special tokens.
         */
        static BpeSpecialTokens none() {
            return BpeSpecialTokens{ .enabled = false };
        }

        /**
         * @brief Create GPT-style special tokens.
         */
        static BpeSpecialTokens gptStyle() {
            return BpeSpecialTokens{
                .enabled = true,
                .pad_token = "<|endoftext|>",  // GPT uses same token for multiple purposes
                .unk_token = "<|endoftext|>",
                .bos_token = "<|endoftext|>",
                .eos_token = "<|endoftext|>"
            };
        }
    };
}