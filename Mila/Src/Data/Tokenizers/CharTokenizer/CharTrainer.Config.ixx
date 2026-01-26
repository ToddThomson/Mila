export module Data.CharTrainer:Config;

import Data.TrainerConfig;
import Data.SpecialTokens;

namespace Mila::Data
{
    /**
     * @brief Configuration for Character-level tokenizer training.
     *
     * Character tokenizers split text into individual characters (or bytes).
     * This is the simplest tokenization approach with minimal configuration needs.
     *
     * Fluent interface allows chaining:
     * @code
     * auto config = CharTokenizerConfig()
     *     .withSpecialTokens(my_tokens)
     *     .withCaseSensitive(false)
     *     .withNormalizeUnicode(true);
     * @endcode
     */
    export class CharTrainerConfig : public TrainerConfig {
    public:
        CharTrainerConfig() = default;

        /**
         * @brief Configure special tokens.
         *
         * @param tokens SpecialTokens configuration.
         * @return Reference to this config for method chaining.
         */
        CharTrainerConfig& withSpecialTokens( const SpecialTokens& tokens ) {
            special_tokens_ = tokens;
            return *this;
        }

        /**
         * @brief Set whether tokenization is case-sensitive.
         *
         * When false, text is converted to lowercase before tokenization.
         * This reduces vocabulary size but loses case information.
         *
         * @param sensitive True for case-sensitive (default), false for case-insensitive.
         * @return Reference to this config for method chaining.
         */
        CharTrainerConfig& withCaseSensitive( bool sensitive ) {
            case_sensitive_ = sensitive;
            return *this;
        }

        /**
         * @brief Set whether to normalize Unicode characters.
         *
         * When true, applies Unicode normalization (e.g., NFC) to ensure
         * consistent representation of characters with multiple encodings.
         *
         * @param normalize True to normalize Unicode, false otherwise (default).
         * @return Reference to this config for method chaining.
         */
        CharTrainerConfig& withNormalizeUnicode( bool normalize ) {
            normalize_unicode_ = normalize;
            return *this;
        }

        /**
         * @brief Set whether to use byte-level encoding.
         *
         * When true, operates on raw UTF-8 bytes instead of Unicode characters.
         * This guarantees any text can be represented but increases sequence length.
         *
         * @param byte_level True for byte-level, false for character-level (default).
         * @return Reference to this config for method chaining.
         */
        CharTrainerConfig& withByteLevel( bool byte_level ) {
            byte_level_ = byte_level;
            return *this;
        }

        bool isCaseSensitive() const {
            return case_sensitive_;
        }
        
        bool shouldNormalizeUnicode() const {
            return normalize_unicode_;
        }
        
        bool isByteLevel() const {
            return byte_level_;
        }

    private:
        bool case_sensitive_ = true;
        bool normalize_unicode_ = false;
        bool byte_level_ = false;
    };
}