/**
 * @file CharVocabularyConfig.ixx
 * @brief Configuration for Character-level tokenizer training.
 *
 * Provides fluent setters and serialization/validation for character tokenizer options.
 */

module;
#include <stdexcept>
#include <string>
#include <sstream>

export module Data.CharVocabularyConfig;

import Data.SpecialTokens;
import Serialization.Metadata;

namespace Mila::Data
{
    using Mila::Dnn::Serialization::SerializationMetadata;

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
    export class CharVocabularyConfig
    {
    public:
        CharVocabularyConfig() = default;

        /**
         * @brief Configure special tokens.
         *
         * @param tokens SpecialTokens configuration.
         * @return Reference to this config for method chaining.
         */
        CharVocabularyConfig& withSpecialTokens( const SpecialTokens& tokens )
        {
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
        CharVocabularyConfig& withCaseSensitive( bool sensitive )
        {
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
        CharVocabularyConfig& withNormalizeUnicode( bool normalize )
        {
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
        CharVocabularyConfig& withByteLevel( bool byte_level )
        {
            byte_level_ = byte_level;

            return *this;
        }

        /**
         * @brief Validate configuration parameters.
         *
         * @throws std::invalid_argument if configuration is invalid.
         */
        void validate() const
        {
            //special_tokens_.validate();
        }

        /**
         * @brief Convert configuration to metadata for serialization.
         *
         * @return SerializationMetadata containing all configuration parameters.
         */
        SerializationMetadata toMetadata() const
        {
            SerializationMetadata meta;

            meta.set( "case_sensitive", case_sensitive_ )
                .set( "normalize_unicode", normalize_unicode_ )
                .set( "byte_level", byte_level_ );

            meta.set( "use_pad", special_tokens_.use_pad )
                .set( "use_unk", special_tokens_.use_unk )
                .set( "use_bos", special_tokens_.use_bos )
                .set( "use_eos", special_tokens_.use_eos )
                .set( "use_mask", special_tokens_.use_mask )
                .set( "use_sep", special_tokens_.use_sep )
                .set( "use_cls", special_tokens_.use_cls );

            if ( special_tokens_.use_pad )
            {
                meta.set( "pad_token", special_tokens_.pad_token );
            }
            if ( special_tokens_.use_unk )
            {
                meta.set( "unk_token", special_tokens_.unk_token );
            }
            if ( special_tokens_.use_bos )
            {
                meta.set( "bos_token", special_tokens_.bos_token );
            }
            if ( special_tokens_.use_eos )
            {
                meta.set( "eos_token", special_tokens_.eos_token );
            }
            if ( special_tokens_.use_mask )
            {
                meta.set( "mask_token", special_tokens_.mask_token );
            }
            if ( special_tokens_.use_sep )
            {
                meta.set( "sep_token", special_tokens_.sep_token );
            }
            if ( special_tokens_.use_cls )
            {
                meta.set( "cls_token", special_tokens_.cls_token );
            }

            return meta;
        }

        /**
         * @brief Populate configuration from metadata.
         *
         * Missing keys are ignored leaving defaults intact.
         *
         * @param meta Metadata to read configuration from.
         */
        void fromMetadata( const SerializationMetadata& meta )
        {
            if ( auto case_sens = meta.tryGetBool( "case_sensitive" ) )
            {
                case_sensitive_ = *case_sens;
            }

            if ( auto normalize = meta.tryGetBool( "normalize_unicode" ) )
            {
                normalize_unicode_ = *normalize;
            }

            if ( auto byte_level = meta.tryGetBool( "byte_level" ) )
            {
                byte_level_ = *byte_level;
            }

            if ( auto use_pad = meta.tryGetBool( "use_pad" ) )
            {
                special_tokens_.use_pad = *use_pad;
            }
            if ( auto use_unk = meta.tryGetBool( "use_unk" ) )
            {
                special_tokens_.use_unk = *use_unk;
            }
            if ( auto use_bos = meta.tryGetBool( "use_bos" ) )
            {
                special_tokens_.use_bos = *use_bos;
            }
            if ( auto use_eos = meta.tryGetBool( "use_eos" ) )
            {
                special_tokens_.use_eos = *use_eos;
            }
            if ( auto use_mask = meta.tryGetBool( "use_mask" ) )
            {
                special_tokens_.use_mask = *use_mask;
            }
            if ( auto use_sep = meta.tryGetBool( "use_sep" ) )
            {
                special_tokens_.use_sep = *use_sep;
            }
            if ( auto use_cls = meta.tryGetBool( "use_cls" ) )
            {
                special_tokens_.use_cls = *use_cls;
            }

            if ( auto pad = meta.tryGetString( "pad_token" ) )
            {
                special_tokens_.pad_token = *pad;
            }
            if ( auto unk = meta.tryGetString( "unk_token" ) )
            {
                special_tokens_.unk_token = *unk;
            }
            if ( auto bos = meta.tryGetString( "bos_token" ) )
            {
                special_tokens_.bos_token = *bos;
            }
            if ( auto eos = meta.tryGetString( "eos_token" ) )
            {
                special_tokens_.eos_token = *eos;
            }
            if ( auto mask = meta.tryGetString( "mask_token" ) )
            {
                special_tokens_.mask_token = *mask;
            }
            if ( auto sep = meta.tryGetString( "sep_token" ) )
            {
                special_tokens_.sep_token = *sep;
            }
            if ( auto cls = meta.tryGetString( "cls_token" ) )
            {
                special_tokens_.cls_token = *cls;
            }
        }

        /**
         * @brief Produce human-readable summary of configuration.
         *
         * Suitable for logging and debugging.
         *
         * @return std::string Configuration summary.
         */
        std::string toString() const
        {
            std::ostringstream oss;
            oss << "CharVocabularyConfig: { ";
            oss << "case_sensitive=" << ( case_sensitive_ ? "true" : "false" ) << ", ";
            oss << "normalize_unicode=" << ( normalize_unicode_ ? "true" : "false" ) << ", ";
            oss << "byte_level=" << ( byte_level_ ? "true" : "false" ) << ", ";
            oss << "special_tokens=[";

            bool first = true;
            if ( special_tokens_.use_pad )
            {
                oss << "PAD";
                first = false;
            }
            if ( special_tokens_.use_unk )
            {
                if ( !first ) oss << ",";
                oss << "UNK";
                first = false;
            }
            if ( special_tokens_.use_bos )
            {
                if ( !first ) oss << ",";
                oss << "BOS";
                first = false;
            }
            if ( special_tokens_.use_eos )
            {
                if ( !first ) oss << ",";
                oss << "EOS";
                first = false;
            }
            if ( special_tokens_.use_mask )
            {
                if ( !first ) oss << ",";
                oss << "MASK";
                first = false;
            }
            if ( special_tokens_.use_sep )
            {
                if ( !first ) oss << ",";
                oss << "SEP";
                first = false;
            }
            if ( special_tokens_.use_cls )
            {
                if ( !first ) oss << ",";
                oss << "CLS";
                first = false;
            }

            if ( first )
            {
                oss << "none";
            }

            oss << "] }";

            return oss.str();
        }

        const SpecialTokens& getSpecialTokens() const
        {
            return special_tokens_;
        }

        bool isCaseSensitive() const
        {
            return case_sensitive_;
        }

        bool shouldNormalizeUnicode() const
        {
            return normalize_unicode_;
        }

        bool isByteLevel() const
        {
            return byte_level_;
        }

    private:
        bool case_sensitive_ = true;
        bool normalize_unicode_ = false;
        bool byte_level_ = false;
        SpecialTokens special_tokens_{};
    };
}