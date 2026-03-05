/**
 * @file BpeVocabularyConfig.ixx
 * @brief Unified configuration for BPE vocabulary construction and runtime properties.
 *
 * Covers training hyperparameters (used only by BpeTrainer) and vocabulary properties
 * shared across the GPT-2, Llama 3.x, and Mistral BPE families. Training fields
 * (min_frequency, max_merges, enable_merge_caching) are ignored when loading
 * pretrained vocabularies; validate() enforces them only when called by BpeTrainer.
 */

module;
#include <cstddef>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <sstream>

export module Data.BpeVocabularyConfig;

import Data.BpePreTokenizationMode;
import Data.SpecialTokens;
import Serialization.Metadata;

namespace Mila::Data
{
    using Mila::Dnn::Serialization::SerializationMetadata;

    /**
     * @brief Configuration for the BPE vocabulary.
     *
     * Describes both training hyperparameters and the runtime properties
     * (byte-level encoding, pre-tokenization pattern, special token set) that
     * apply to all BPE families. Serialized with vocabulary files to provide
     * full provenance and to enable validation of vocabulary compatibility.
     *
     * Typical usage for pretrained models (no training validation needed):
     * @code
     * auto config = BpeVocabularyConfig()
     *     .withVocabSize( 128256 )
     *     .withByteLevel( true )
     *     .withPreTokenization( PreTokenizationMode::Llama3Regex )
     *     .withPreTokenizationPattern( LLAMA3_PRETOKENIZATION_PATTERN )
     *     .withSpecialTokens( SpecialTokens::llamaStyle() );
     * @endcode
     */
    export class BpeVocabularyConfig
    {
    public:
        BpeVocabularyConfig() = default;

        // ====================================================================
        // Fluent Setters
        // ====================================================================

        BpeVocabularyConfig& withVocabSize( size_t size )
        {
            vocab_size_ = size;
            return *this;
        }

        BpeVocabularyConfig& withSpecialTokens( const SpecialTokens& tokens )
        {
            special_tokens_ = tokens;
            return *this;
        }

        BpeVocabularyConfig& withMinFrequency( size_t frequency )
        {
            min_frequency_ = frequency;
            return *this;
        }

        BpeVocabularyConfig& withByteLevel( bool byte_level )
        {
            byte_level_ = byte_level;
            return *this;
        }

        BpeVocabularyConfig& withPreTokenization( PreTokenizationMode mode )
        {
            pre_tokenization_mode_ = mode;
            return *this;
        }

        BpeVocabularyConfig& withPreTokenizationPattern( const std::string& pattern )
        {
            pre_tokenization_pattern_ = pattern;
            return *this;
        }

        BpeVocabularyConfig& withMaxMerges( size_t max_merges )
        {
            max_merges_ = max_merges;
            return *this;
        }

        BpeVocabularyConfig& withMergeCaching( bool enable )
        {
            enable_merge_caching_ = enable;
            return *this;
        }

        // ====================================================================
        // Validation (training only)
        // ====================================================================

        /**
         * @brief Validate configuration for training.
         *
         * Called by BpeTrainer before training begins. Must not be called for
         * pretrained vocabularies loaded via factory methods.
         *
         * @throws std::invalid_argument on invalid training configuration.
         */
        void validate() const
        {
            if ( vocab_size_ == 0 )
            {
                throw std::invalid_argument( "BpeVocabularyConfig: vocab_size must be > 0" );
            }

            if ( min_frequency_ == 0 )
            {
                throw std::invalid_argument( "BpeVocabularyConfig: min_frequency must be > 0" );
            }

            if ( byte_level_ )
            {
                const size_t min_base_size = 256 + special_tokens_.count();

                if ( vocab_size_ < min_base_size )
                {
                    throw std::invalid_argument(
                        "BpeVocabularyConfig: vocab_size (" + std::to_string( vocab_size_ ) +
                        ") must be >= base vocabulary size (" + std::to_string( min_base_size ) +
                        ") [256 bytes + " + std::to_string( special_tokens_.count() ) + " special tokens]" );
                }
            }

            if ( max_merges_ > 0 && max_merges_ >= vocab_size_ )
            {
                throw std::invalid_argument(
                    "BpeVocabularyConfig: max_merges must be < vocab_size (or 0 for unlimited)" );
            }
        }

        // ====================================================================
        // Serialization
        // ====================================================================

        /**
         * @brief Serialize configuration to metadata.
         *
         * Persists all fields including token strings so that round-tripped
         * vocabularies reproduce the correct special token set on load.
         */
        SerializationMetadata toMetadata() const
        {
            SerializationMetadata meta;

            meta.set( "vocab_size", static_cast<int64_t>(vocab_size_) )
                .set( "min_frequency", static_cast<int64_t>(min_frequency_) )
                .set( "byte_level", byte_level_ )
                .set( "pre_tokenization_mode", static_cast<int64_t>(pre_tokenization_mode_) )
                .set( "pre_tokenization_pattern", pre_tokenization_pattern_ )
                .set( "max_merges", static_cast<int64_t>(max_merges_) )
                .set( "enable_merge_caching", enable_merge_caching_ )
                .set( "use_pad", special_tokens_.use_pad )
                .set( "use_unk", special_tokens_.use_unk )
                .set( "use_bos", special_tokens_.use_bos )
                .set( "use_eos", special_tokens_.use_eos )
                .set( "use_mask", special_tokens_.use_mask )
                .set( "use_sep", special_tokens_.use_sep )
                .set( "use_cls", special_tokens_.use_cls )
                .set( "pad_token", special_tokens_.pad_token )
                .set( "unk_token", special_tokens_.unk_token )
                .set( "bos_token", special_tokens_.bos_token )
                .set( "eos_token", special_tokens_.eos_token )
                .set( "mask_token", special_tokens_.mask_token )
                .set( "sep_token", special_tokens_.sep_token )
                .set( "cls_token", special_tokens_.cls_token );

            return meta;
        }

        /**
         * @brief Restore configuration from metadata.
         *
         * All fields use tryGet* so that files produced by older builds
         * without a given field fall back silently to the in-class defaults.
         */
        void fromMetadata( const SerializationMetadata& meta )
        {
            if ( auto v = meta.tryGetInt( "vocab_size" ) )
                vocab_size_ = static_cast<size_t>(*v);

            if ( auto v = meta.tryGetInt( "min_frequency" ) )
                min_frequency_ = static_cast<size_t>(*v);

            if ( auto v = meta.tryGetBool( "byte_level" ) )
                byte_level_ = *v;

            if ( auto v = meta.tryGetInt( "pre_tokenization_mode" ) )
                pre_tokenization_mode_ = static_cast<PreTokenizationMode>(*v);

            if ( auto v = meta.tryGetString( "pre_tokenization_pattern" ) )
                pre_tokenization_pattern_ = *v;

            if ( auto v = meta.tryGetInt( "max_merges" ) )
                max_merges_ = static_cast<size_t>(*v);

            if ( auto v = meta.tryGetBool( "enable_merge_caching" ) )
                enable_merge_caching_ = *v;

            // use_* flags
            if ( auto v = meta.tryGetBool( "use_pad" ) )  special_tokens_.use_pad = *v;
            if ( auto v = meta.tryGetBool( "use_unk" ) )  special_tokens_.use_unk = *v;
            if ( auto v = meta.tryGetBool( "use_bos" ) )  special_tokens_.use_bos = *v;
            if ( auto v = meta.tryGetBool( "use_eos" ) )  special_tokens_.use_eos = *v;
            if ( auto v = meta.tryGetBool( "use_mask" ) ) special_tokens_.use_mask = *v;
            if ( auto v = meta.tryGetBool( "use_sep" ) )  special_tokens_.use_sep = *v;
            if ( auto v = meta.tryGetBool( "use_cls" ) )  special_tokens_.use_cls = *v;

            // token strings — absent in old files, defaults are preserved via tryGetString
            if ( auto v = meta.tryGetString( "pad_token" ) )  special_tokens_.pad_token = *v;
            if ( auto v = meta.tryGetString( "unk_token" ) )  special_tokens_.unk_token = *v;
            if ( auto v = meta.tryGetString( "bos_token" ) )  special_tokens_.bos_token = *v;
            if ( auto v = meta.tryGetString( "eos_token" ) )  special_tokens_.eos_token = *v;
            if ( auto v = meta.tryGetString( "mask_token" ) ) special_tokens_.mask_token = *v;
            if ( auto v = meta.tryGetString( "sep_token" ) )  special_tokens_.sep_token = *v;
            if ( auto v = meta.tryGetString( "cls_token" ) )  special_tokens_.cls_token = *v;
        }

        // ====================================================================
        // Diagnostics
        // ====================================================================

        std::string toString() const
        {
            std::ostringstream oss;
            oss << "BpeVocabularyConfig { "
                << "vocab_size=" << vocab_size_ << ", "
                << "min_frequency=" << min_frequency_ << ", "
                << "byte_level=" << (byte_level_ ? "true" : "false") << ", "
                << "max_merges=" << max_merges_ << ", "
                << "special_tokens=" << special_tokens_.count()
                << " }";

            return oss.str();
        }

        // ====================================================================
        // Accessors
        // ====================================================================

        const SpecialTokens& getSpecialTokens()          const
        {
            return special_tokens_;
        }
        size_t getVocabSize()               const
        {
            return vocab_size_;
        }
        size_t getMinFrequency()            const
        {
            return min_frequency_;
        }
        bool isByteLevel()                const
        {
            return byte_level_;
        }

        PreTokenizationMode   getPreTokenizationMode()     const
        {
            return pre_tokenization_mode_;
        }
        const std::string& getPreTokenizationPattern()  const
        {
            return pre_tokenization_pattern_;
        }
        size_t                getMaxMerges()               const
        {
            return max_merges_;
        }
        bool                  isMergeCachingEnabled()      const
        {
            return enable_merge_caching_;
        }

    private:

        // ====================================================================
        // Member Variables
        // ====================================================================

        // Shared runtime properties (all BPE families)
        size_t              vocab_size_ = 32000;
        bool                byte_level_ = true;
        SpecialTokens       special_tokens_ = SpecialTokens::standard();
        PreTokenizationMode pre_tokenization_mode_ = PreTokenizationMode::None;
        std::string         pre_tokenization_pattern_ = "";

        // Training hyperparameters (BpeTrainer only; ignored for pretrained loads)
        size_t              min_frequency_ = 2;
        size_t              max_merges_ = 0;
        bool                enable_merge_caching_ = true;
    };
}