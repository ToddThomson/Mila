/**
 * @file BpeVocabularyConfig.ixx
 * @brief Configuration for BPE vocabulary creation and properties.
 *
 * Describes both training hyperparameters and vocabulary characteristics.
 * Serialized with vocabulary files for full provenance tracking.
 */

module;
#include <cstddef>
#include <string>
#include <stdexcept>
#include <sstream>

export module Data.BpeVocabularyConfig;

import Data.PreTokenizationMode;
import Data.SpecialTokens;
import Serialization.Metadata;

namespace Mila::Data
{
    using Mila::Dnn::Serialization::SerializationMetadata;

    /**
     * @brief Configuration for Byte Pair Encoding (BPE) vocabulary.
     *
     * Defines both training hyperparameters and vocabulary properties.
     * This configuration is serialized with the vocabulary file to provide
     * full provenance and enable validation of vocabulary compatibility.
     */
    export class BpeVocabularyConfig
    {
    public:
        BpeVocabularyConfig() = default;

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
                size_t min_base_size = 256 + special_tokens_.count();

                if ( vocab_size_ < min_base_size )
                {
                    throw std::invalid_argument(
                        "BpeVocabularyConfig: vocab_size (" + std::to_string( vocab_size_ ) +
                        ") must be > base vocabulary size (" + std::to_string( min_base_size ) +
                        ") [256 bytes + " + std::to_string( special_tokens_.count() ) + " special tokens]" );
                }
            }

            if ( max_merges_ > 0 && max_merges_ >= vocab_size_ )
            {
                throw std::invalid_argument(
                    "BpeVocabularyConfig: max_merges must be < vocab_size (or 0 for unlimited)" );
            }
        }

        SerializationMetadata toMetadata() const
        {
            SerializationMetadata meta;

            meta.set( "vocab_size", static_cast<int64_t>( vocab_size_ ) )
                .set( "min_frequency", static_cast<int64_t>( min_frequency_ ) )
                .set( "byte_level", byte_level_ )
                .set( "pre_tokenization_pattern", pre_tokenization_pattern_ )
                .set( "max_merges", static_cast<int64_t>( max_merges_ ) )
                .set( "enable_merge_caching", enable_merge_caching_ )
                .set( "use_pad", special_tokens_.use_pad )
                .set( "use_unk", special_tokens_.use_unk )
                .set( "use_bos", special_tokens_.use_bos )
                .set( "use_eos", special_tokens_.use_eos )
                .set( "use_mask", special_tokens_.use_mask )
                .set( "use_sep", special_tokens_.use_sep )
                .set( "use_cls", special_tokens_.use_cls );

            return meta;
        }

        void fromMetadata( const SerializationMetadata& meta )
        {
            if ( auto vocab_size = meta.tryGetInt( "vocab_size" ) )
            {
                vocab_size_ = static_cast<size_t>( *vocab_size );
            }

            if ( auto min_freq = meta.tryGetInt( "min_frequency" ) )
            {
                min_frequency_ = static_cast<size_t>( *min_freq );
            }

            if ( auto byte_level = meta.tryGetBool( "byte_level" ) )
            {
                byte_level_ = *byte_level;
            }

            if ( auto pattern = meta.tryGetString( "pre_tokenization_pattern" ) )
            {
                pre_tokenization_pattern_ = *pattern;
            }

            if ( auto max_merges = meta.tryGetInt( "max_merges" ) )
            {
                max_merges_ = static_cast<size_t>( *max_merges );
            }

            if ( auto caching = meta.tryGetBool( "enable_merge_caching" ) )
            {
                enable_merge_caching_ = *caching;
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
        }

        std::string toString() const
        {
            std::ostringstream oss;
            oss << "BpeVocabularyConfig: { ";
            oss << "vocab_size=" << vocab_size_ << ", ";
            oss << "min_frequency=" << min_frequency_ << ", ";
            oss << "byte_level=" << (byte_level_ ? "true" : "false") << ", ";
            oss << "max_merges=" << max_merges_ << ", ";
            oss << "special_tokens=" << special_tokens_.count();
            oss << " }";

            return oss.str();
        }

        const SpecialTokens& getSpecialTokens() const { return special_tokens_; }
        size_t getVocabSize() const { return vocab_size_; }
        size_t getMinFrequency() const { return min_frequency_; }
        bool isByteLevel() const { return byte_level_; }
        PreTokenizationMode getPreTokenizationMode() const { return pre_tokenization_mode_; }
        const std::string& getPreTokenizationPattern() const { return pre_tokenization_pattern_; }
        size_t getMaxMerges() const { return max_merges_; }
        bool isMergeCachingEnabled() const { return enable_merge_caching_; }

    private:
        size_t vocab_size_ = 32000;
        size_t min_frequency_ = 2;
        bool byte_level_ = true;
        SpecialTokens special_tokens_ = SpecialTokens::standard();
        PreTokenizationMode pre_tokenization_mode_ = PreTokenizationMode::None;
        std::string pre_tokenization_pattern_ = "";
        size_t max_merges_ = 0;
        bool enable_merge_caching_ = true;
    };
}