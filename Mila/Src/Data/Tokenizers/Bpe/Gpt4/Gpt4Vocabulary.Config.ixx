/**
 * @file Gpt4BpeVocabularyConfig.ixx
 * @brief Configuration for GPT-4 style BPE vocabulary (Llama 3.x target).
 *
 * Mirrors BpeVocabularyConfig from the BpeTokenizer module but is scoped
 * to the GPT-4 / Llama 3.x BPE family. Key differences from BpeVocabularyConfig:
 *
 *  - Default vocab_size is 128256 (Llama 3.x) rather than 32000 (GPT-2)
 *  - Default pre_tokenization_mode is Llama3Regex rather than None
 *  - No min_frequency or max_merges: GPT-4 style vocabularies are always
 *    loaded from pretrained files, never trained from scratch in Mila alpha.2
 *  - Special tokens default to llamaStyle() rather than standard()
 */

module;
#include <cstddef>
#include <string>
#include <stdexcept>
#include <sstream>

export module Data.Tokenizers.Bpe.Gpt4VocabularyConfig;

import Data.Tokenizers.Bpe.PreTokenizationMode;
import Data.SpecialTokens;
import Serialization.Metadata;

namespace Mila::Data
{
    using Mila::Dnn::Serialization::SerializationMetadata;

    /**
     * @brief Configuration for GPT-4 style BPE vocabulary.
     *
     * Designed for loading pretrained Llama 3.x vocabularies from Mila binary
     * format. Training from scratch is not supported for this vocabulary type.
     */
    export class Gpt4BpeVocabularyConfig
    {
    public:
        Gpt4BpeVocabularyConfig() = default;

        Gpt4BpeVocabularyConfig& withVocabSize( size_t size )
        {
            vocab_size_ = size;
            return *this;
        }

        Gpt4BpeVocabularyConfig& withSpecialTokens( const SpecialTokens& tokens )
        {
            special_tokens_ = tokens;
            return *this;
        }

        Gpt4BpeVocabularyConfig& withByteLevel( bool byte_level )
        {
            byte_level_ = byte_level;
            return *this;
        }

        Gpt4BpeVocabularyConfig& withPreTokenization( PreTokenizationMode mode )
        {
            pre_tokenization_mode_ = mode;
            return *this;
        }

        Gpt4BpeVocabularyConfig& withPreTokenizationPattern( const std::string& pattern )
        {
            pre_tokenization_pattern_ = pattern;
            return *this;
        }

        void validate() const
        {
            if ( vocab_size_ == 0 )
            {
                throw std::invalid_argument( "Gpt4BpeVocabularyConfig: vocab_size must be > 0" );
            }

            if ( byte_level_ )
            {
                // 256 base bytes + named special tokens (extended tokens have explicit IDs)
                const size_t min_base_size = 256 + special_tokens_.count();
                if ( vocab_size_ < min_base_size )
                {
                    throw std::invalid_argument(
                        "Gpt4BpeVocabularyConfig: vocab_size (" + std::to_string( vocab_size_ ) +
                        ") must be >= base vocabulary size (" + std::to_string( min_base_size ) +
                        ") [256 bytes + " + std::to_string( special_tokens_.count() ) + " special tokens]" );
                }
            }
        }

        SerializationMetadata toMetadata() const
        {
            SerializationMetadata meta;

            meta.set( "vocab_size", static_cast<int64_t>(vocab_size_) )
                .set( "byte_level", byte_level_ )
                .set( "pre_tokenization_pattern", pre_tokenization_pattern_ )
                .set( "use_pad", special_tokens_.use_pad )
                .set( "use_unk", special_tokens_.use_unk )
                .set( "use_bos", special_tokens_.use_bos )
                .set( "use_eos", special_tokens_.use_eos )
                .set( "use_mask", special_tokens_.use_mask )
                .set( "use_sep", special_tokens_.use_sep )
                .set( "use_cls", special_tokens_.use_cls )
                .set( "bos_token", special_tokens_.bos_token )
                .set( "eos_token", special_tokens_.eos_token );

            return meta;
        }

        void fromMetadata( const SerializationMetadata& meta )
        {
            if ( auto v = meta.tryGetInt( "vocab_size" ) )
                vocab_size_ = static_cast<size_t>(*v);

            if ( auto v = meta.tryGetBool( "byte_level" ) )
                byte_level_ = *v;

            if ( auto v = meta.tryGetString( "pre_tokenization_pattern" ) )
                pre_tokenization_pattern_ = *v;

            if ( auto v = meta.tryGetBool( "use_pad" ) )   special_tokens_.use_pad = *v;
            if ( auto v = meta.tryGetBool( "use_unk" ) )   special_tokens_.use_unk = *v;
            if ( auto v = meta.tryGetBool( "use_bos" ) )   special_tokens_.use_bos = *v;
            if ( auto v = meta.tryGetBool( "use_eos" ) )   special_tokens_.use_eos = *v;
            if ( auto v = meta.tryGetBool( "use_mask" ) )  special_tokens_.use_mask = *v;
            if ( auto v = meta.tryGetBool( "use_sep" ) )   special_tokens_.use_sep = *v;
            if ( auto v = meta.tryGetBool( "use_cls" ) )   special_tokens_.use_cls = *v;

            if ( auto v = meta.tryGetString( "bos_token" ) ) special_tokens_.bos_token = *v;
            if ( auto v = meta.tryGetString( "eos_token" ) ) special_tokens_.eos_token = *v;
        }

        std::string toString() const
        {
            std::ostringstream oss;
            oss << "Gpt4BpeVocabularyConfig: { ";
            oss << "vocab_size=" << vocab_size_ << ", ";
            oss << "byte_level=" << (byte_level_ ? "true" : "false") << ", ";
            oss << "special_tokens=" << special_tokens_.count();
            oss << " }";
            return oss.str();
        }

        // Accessors
        const SpecialTokens& getSpecialTokens() const
        {
            return special_tokens_;
        }
        
        size_t getVocabSize() const
        {
            return vocab_size_;
        }
        
        bool isByteLevel() const
        {
            return byte_level_;
        }
        
        PreTokenizationMode getPreTokenizationMode() const
        {
            return pre_tokenization_mode_;
        }
        
        const std::string& getPreTokenizationPattern() const
        {
            return pre_tokenization_pattern_;
        }

    private:
        size_t vocab_size_ = 128256;                                      // Llama 3.x default
        bool byte_level_ = true;
        SpecialTokens special_tokens_ = SpecialTokens::llamaStyle();
        PreTokenizationMode pre_tokenization_mode_ = PreTokenizationMode::Llama3Regex;
        std::string pre_tokenization_pattern_ = LLAMA3_PRETOKENIZATION_PATTERN;
    };
}
