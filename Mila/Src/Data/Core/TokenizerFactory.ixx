/*!
 * \file
 * \brief Factory helpers to construct tokenizer trainers and load vocabularies.
 *
 * Provides simple factory functions used by preprocessing tools to obtain
 * TokenizerTrainer and TokenizerVocabulary implementations for a given
 * TokenizerType. Currently supports `char` tokenizers; other types are
 * stubbed for future implementation.
 */

module;
#include <memory>
#include <filesystem>
#include <string>

export module Data.TokenizerFactory;

import Data.TokenizerType;
import Data.TokenizerTrainer;
import Data.TokenizerVocabulary;

import Data.CharTokenizerTrainer;

namespace Mila::Dnn::Data
{
    /**
     * @brief Small factory for tokenizer-related components.
     *
     * This class centralizes construction/loading logic so CLI/tools can be
     * tokenizer-agnostic. Only lightweight functionality is exposed; callers
     * still own the resulting objects.
     */
    export class TokenizerFactory
    {
    public:
        /**
         * @brief Create a TokenizerTrainer for the requested type.
         *
         * Returns nullptr for unsupported types.
         *
         * @param type TokenizerType discriminator.
         * @return std::unique_ptr<Mila::Data::TokenizerTrainer> Trainer instance or nullptr.
         */
        static std::unique_ptr<Mila::Data::TokenizerTrainer> createTrainer( TokenizerType type )
        {
            switch ( type )
            {
                case TokenizerType::Char:
                    return std::make_unique<Mila::Data::CharTokenizerTrainer>();
                default:
                    return nullptr;
            }
        }

        /**
         * @brief Load a TokenizerVocabulary for the requested type from disk.
         *
         * Returns nullptr for unsupported types or on load failure (exceptions
         * from implementations propagate to the caller).
         *
         * @param type TokenizerType discriminator.
         * @param path Filesystem path to the saved vocabulary.
         * @return std::unique_ptr<TokenizerVocabulary> Vocabulary instance or nullptr.
         */
        static std::unique_ptr<TokenizerVocabulary> loadVocabulary( TokenizerType type, const std::filesystem::path& path )
        {
            switch ( type )
            {
                case TokenizerType::Char:
                {
                    auto v = std::make_unique<Mila::Data::CharTokenizerVocabulary>();
                    v->load( path );
                    return v;
                }
                default:
                    return nullptr;
            }
        }
    };
}