/*!
 * \file
 * \brief Factory helpers to construct tokenizer trainers and load vocabularies.
 *
 * Provides simple factory functions used by preprocessing tools to obtain
 * TokenizerTrainer and TokenizerVocabulary implementations for a given
 * TokenizerType. Currently supports `char` and BPE tokenizers
 */

module;
#include <memory>
#include <filesystem>
#include <string>
#include <span>
#include <stdexcept>

export module Data.TrainerFactory;

import Data.Tokenizer;
import Data.TokenizerType;
import Data.TokenizerTrainer;
import Data.TokenizerVocabulary;

import Data.CharTrainer;
import Data.CharVocabulary;
import Data.CharVocabularyConfig;
import Data.CharTokenizer;

import Data.BpeTrainer;
import Data.BpeVocabulary;
import Data.BpeVocabularyConfig;
import Data.BpeTokenizer;

namespace Mila::Data
{
    using Mila::Dnn::Data::Tokenizer;
    using Mila::Dnn::Data::TokenizerType;
    using Mila::Dnn::Data::TokenizerVocabulary;

    /**
     * @brief Factory for creating tokenizer trainers and loading vocabularies.
     *
     * This class centralizes construction and loading logic so CLI tools and
     * applications can remain tokenizer-agnostic. The factory creates trainers
     * and loads vocabularies based on TokenizerType discriminator.
     */
    export class TrainerFactory
    {
    public:
        /**
         * @brief Train a vocabulary from corpus files.
         *
         * Creates a trainer with the provided config, adds corpus files,
         * trains the vocabulary, and saves it to disk.
         *
         * @param type TokenizerType discriminator (Char, Bpe, etc.).
         * @param corpusFiles List of corpus file paths to train on.
         * @param outputPath Where to save the trained vocabulary.
         *
         * Example:
         * @code
         * TrainerFactory::trainVocabulary(
         *     TokenizerType::Bpe,
         *     {"corpus1.txt", "corpus2.txt"},
         *     "vocab.bin"
         * );
         * @endcode
         */
        template<typename Config>
        static void trainVocabulary(
            TokenizerType type,
            const Config& config,
            std::span<const std::filesystem::path> corpusFiles,
            const std::filesystem::path& outputPath )
        {
            switch ( type )
            {
                case TokenizerType::Char:
                {
                    if constexpr ( std::is_same_v<Config, CharVocabularyConfig> ) {
                        CharTrainer trainer( config );
                        for ( const auto& file : corpusFiles ) {
                            trainer.addCorpusFromFile( file );
                        }

                        CharVocabulary vocab = trainer.train();
                        vocab.save( outputPath );
                    }
                    else {
                        throw std::invalid_argument( "Config type mismatch for Char tokenizer" );
                    }
                    break;
                }

                case TokenizerType::Bpe:
                {
                    if constexpr ( std::is_same_v<Config, BpeVocabularyConfig> ) {
                        BpeTrainer trainer( config );
                        for ( const auto& file : corpusFiles ) {
                            trainer.addCorpusFromFile( file );
                        }

                        BpeVocabulary vocab = trainer.train();
                        vocab.save( outputPath );
                    }
                    else {
                        throw std::invalid_argument( "Config type mismatch for Bpe tokenizer" );
                    }
                    break;
                }

                default:
                    throw std::invalid_argument( "Unsupported tokenizer type" );
            }
        }

        /**
         * @brief Load a tokenizer from a saved vocabulary file.
         *
         * This loads the vocabulary and creates a tokenizer in one operation.
         * The returned tokenizer owns its vocabulary.
         *
         * @param type TokenizerType discriminator (Char, Bpe, etc.).
         * @param vocabPath Path to the saved vocabulary file.
         * @return std::unique_ptr<Tokenizer> Ready-to-use tokenizer.
         *
         * Example:
         * @code
         * auto tokenizer = TrainerFactory::loadTokenizer(
         *     TokenizerType::Bpe,
         *     "vocab.bin"
         * );
         * auto tokens = tokenizer->encode("Hello, world!");
         * @endcode
         */
        static std::unique_ptr<Tokenizer> loadTokenizer(
            TokenizerType type,
            const std::filesystem::path& vocabPath )
        {
            switch ( type )
            {
                case TokenizerType::Char:
                {
                    CharVocabulary vocab = CharVocabulary::load( vocabPath );
                    return std::make_unique<CharTokenizer>( std::move( vocab ) );
                }

                case TokenizerType::Bpe:
                {
                    BpeVocabulary vocab = BpeVocabulary::load( vocabPath );
                    return std::make_unique<BpeTokenizer>( std::move( vocab ) );
                }

                default:
                    return nullptr;
            }
        }
    };
}