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

import Data.Tokenizer;
import Data.TokenizerType;
import Data.TokenizerTrainer;
import Data.TokenizerVocabulary;

import Data.CharTokenizerTrainer;
import Data.CharTokenizerVocabulary;
import Data.CharTokenizer;
// Import BPE when available:
// import Data.BpeTokenizerTrainer;
// import Data.BpeTokenizerVocabulary;
// import Data.BpeTokenizer;

namespace Mila::Data
{
    using Mila::Dnn::Data::Tokenizer;
    using Mila::Dnn::Data::TokenizerType;
    using Mila::Dnn::Data::TokenizerVocabulary;

    /**
     * @brief Factory for creating tokenizer-related components.
     *
     * This class centralizes construction and loading logic so CLI tools and
     * applications can remain tokenizer-agnostic. The factory creates trainers,
     * vocabularies, and tokenizers based on TokenizerType discriminator.
     *
     * All factory methods return nullptr for unsupported tokenizer types.
     * Callers should check for nullptr and handle accordingly.
     *
     * Ownership is transferred to the caller via smart pointers.
     */
    export class TokenizerFactory
    {
    public:
        /**
         * @brief Create a TokenizerTrainer for the requested type.
         *
         * Returns a trainer instance suitable for building vocabularies from
         * text corpora. The trainer's lifetime and usage are managed by the caller.
         *
         * @param type TokenizerType discriminator (Char, Bpe, etc.).
         * @return std::unique_ptr<TokenizerTrainer> Trainer instance, or nullptr
         *         if the type is unsupported or Unknown.
         *
         * Example:
         * @code
         * auto trainer = TokenizerFactory::createTrainer(TokenizerType::Char);
         * if (trainer) {
         *     trainer->addCorpusFromFile("corpus.txt");
         *     auto vocab = trainer->train();
         * }
         * @endcode
         */
        static std::unique_ptr<TokenizerTrainer> createTrainer( TokenizerType type )
        {
            switch ( type )
            {
                case TokenizerType::Char:
                    return std::make_unique<CharTokenizerTrainer>();

                    // case TokenizerType::Bpe:
                    //     return std::make_unique<BpeTokenizerTrainer>(vocab_size);

                default:
                    return nullptr;
            }
        }

        /**
         * @brief Create an empty TokenizerVocabulary for the requested type.
         *
         * Returns an uninitialized vocabulary instance suitable for loading from
         * disk via its load() method. The vocabulary is not usable until loaded.
         *
         * This method is typically used in conjunction with vocabulary loading:
         * create an empty vocabulary, then call load() to populate it.
         *
         * @param type TokenizerType discriminator (Char, Bpe, etc.).
         * @return std::shared_ptr<TokenizerVocabulary> Empty vocabulary instance,
         *         or nullptr if the type is unsupported or Unknown.
         *
         * Example:
         * @code
         * auto vocab = TokenizerFactory::createVocabulary(TokenizerType::Char);
         * if (vocab) {
         *     vocab->load("vocab.bin");
         *     // Now vocab is ready for use
         * }
         * @endcode
         */
        static std::shared_ptr<TokenizerVocabulary> createVocabulary( TokenizerType type )
        {
            switch ( type )
            {
                case TokenizerType::Char:
                    return std::make_shared<CharTokenizerVocabulary>();

                    // case TokenizerType::Bpe:
                    //     return std::make_shared<BpeTokenizerVocabulary>();

                default:
                    return nullptr;
            }
        }

        /**
         * @brief Load a TokenizerVocabulary from disk.
         *
         * Convenience method that creates an empty vocabulary and loads it from
         * the specified path in one step. Equivalent to calling createVocabulary()
         * followed by load().
         *
         * @param type TokenizerType discriminator (Char, Bpe, etc.).
         * @param path Filesystem path to the saved vocabulary file.
         * @return std::shared_ptr<TokenizerVocabulary> Loaded vocabulary instance,
         *         or nullptr if the type is unsupported.
         *
         * @throws std::runtime_error (or derived) if the vocabulary file cannot
         *         be loaded (propagated from the vocabulary's load() implementation).
         *
         * Example:
         * @code
         * auto vocab = TokenizerFactory::loadVocabulary(TokenizerType::Char, "vocab.bin");
         * if (vocab) {
         *     auto id = vocab->tokenToId("hello");
         * }
         * @endcode
         */
        static std::shared_ptr<TokenizerVocabulary> loadVocabulary(
            TokenizerType type,
            const std::filesystem::path& path )
        {
            auto vocab = createVocabulary( type );
            if ( vocab ) {
                vocab->load( path );
            }
            return vocab;
        }

        /**
         * @brief Create a Tokenizer for the requested type with the given vocabulary.
         *
         * Returns a tokenizer instance ready to encode text to token IDs or decode
         * token IDs back to text. The vocabulary must be fully initialized (either
         * loaded from disk or obtained from a trainer).
         *
         * The tokenizer holds a shared reference to the vocabulary, allowing the
         * same vocabulary to be used by multiple tokenizers concurrently.
         *
         * @param type TokenizerType discriminator (Char, Bpe, etc.).
         * @param vocab Shared pointer to an initialized TokenizerVocabulary.
         *              Must not be nullptr and must be loaded/trained.
         * @return std::unique_ptr<Tokenizer> Tokenizer instance, or nullptr if
         *         the type is unsupported or vocab is nullptr.
         *
         * Example:
         * @code
         * auto vocab = TokenizerFactory::loadVocabulary(TokenizerType::Char, "vocab.bin");
         * auto tokenizer = TokenizerFactory::createTokenizer(TokenizerType::Char, vocab);
         * if (tokenizer) {
         *     auto tokens = tokenizer->encode("Hello, world!");
         * }
         * @endcode
         */
        static std::unique_ptr<Tokenizer> createTokenizer(
            TokenizerType type,
            std::shared_ptr<TokenizerVocabulary> vocab )
        {
            if ( !vocab ) {
                return nullptr;
            }

            switch ( type )
            {
                case TokenizerType::Char:
                    return std::make_unique<CharTokenizer>( vocab );

                    // case TokenizerType::Bpe:
                    //     return std::make_unique<BpeTokenizer>(vocab);

                default:
                    return nullptr;
            }
        }
    };
}