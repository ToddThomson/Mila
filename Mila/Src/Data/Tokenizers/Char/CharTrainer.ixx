/*!
 * \file
 * \brief Character-level tokenizer trainer for corpus accumulation and vocabulary building.
 *
 * Provides corpus management and delegates to CharVocabulary factory methods.
 * Maintained for API consistency with BpeTrainer, though character tokenization
 * is simple enough that direct use of CharVocabulary::trainFromFile() is often preferred.
 */

module;
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <memory>
#include <optional>
#include <algorithm>

export module Data.CharTrainer;

import Data.CharVocabularyConfig;
import Data.CharTokenizer;
import Data.CharVocabulary;
import Data.Tokenizer;
import Data.TokenizerTrainer;
import Data.TokenizerVocabulary;

namespace Mila::Data
{
    namespace fs = std::filesystem;

    using Mila::Data::TokenizerVocabulary;
    using Mila::Data::TokenId;

    /**
     * @brief Character-level tokenizer trainer.
     *
     * Manages corpus accumulation and delegates vocabulary building to
     * CharVocabulary factory methods. Provides a convenient API for
     * incremental corpus loading and batch training.
     *
     * Note: Character tokenization is simple enough that direct use of
     * CharVocabulary::train() or CharVocabulary::trainFromFile() is often
     * preferred. This trainer is maintained for API consistency with
     * BpeTrainer and potential future extensions.
     */
    export class CharTrainer
    {
    public:
        /**
         * @brief Construct with configuration.
         *
         * @param config Character vocabulary configuration.
         * @throws std::invalid_argument if config is invalid.
         */
        explicit CharTrainer( const CharVocabularyConfig& config = CharVocabularyConfig{} )
            : config_( config )
        {
            config_.validate();
        }

        /**
         * @brief Add corpus text from input stream.
         *
         * Accumulates text for training. Can be called multiple times to
         * add corpus from different sources.
         *
         * @param stream Input stream containing corpus text.
         */
        void addCorpusFromStream( std::istream& stream )
        {
            std::string buffer;
            buffer.resize( 64 * 1024 );

            while ( stream.read( buffer.data(), buffer.size() ) || stream.gcount() > 0 )
            {
                size_t bytes_read = stream.gcount();
                corpus_.append( buffer.data(), bytes_read );
            }
        }

        /**
         * @brief Add corpus text from file.
         *
         * Convenience method for loading corpus from filesystem.
         *
         * @param path Path to corpus text file.
         * @throws std::runtime_error if file cannot be opened.
         */
        void addCorpusFromFile( const fs::path& path )
        {
            std::ifstream file( path, std::ios::binary );

            if ( !file )
            {
                throw std::runtime_error( "Cannot open corpus file: " + path.string() );
            }

            addCorpusFromStream( file );
        }

        /**
         * @brief Build vocabulary on accumulated corpus.
         *
         * Delegates to CharVocabulary::train() factory method. Clears
         * accumulated corpus after training to free memory.
         *
         * @return Built CharVocabulary instance.
         * @throws std::runtime_error if corpus is empty.
         * @throws std::invalid_argument if config is invalid.
         */
        CharVocabulary train()
        {
            if ( corpus_.empty() )
            {
                throw std::runtime_error( "CharTrainer: Cannot train on empty corpus" );
            }

            CharVocabulary vocab = CharVocabulary::train( corpus_, config_ );

            corpus_.clear();

            return vocab;
        }

        /**
         * @brief Get accumulated corpus size in bytes.
         *
         * @return Size of accumulated corpus.
         */
        size_t getCorpusSize() const
        {
            return corpus_.size();
        }

        /**
         * @brief Clear accumulated corpus.
         *
         * Frees memory used by corpus accumulation.
         */
        void clearCorpus()
        {
            corpus_.clear();
            corpus_.shrink_to_fit();
        }

        /**
         * @brief Get the trainer configuration.
         *
         * @return const CharVocabularyConfig& Configuration reference.
         */
        const CharVocabularyConfig& getConfig() const
        {
            return config_;
        }

    private:
        CharVocabularyConfig config_;
        std::string corpus_;
    };
}