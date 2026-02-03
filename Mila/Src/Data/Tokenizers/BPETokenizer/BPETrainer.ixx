/*!
 * \file
 * \brief BPE tokenizer trainer for corpus accumulation and vocabulary training.
 *
 * Provides corpus management and delegates to BpeVocabulary factory methods.
 * Kept for future extensibility (progress callbacks, streaming, checkpointing).
 */

module;
#include <string>
#include <span>
#include <vector>
#include <memory>
#include <stdexcept>
#include <istream>
#include <fstream>
#include <filesystem>

export module Data.BpeTrainer;

import Data.BpeVocabularyConfig;
import Data.BpeTokenizer;

import Data.TokenizerTrainer;
import Data.TokenizerVocabulary;
import Data.BpeVocabulary;

namespace Mila::Data
{
    using Mila::Dnn::Data::TokenizerVocabulary;

    /**
     * @brief Byte-Pair Encoding (BPE) tokenizer trainer.
     *
     * Manages corpus accumulation and delegates vocabulary training to
     * BpeVocabulary factory methods. Provides a convenient API for
     * incremental corpus loading and batch training.
     *
     * Future extensibility:
     * - Progress callbacks during training
     * - Streaming corpus processing for large files
     * - Training checkpointing and resumption
     * - Validation evaluation during training
     */
    export class BpeTrainer
    {
    public:
        /**
         * @brief Construct with configuration.
         *
         * @param config BPE vocabulary configuration.
         * @throws std::invalid_argument if config is invalid.
         */
        explicit BpeTrainer( const BpeVocabularyConfig& config = BpeVocabularyConfig{} )
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
        void addCorpusFromFile( const std::filesystem::path& path )
        {
            std::ifstream file( path, std::ios::binary );
            
            if ( !file )
            {
                throw std::runtime_error( "Cannot open corpus file: " + path.string() );
            }
            
            addCorpusFromStream( file );
        }

        /**
         * @brief Train vocabulary on accumulated corpus.
         *
         * Delegates to BpeVocabulary::train() factory method. Clears
         * accumulated corpus after training to free memory.
         *
         * @return Trained BpeVocabulary instance.
         * @throws std::runtime_error if corpus is empty.
         * @throws std::invalid_argument if config is invalid.
         */
        BpeVocabulary train()
        {
            if ( corpus_.empty() )
            {
                throw std::runtime_error( "BpeTrainer: Cannot train on empty corpus" );
            }

            BpeVocabulary vocab = BpeVocabulary::train( corpus_, config_ );
            
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
         * @return const BpeVocabularyConfig& Configuration reference.
         */
        const BpeVocabularyConfig& getConfig() const
        {
            return config_;
        }

    private:
        BpeVocabularyConfig config_;
        std::string corpus_;
    };
}