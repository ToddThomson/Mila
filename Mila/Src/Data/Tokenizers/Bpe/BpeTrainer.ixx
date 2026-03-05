/**
 * @file BpeTrainer.ixx
 * @brief BPE vocabulary trainer with incremental corpus accumulation.
 *
 * Delegates vocabulary construction to BpeVocabulary::train(). Retained as a
 * separate class for future extensibility: progress callbacks, streaming corpus
 * processing, and training checkpointing.
 */

module;
#include <string>
#include <stdexcept>
#include <istream>
#include <fstream>
#include <filesystem>

export module Data.BpeTrainer;

import Data.BpeVocabularyConfig;
import Data.BpeVocabulary;
import Data.TokenizerTrainer;

namespace Mila::Data
{
    /**
     * @brief Corpus accumulator and trainer for BPE vocabularies.
     *
     * Typical usage:
     * @code
     * BpeTrainer trainer( BpeVocabularyConfig()
     *     .withVocabSize( 32000 )
     *     .withByteLevel( true )
     *     .withPreTokenization( PreTokenizationMode::Gpt2Regex )
     *     .withPreTokenizationPattern( GPT2_PRETOKENIZATION_PATTERN ) );
     *
     * trainer.addCorpusFromFile( "corpus.txt" );
     * BpeVocabulary vocab = trainer.train();
     * vocab.save( "my_vocab.bin" );
     * @endcode
     */
    export class BpeTrainer
    {
    public:

        /**
         * @brief Construct with a vocabulary configuration.
         *
         * validate() is called immediately so misconfigured trainers fail at
         * construction rather than at train() time.
         *
         * @param config BPE vocabulary configuration.
         * @throws std::invalid_argument if config fails validation.
         */
        explicit BpeTrainer( const BpeVocabularyConfig& config = BpeVocabularyConfig{} )
            : config_( config )
        {
            config_.validate();
        }

        // ====================================================================
        // Corpus Accumulation
        // ====================================================================

        /**
         * @brief Append corpus text from a stream.
         *
         * May be called multiple times to accumulate text from different sources
         * before a single train() call.
         *
         * @param stream Input stream containing UTF-8 text.
         */
        void addCorpusFromStream( std::istream& stream )
        {
            std::string buffer( 64 * 1024, '\0' );

            while ( stream.read( buffer.data(), static_cast<std::streamsize>(buffer.size()) )
                || stream.gcount() > 0 )
            {
                corpus_.append( buffer.data(), static_cast<size_t>(stream.gcount()) );
            }
        }

        /**
         * @brief Append corpus text from a file.
         *
         * @param path Path to a UTF-8 text file.
         * @throws std::runtime_error if the file cannot be opened.
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

        // ====================================================================
        // Training
        // ====================================================================

        /**
         * @brief Train a BPE vocabulary on the accumulated corpus.
         *
         * Delegates to BpeVocabulary::train() and clears the accumulated corpus
         * afterwards to release memory. The returned vocabulary can be saved via
         * BpeVocabulary::save() and later reloaded with BpeVocabulary::load().
         *
         * @return Trained BpeVocabulary instance.
         * @throws std::runtime_error if no corpus has been accumulated.
         */
        BpeVocabulary train()
        {
            if ( corpus_.empty() )
            {
                throw std::runtime_error( "BpeTrainer: cannot train on an empty corpus" );
            }

            BpeVocabulary vocab = BpeVocabulary::train( corpus_, config_ );

            corpus_.clear();
            corpus_.shrink_to_fit();

            return vocab;
        }

        // ====================================================================
        // Accessors
        // ====================================================================

        size_t getCorpusSize() const
        {
            return corpus_.size();
        }

        const BpeVocabularyConfig& getConfig() const
        {
            return config_;
        }

        void clearCorpus()
        {
            corpus_.clear();
            corpus_.shrink_to_fit();
        }

    private:

        BpeVocabularyConfig config_;
        std::string         corpus_;
    };
}