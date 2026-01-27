/*!
 * \file
 * \brief BPE tokenizer trainer implementation (skeleton).
 *
 * Concrete TokenizerTrainer for BPE-style tokenizers; algorithm implementation
 * is left to the training implementation (TODO).
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

import Data.BpeTrainerConfig;
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
     * Implements the TokenizerTrainer interface so the BPE trainer can be used
     * by generic preprocessing tooling (TokenizerFactory, CLI tools, etc.).
     *
     * Training algorithm:
     * - Corpus text is accumulated via addCorpus().
     * - train() runs the algorithm over the accumulated corpus.
     * - buildVocabulary() returns ownership of the trained TokenizerVocabulary.
     */
    export class BpeTrainer { // }; : public TokenizerTrainer {
    public:
        /**
         * @brief Construct with configuration.
         *
         * @param config BPE tokenizer configuration.
         * @throws std::invalid_argument if config is invalid.
         */
        explicit BpeTrainer( const BpeTrainerConfig& config = BpeTrainerConfig{} )
            : config_( config )
        {
            if ( !config_.isValid() ) {
                throw std::invalid_argument( "Invalid BPE tokenizer configuration" );
            }
        }

        void addCorpusFromStream( std::istream& stream ) {
            std::string buffer;
            buffer.resize( 64 * 1024 );

            while ( stream.read( buffer.data(), buffer.size() ) || stream.gcount() > 0 ) {
                size_t bytes_read = stream.gcount();
                corpus_.append( buffer.data(), bytes_read );
            }
        }

        void addCorpusFromFile( const std::filesystem::path& path ) {
            std::ifstream file( path, std::ios::binary );
            if ( !file ) {
                throw std::runtime_error( "Cannot open corpus file: " + path.string() );
            }
            addCorpusFromStream( file );
        }

        BpeVocabulary train() {
            auto vocab = BpeVocabulary();

            // Build vocabulary using config
            vocab.buildFromText( corpus_, config_ );

            corpus_.clear();  // Free memory

            return vocab;
        }

    private:
        BpeTrainerConfig config_;
        std::string corpus_;
    };
}