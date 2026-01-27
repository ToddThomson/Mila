/*!
 * \file
 * \brief Character-level tokenizer trainer using the TokenizerTrainer API.
 *
 * Provides a concrete TokenizerTrainer that builds a CharTokenizerVocabulary
 * and exposes it via the generic TokenizerVocabulary interface.
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

import Data.CharTrainerConfig;

import Data.CharTokenizer;
import Data.CharVocabulary;

import Data.Tokenizer;
import Data.TokenizerTrainer;
import Data.TokenizerVocabulary;

namespace Mila::Data
{
    namespace fs = std::filesystem;

    using Mila::Dnn::Data::TokenizerVocabulary;
    using Mila::Dnn::Data::TokenId;

    export class CharTrainer {
    public:
        /**
         * @brief Construct with configuration.
         *
         * @param config Character tokenizer configuration.
         */
        explicit CharTrainer( const CharTrainerConfig& config = CharTrainerConfig{} )
            : config_( config )
        {}

        void addCorpusFromStream( std::istream& stream ) {
            // Read and accumulate corpus
            std::string buffer;
            buffer.resize( 64 * 1024 );

            while ( stream.read( buffer.data(), buffer.size() ) || stream.gcount() > 0 ) {
                size_t bytes_read = stream.gcount();
                corpus_.append( buffer.data(), bytes_read );
            }
        }

        void addCorpusFromFile( const fs::path& path ) {
            std::ifstream file( path, std::ios::binary );

            if ( !file ) {
                throw std::runtime_error( "Cannot open corpus file: " + path.string() );
            }

            addCorpusFromStream( file );
        }

        CharVocabulary train()
        {
            auto vocab = CharVocabulary();

            vocab.buildFromText( corpus_, config_ );

            corpus_.clear();  // Free memory

            return vocab;
        }

    private:
        CharTrainerConfig config_;
        std::string corpus_;
    };
}