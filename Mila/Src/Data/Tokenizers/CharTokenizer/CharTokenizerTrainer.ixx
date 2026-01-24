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

export module Data.CharTokenizerTrainer;

import Data.CharTokenizerVocabulary;

import Data.Tokenizer;
import Data.TokenizerTrainer;
import Data.TokenizerVocabulary;

namespace Mila::Data
{
    namespace fs = std::filesystem;

    using Mila::Dnn::Data::TokenizerVocabulary;
    using Mila::Dnn::Data::TokenId;

    /**
     * @brief Character-level tokenizer trainer implementing TokenizerTrainer.
     *
     * Accumulates corpus text, constructs a CharTokenizerVocabulary during
     * train(), and returns ownership of the vocabulary via buildVocabulary().
     */
    export class CharTokenizerTrainer : public TokenizerTrainer
    {
    public:
        /**
         * @brief Construct a trainer.
         *
         * @param add_special_tokens When true, PAD/UNK tokens are added to the vocabulary.
         */
        explicit CharTokenizerTrainer( bool add_special_tokens = true )
            : add_special_tokens_( add_special_tokens ),
              trained_( false )
        {
        }

        /**
         * @brief Execute training to build the character vocabulary.
         *
         * After this call buildVocabulary() will produce a TokenizerVocabulary
         * reflecting the trained character vocabulary.
         */
        std::shared_ptr<TokenizerVocabulary> train() override
        {
            char_vocab_.buildFromText( corpus_, add_special_tokens_ );
            trained_ = true;

            return std::make_shared<CharTokenizerVocabulary>( char_vocab_ );
        }

        void addCorpusFromStream( std::istream& stream ) override
        {
            std::string line;
            while ( std::getline( stream, line ) )
            {
                corpus_ += line + '\n';
            }
        }

        /**
         * @brief Convenience one-shot preprocessing: load text file, train and
         *        optionally write vocabulary/tokens to disk.
         *
         * Produces <text_file>.vocab and <text_file>.tokens when write_outputs is true.
         *
         * @param text_file Path to input text file.
         * @param force_rebuild If true, always rebuild even if outputs exist.
         * @param write_outputs If true, write .vocab and .tokens files next to the input.
         * @return Pair of (vocab_size, num_tokens)
         */
        static std::pair<size_t, size_t> preprocessFile(
            const std::string& text_file,
            bool force_rebuild = false,
            bool write_outputs = true )
        {
            if ( !fs::exists( text_file ) )
            {
                throw std::runtime_error( "Text file not found: " + text_file );
            }

            std::string vocab_file = text_file + ".vocab";
            std::string tokens_file = text_file + ".tokens";

            if ( !force_rebuild && fs::exists( vocab_file ) && fs::exists( tokens_file ) )
            {
                std::ifstream vf( vocab_file, std::ios::binary );
                if ( !vf )
                {
                    throw std::runtime_error( "Cannot open vocab file: " + vocab_file );
                }

                size_t vocab_size;
                vf.read( reinterpret_cast<char*>( &vocab_size ), sizeof( vocab_size ) );

                std::ifstream tf( tokens_file, std::ios::binary );
                if ( !tf )
                {
                    throw std::runtime_error( "Cannot open tokens file: " + tokens_file );
                }

                size_t num_tokens;
                tf.read( reinterpret_cast<char*>( &num_tokens ), sizeof( num_tokens ) );

                return { vocab_size, num_tokens };
            }

            std::string text = loadText( text_file );

            CharTokenizerTrainer trainer;
            trainer.addCorpus( text );
            auto vocab = trainer.train();

            //auto vocab_ptr = trainer.buildVocabulary();
            //size_t vocab_size = vocab_ptr->getSize();

            //size_t num_tokens = tokenizeAndOptionallySave( text, *vocab_ptr, tokens_file, write_outputs );

            //if ( write_outputs )
            //{
            //    vocab_ptr->save( vocab_file );
            //}

            return { 10, 10 };// FIXME: vocab_size, num_tokens
        }

    private:
        
        static std::string loadText( const std::string& filename )
        {
            std::ifstream file( filename, std::ios::binary );
            if ( !file )
            {
                throw std::runtime_error( "Cannot open text file: " + filename );
            }

            file.seekg( 0, std::ios::end );
            size_t file_size = static_cast<size_t>( file.tellg() );
            file.seekg( 0, std::ios::beg );

            std::string text( file_size, '\0' );
            file.read( &text[0], static_cast<std::streamsize>( file_size ) );

            if ( !file && !file.eof() )
            {
                throw std::runtime_error( "Error reading text file: " + filename );
            }

            return text;
        }

        static size_t tokenizeAndOptionallySave(
            const std::string& text,
            const TokenizerVocabulary& vocab,
            const std::string& tokens_output,
            bool write_output )
        {
            size_t num_tokens = text.size();

            if ( write_output )
            {
                std::ofstream file( tokens_output, std::ios::binary );
                if ( !file )
                {
                    throw std::runtime_error( "Cannot open tokens file for writing: " + tokens_output );
                }

                file.write( reinterpret_cast<const char*>( &num_tokens ), sizeof( num_tokens ) );

                constexpr size_t CHUNK_SIZE = 1024 * 1024;
                std::vector<uint32_t> token_buffer;
                token_buffer.reserve( std::min( num_tokens, CHUNK_SIZE ) );

                for ( size_t i = 0; i < text.size(); ++i )
                {
                    std::string tk( 1, text[i] );
                    auto id_opt = vocab.tokenToId( tk );
                    uint32_t id = id_opt ? *id_opt : 0u;
                    token_buffer.push_back( id );

                    if ( token_buffer.size() >= CHUNK_SIZE || i == text.size() - 1 )
                    {
                        file.write(
                            reinterpret_cast<const char*>( token_buffer.data() ),
                            static_cast<std::streamsize>( token_buffer.size() * sizeof( uint32_t ) ) );

                        token_buffer.clear();
                    }
                }

                if ( !file )
                {
                    throw std::runtime_error( "Error writing tokens file: " + tokens_output );
                }
            }

            return num_tokens;
        }

    private:
        std::string corpus_;
        CharTokenizerVocabulary char_vocab_;
        bool add_special_tokens_;
        bool trained_;
    };
}
