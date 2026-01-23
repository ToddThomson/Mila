/**
 * \file
 * \brief Generic tokenize utility that can preprocess text using different tokenizers.
 *
 * Uses TokenizerFactory to obtain a trainer and vocabulary loader for the
 * selected TokenizerType. Currently supports `char` tokenizer.
 */

#include <iostream>
#include <string>
#include <stdexcept>
#include <exception>
#include <filesystem>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <optional>

import Data.TokenizerFactory;
import Data.TokenizerType;
import Data.TokenizerVocabulary;

namespace fs = std::filesystem;
using namespace Mila::Dnn::Data;

static void printUsage()
{
    std::cout << "Usage: Tokenize <input_text_file> [--force] [--tokenizer <char|bpe>]" << std::endl;
    std::cout << std::endl;
    std::cout << "Preprocesses text file producing <input>.vocab and <input>.tokens." << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --force            Force rebuild even if preprocessed files exist" << std::endl;
    std::cout << "  --tokenizer <type> Tokenizer type (default: char). Supported: char" << std::endl;
    std::cout << "  --help             Show this help message" << std::endl;
}

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

static bool isPreprocessedUpToDate( const fs::path& src, const fs::path& vocab, const fs::path& tokens )
{
    if ( !fs::exists( vocab ) || !fs::exists( tokens ) )
    {
        return false;
    }

    auto src_time = fs::last_write_time( src );
    auto vocab_time = fs::last_write_time( vocab );
    auto tokens_time = fs::last_write_time( tokens );

    return vocab_time >= src_time && tokens_time >= src_time;
}

int main( int argc, char** argv )
{
    try
    {
        std::string input_file;
        bool force_rebuild = false;
        TokenizerType tokenizer_type = TokenizerType::Char;

        for ( int i = 1; i < argc; ++i )
        {
            std::string arg = argv[i];

            if ( arg == "--help" || arg == "-h" )
            {
                printUsage();
                return 0;
            }
            else if ( arg == "--force" || arg == "-f" )
            {
                force_rebuild = true;
            }
            else if ( arg == "--tokenizer" && i + 1 < argc )
            {
                ++i;
                tokenizer_type = from_string( argv[i] );
                if ( tokenizer_type == TokenizerType::Unknown )
                {
                    std::cerr << "Unknown tokenizer: " << argv[i] << std::endl;
                    return 1;
                }
            }
            else if ( input_file.empty() )
            {
                input_file = arg;
            }
            else
            {
                std::cerr << "Unknown argument: " << arg << std::endl;
                printUsage();
                return 1;
            }
        }

        if ( input_file.empty() )
        {
            input_file = "../Data/DataSets/TinyShakespeare/input.txt";
            std::cout << "No input file specified. Using default: " << input_file << std::endl;
        }

        std::cout << "Preprocessing: " << input_file << std::endl;
        std::cout << "Force rebuild: " << (force_rebuild ? "yes" : "no") << std::endl;
        std::cout << "Tokenizer: " << to_string( tokenizer_type ) << std::endl;
        std::cout << std::endl;

        fs::path vocab_file = input_file + ".vocab";
        fs::path tokens_file = input_file + ".tokens";

        if ( !force_rebuild && isPreprocessedUpToDate( input_file, vocab_file, tokens_file ) )
        {
            std::cout << "Using existing preprocessed files for: " << input_file << std::endl;

            std::ifstream vf( vocab_file, std::ios::binary );
            if ( !vf ) throw std::runtime_error( "Cannot open vocab file: " + vocab_file.string() );

            size_t vocab_size;
            vf.read( reinterpret_cast<char*>( &vocab_size ), sizeof( vocab_size ) );

            std::ifstream tf( tokens_file, std::ios::binary );
            if ( !tf ) throw std::runtime_error( "Cannot open tokens file: " + tokens_file.string() );

            size_t num_tokens;
            tf.read( reinterpret_cast<char*>( &num_tokens ), sizeof( num_tokens ) );

            std::cout << std::endl;
            std::cout << "Preprocessing complete (existing files used)!" << std::endl;
            std::cout << "  Vocabulary size: " << vocab_size << std::endl;
            std::cout << "  Total tokens: " << num_tokens << std::endl;
            return 0;
        }

        // Load raw text
        std::string text = loadText( input_file );
        std::cout << "  Loaded " << text.size() << " characters" << std::endl;

        // Create trainer via factory
        auto trainer = TokenizerFactory::createTrainer( tokenizer_type );
        if ( !trainer )
        {
            throw std::runtime_error( "No trainer available for tokenizer type: " + std::string( to_string( tokenizer_type ) ) );
        }

        trainer->addCorpus( text );
        trainer->train();

        auto vocab_ptr = trainer->buildVocabulary();
        if ( !vocab_ptr )
        {
            throw std::runtime_error( "Trainer failed to produce a vocabulary" );
        }

        size_t vocab_size = vocab_ptr->getSize();
        std::cout << "  Built vocabulary: " << vocab_size << " tokens" << std::endl;

        // Save vocabulary
        vocab_ptr->save( vocab_file );
        std::cout << "  Saved vocabulary to: " << vocab_file.string() << std::endl;

        // Tokenize and save tokens file (header: size_t num_tokens, then uint32_t ids)
        size_t num_tokens = text.size();

        std::ofstream out( tokens_file, std::ios::binary );
        if ( !out ) throw std::runtime_error( "Cannot open tokens file for writing: " + tokens_file.string() );

        out.write( reinterpret_cast<const char*>( &num_tokens ), sizeof( num_tokens ) );

        constexpr size_t CHUNK_SIZE = 1024 * 1024;
        std::vector<uint32_t> buffer;
        buffer.reserve( std::min( num_tokens, CHUNK_SIZE ) );

        for ( size_t i = 0; i < text.size(); ++i )
        {
            std::string tk( 1, text[i] );
            auto id_opt = vocab_ptr->tokenToId( tk );
            uint32_t id = id_opt ? *id_opt : 0u;
            buffer.push_back( id );

            if ( buffer.size() >= CHUNK_SIZE || i == text.size() - 1 )
            {
                out.write( reinterpret_cast<const char*>( buffer.data() ), static_cast<std::streamsize>( buffer.size() * sizeof( uint32_t ) ) );
                buffer.clear();
            }
        }

        if ( !out ) throw std::runtime_error( "Error writing tokens file: " + tokens_file.string() );

        std::cout << "  Tokenized and saved " << num_tokens << " tokens to: " << tokens_file.string() << std::endl;

        std::cout << std::endl;
        std::cout << "Preprocessing complete!" << std::endl;
        std::cout << "  Vocabulary size: " << vocab_size << std::endl;
        std::cout << "  Total tokens: " << num_tokens << std::endl;

        return 0;
    }
    catch ( const std::exception& e )
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
