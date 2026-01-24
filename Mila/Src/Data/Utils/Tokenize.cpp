/**
 * \file
 * \brief Tokenize utility for training vocabularies and encoding/decoding text corpora.
 *
 * Supports three commands:
 * - train:  Build a vocabulary from a text corpus
 * - encode: Convert text to token IDs using a vocabulary
 * - decode: Convert token IDs back to text using a vocabulary
 *
 * Uses TokenizerFactory to obtain trainers and tokenizers for different types (char, bpe).
 */

#include <iostream>
#include <string>
#include <string_view>
#include <stdexcept>
#include <exception>
#include <filesystem>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <optional>
#include <memory>

import Data.TokenizerFactory;
import Data.TokenizerType;
import Data.TokenizerVocabulary;
import Data.Tokenizer;

namespace fs = std::filesystem;
using namespace Mila::Data;
using namespace Mila::Dnn::Data;

// ============================================================================
// Command-line argument structure
// ============================================================================

enum class Command {
    Unknown,
    Train,
    Encode,
    Decode,
    Help
};

struct Args {
    Command command = Command::Unknown;
    std::string input_file;
    std::string output_file;
    std::string vocab_file;
    TokenizerType tokenizer_type = TokenizerType::Char;
    size_t vocab_size = 0;  // For BPE training
    bool force = false;
};

// ============================================================================
// Usage and help
// ============================================================================

static void printUsage()
{
    std::cout << R"(Usage: tokenize <command> [options]

Commands:
  train   Build a vocabulary from a text corpus
  encode  Convert text to token IDs using a vocabulary
  decode  Convert token IDs back to text using a vocabulary
  help    Show this help message

Train Options:
  --input <file>       Input text corpus file (required)
  --output <file>      Output vocabulary file (required)
  --type <char|bpe>    Tokenizer type (default: char)
  --vocab-size <n>     Vocabulary size for BPE (required for BPE)
  --force              Force rebuild even if output exists

Encode Options:
  --vocab <file>       Vocabulary file (required)
  --input <file>       Input text file (required)
  --output <file>      Output tokens file (required)
  --type <char|bpe>    Tokenizer type (default: char)

Decode Options:
  --vocab <file>       Vocabulary file (required)
  --input <file>       Input tokens file (required)
  --output <file>      Output text file (required)
  --type <char|bpe>    Tokenizer type (default: char)

Examples:
  # Train a character-level vocabulary
  tokenize train --input corpus.txt --output vocab.bin --type char

  # Train a BPE vocabulary with 32000 tokens
  tokenize train --input corpus.txt --output vocab.bin --type bpe --vocab-size 32000

  # Encode text using the vocabulary
  tokenize encode --vocab vocab.bin --input train.txt --output train.tokens --type char

  # Decode tokens back to text
  tokenize decode --vocab vocab.bin --input train.tokens --output train.txt --type char
)";
}

static Command commandFromString( std::string_view str )
{
    if ( str == "train" ) return Command::Train;
    if ( str == "encode" ) return Command::Encode;
    if ( str == "decode" ) return Command::Decode;
    if ( str == "help" || str == "--help" || str == "-h" ) return Command::Help;
    return Command::Unknown;
}

// ============================================================================
// Argument parsing
// ============================================================================

static Args parseArgs( int argc, char** argv )
{
    Args args;

    if ( argc < 2 ) {
        return args;  // Will trigger help
    }

    args.command = commandFromString( argv[ 1 ] );

    for ( int i = 2; i < argc; ++i ) {
        std::string arg = argv[ i ];

        if ( arg == "--input" && i + 1 < argc ) {
            args.input_file = argv[ ++i ];
        }
        else if ( arg == "--output" && i + 1 < argc ) {
            args.output_file = argv[ ++i ];
        }
        else if ( arg == "--vocab" && i + 1 < argc ) {
            args.vocab_file = argv[ ++i ];
        }
        else if ( arg == "--type" && i + 1 < argc ) {
            args.tokenizer_type = from_string( argv[ ++i ] );
            if ( args.tokenizer_type == TokenizerType::Unknown ) {
                throw std::runtime_error( "Unknown tokenizer type: " + std::string( argv[ i ] ) );
            }
        }
        else if ( arg == "--vocab-size" && i + 1 < argc ) {
            args.vocab_size = std::stoull( argv[ ++i ] );
        }
        else if ( arg == "--force" || arg == "-f" ) {
            args.force = true;
        }
        else {
            throw std::runtime_error( "Unknown argument: " + arg );
        }
    }

    return args;
}

static void validateTrainArgs( const Args& args )
{
    if ( args.input_file.empty() ) {
        throw std::runtime_error( "--input is required for train command" );
    }
    if ( args.output_file.empty() ) {
        throw std::runtime_error( "--output is required for train command" );
    }
    if ( args.tokenizer_type == TokenizerType::Bpe && args.vocab_size == 0 ) {
        throw std::runtime_error( "--vocab-size is required for BPE tokenizer" );
    }
}

static void validateEncodeArgs( const Args& args )
{
    if ( args.vocab_file.empty() ) {
        throw std::runtime_error( "--vocab is required for encode command" );
    }
    if ( args.input_file.empty() ) {
        throw std::runtime_error( "--input is required for encode command" );
    }
    if ( args.output_file.empty() ) {
        throw std::runtime_error( "--output is required for encode command" );
    }
}

static void validateDecodeArgs( const Args& args )
{
    if ( args.vocab_file.empty() ) {
        throw std::runtime_error( "--vocab is required for decode command" );
    }
    if ( args.input_file.empty() ) {
        throw std::runtime_error( "--input is required for decode command" );
    }
    if ( args.output_file.empty() ) {
        throw std::runtime_error( "--output is required for decode command" );
    }
}

// ============================================================================
// Train command
// ============================================================================

static int trainCommand( const Args& args )
{
    validateTrainArgs( args );

    // Check if output exists and skip if not forcing
    if ( !args.force && fs::exists( args.output_file ) ) {
        std::cout << "Vocabulary file already exists: " << args.output_file << std::endl;
        std::cout << "Use --force to rebuild" << std::endl;
        return 0;
    }

    std::cout << "Training vocabulary..." << std::endl;
    std::cout << "  Input:      " << args.input_file << std::endl;
    std::cout << "  Output:     " << args.output_file << std::endl;
    std::cout << "  Type:       " << to_string( args.tokenizer_type ) << std::endl;
    if ( args.vocab_size > 0 ) {
        std::cout << "  Vocab size: " << args.vocab_size << std::endl;
    }
    std::cout << std::endl;

    // Create trainer
    auto trainer = TokenizerFactory::createTrainer( args.tokenizer_type );
    if ( !trainer ) {
        throw std::runtime_error( "Failed to create trainer for type: " +
            std::string( to_string( args.tokenizer_type ) ) );
    }

    // Add corpus from file
    trainer->addCorpusFromFile( args.input_file );

    // Train vocabulary
    auto vocab = trainer->train();
    if ( !vocab ) {
        throw std::runtime_error( "Training failed to produce vocabulary" );
    }

    // Save vocabulary
    vocab->save( args.output_file );

    std::cout << "Training complete!" << std::endl;
    std::cout << "  Vocabulary size: " << vocab->getSize() << std::endl;
    std::cout << "  Saved to: " << args.output_file << std::endl;

    return 0;
}

// ============================================================================
// Encode command
// ============================================================================

static int encodeCommand( const Args& args )
{
    validateEncodeArgs( args );

    std::cout << "Encoding corpus..." << std::endl;
    std::cout << "  Vocabulary: " << args.vocab_file << std::endl;
    std::cout << "  Input:      " << args.input_file << std::endl;
    std::cout << "  Output:     " << args.output_file << std::endl;
    std::cout << "  Type:       " << to_string( args.tokenizer_type ) << std::endl;
    std::cout << std::endl;

    // Load vocabulary
    auto vocab = TokenizerFactory::createVocabulary( args.tokenizer_type );
    if ( !vocab ) {
        throw std::runtime_error( "Failed to create vocabulary for type: " +
            std::string( to_string( args.tokenizer_type ) ) );
    }
    vocab->load( args.vocab_file );

    std::cout << "  Loaded vocabulary: " << vocab->getSize() << " tokens" << std::endl;

    // Create tokenizer
    auto tokenizer = TokenizerFactory::createTokenizer( args.tokenizer_type, vocab );
    if ( !tokenizer ) {
        throw std::runtime_error( "Failed to create tokenizer for type: " +
            std::string( to_string( args.tokenizer_type ) ) );
    }

    // Read input text
    std::ifstream input( args.input_file, std::ios::binary );
    if ( !input ) {
        throw std::runtime_error( "Cannot open input file: " + args.input_file );
    }

    input.seekg( 0, std::ios::end );
    size_t file_size = static_cast<size_t>(input.tellg());
    input.seekg( 0, std::ios::beg );

    std::string text( file_size, '\0' );
    input.read( text.data(), static_cast<std::streamsize>(file_size) );

    if ( !input && !input.eof() ) {
        throw std::runtime_error( "Error reading input file: " + args.input_file );
    }

    std::cout << "  Read " << text.size() << " characters" << std::endl;

    // Encode text
    auto tokens = tokenizer->encode( text );
    std::cout << "  Encoded to " << tokens.size() << " tokens" << std::endl;

    // Write tokens file: [size_t num_tokens][uint32_t token_ids...]
    std::ofstream output( args.output_file, std::ios::binary );
    if ( !output ) {
        throw std::runtime_error( "Cannot open output file: " + args.output_file );
    }

    size_t num_tokens = tokens.size();
    output.write( reinterpret_cast<const char*>(&num_tokens), sizeof( num_tokens ) );
    output.write( reinterpret_cast<const char*>(tokens.data()),
        static_cast<std::streamsize>(tokens.size() * sizeof( uint32_t )) );

    if ( !output ) {
        throw std::runtime_error( "Error writing output file: " + args.output_file );
    }

    std::cout << "Encoding complete!" << std::endl;
    std::cout << "  Tokens saved to: " << args.output_file << std::endl;

    return 0;
}

// ============================================================================
// Decode command
// ============================================================================

static int decodeCommand( const Args& args )
{
    validateDecodeArgs( args );

    std::cout << "Decoding tokens..." << std::endl;
    std::cout << "  Vocabulary: " << args.vocab_file << std::endl;
    std::cout << "  Input:      " << args.input_file << std::endl;
    std::cout << "  Output:     " << args.output_file << std::endl;
    std::cout << "  Type:       " << to_string( args.tokenizer_type ) << std::endl;
    std::cout << std::endl;

    // Load vocabulary
    auto vocab = TokenizerFactory::createVocabulary( args.tokenizer_type );
    if ( !vocab ) {
        throw std::runtime_error( "Failed to create vocabulary for type: " +
            std::string( to_string( args.tokenizer_type ) ) );
    }
    vocab->load( args.vocab_file );

    std::cout << "  Loaded vocabulary: " << vocab->getSize() << " tokens" << std::endl;

    // Create tokenizer
    auto tokenizer = TokenizerFactory::createTokenizer( args.tokenizer_type, vocab );
    if ( !tokenizer ) {
        throw std::runtime_error( "Failed to create tokenizer for type: " +
            std::string( to_string( args.tokenizer_type ) ) );
    }

    // Read tokens file: [size_t num_tokens][uint32_t token_ids...]
    std::ifstream input( args.input_file, std::ios::binary );
    if ( !input ) {
        throw std::runtime_error( "Cannot open input file: " + args.input_file );
    }

    size_t num_tokens;
    input.read( reinterpret_cast<char*>(&num_tokens), sizeof( num_tokens ) );
    if ( !input ) {
        throw std::runtime_error( "Error reading token count from: " + args.input_file );
    }

    std::vector<uint32_t> tokens( num_tokens );
    input.read( reinterpret_cast<char*>(tokens.data()),
        static_cast<std::streamsize>(num_tokens * sizeof( uint32_t )) );

    if ( !input && !input.eof() ) {
        throw std::runtime_error( "Error reading tokens from: " + args.input_file );
    }

    std::cout << "  Read " << num_tokens << " tokens" << std::endl;

    // Decode tokens
    auto text = tokenizer->decode( tokens );
    std::cout << "  Decoded to " << text.size() << " characters" << std::endl;

    // Write output text
    std::ofstream output( args.output_file, std::ios::binary );
    if ( !output ) {
        throw std::runtime_error( "Cannot open output file: " + args.output_file );
    }

    output.write( text.data(), static_cast<std::streamsize>(text.size()) );

    if ( !output ) {
        throw std::runtime_error( "Error writing output file: " + args.output_file );
    }

    std::cout << "Decoding complete!" << std::endl;
    std::cout << "  Text saved to: " << args.output_file << std::endl;

    return 0;
}

// ============================================================================
// Main
// ============================================================================

int main( int argc, char** argv )
{
    try {
        Args args = parseArgs( argc, argv );

        switch ( args.command ) {
            case Command::Train:
                return trainCommand( args );

            case Command::Encode:
                return encodeCommand( args );

            case Command::Decode:
                return decodeCommand( args );

            case Command::Help:
                printUsage();
                return 0;

            case Command::Unknown:
            default:
                std::cerr << "Error: Unknown or missing command" << std::endl << std::endl;
                printUsage();
                return 1;
        }
    }
    catch ( const std::exception& e ) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}