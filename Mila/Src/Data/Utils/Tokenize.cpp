/**
 * \file
 * \brief Tokenize utility for training vocabularies and encoding/decoding text corpora.
 *
 * Supports three commands:
 * - train:  Build a vocabulary from a text corpus
 * - encode: Convert text to token IDs using a vocabulary
 * - decode: Convert token IDs back to text using a vocabulary
 *
 * Uses TrainerFactory to obtain trainers and tokenizers for different types (char, bpe).
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

import Data.TrainerFactory;
import Data.BpeTrainer;
import Data.CharTrainer;
import Data.Tokenizer;
import Data.BpeTokenizer;
import Data.CharTokenizer;

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

enum class TokenizerType {
    Unknown,
    Char,
    Bpe
};

struct Args {
    Command command = Command::Unknown;
    std::string input_file;
    std::string output_file;
    std::string vocab_file;
    TokenizerType tokenizer_type = TokenizerType::Char;
    size_t vocab_size = 30000;      // Default for BPE
    size_t min_frequency = 2;       // Default for BPE
    bool byte_level = false;        // Default for BPE/Char
    bool case_sensitive = true;     // Default for Char
    bool force = false;
};

// ============================================================================
// Helper functions
// ============================================================================

static std::string_view tokenizerTypeToString( TokenizerType type )
{
    switch ( type ) {
        case TokenizerType::Char: return "char";
        case TokenizerType::Bpe:  return "bpe";
        default:                  return "unknown";
    }
}

static TokenizerType tokenizerTypeFromString( std::string_view str )
{
    if ( str == "char" ) return TokenizerType::Char;
    if ( str == "bpe" )  return TokenizerType::Bpe;
    return TokenizerType::Unknown;
}

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
  --vocab-size <n>     Vocabulary size for BPE (default: 30000)
  --min-freq <n>       Minimum frequency for BPE merges (default: 2)
  --byte-level         Use byte-level encoding
  --case-sensitive     Use case-sensitive encoding (char only, default: true)
  --case-insensitive   Use case-insensitive encoding (char only)
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

  # Train a case-insensitive character vocabulary
  tokenize train --input corpus.txt --output vocab.bin --type char --case-insensitive

  # Train a BPE vocabulary with 32000 tokens
  tokenize train --input corpus.txt --output vocab.bin --type bpe --vocab-size 32000

  # Train a byte-level BPE vocabulary
  tokenize train --input corpus.txt --output vocab.bin --type bpe --byte-level

  # Encode text using the vocabulary
  tokenize encode --vocab vocab.bin --input train.txt --output train.tokens --type char

  # Decode tokens back to text
  tokenize decode --vocab vocab.bin --input train.tokens --output train.txt --type char
)";
}

static Command commandFromString( std::string_view str )
{
    if ( str == "train" )  return Command::Train;
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
            args.tokenizer_type = tokenizerTypeFromString( argv[ ++i ] );
            if ( args.tokenizer_type == TokenizerType::Unknown ) {
                throw std::runtime_error( "Unknown tokenizer type: " + std::string( argv[ i ] ) );
            }
        }
        else if ( arg == "--vocab-size" && i + 1 < argc ) {
            args.vocab_size = std::stoull( argv[ ++i ] );
        }
        else if ( arg == "--min-freq" && i + 1 < argc ) {
            args.min_frequency = std::stoull( argv[ ++i ] );
        }
        else if ( arg == "--byte-level" ) {
            args.byte_level = true;
        }
        else if ( arg == "--case-sensitive" ) {
            args.case_sensitive = true;
        }
        else if ( arg == "--case-insensitive" ) {
            args.case_sensitive = false;
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
    std::cout << "  Type:       " << tokenizerTypeToString( args.tokenizer_type ) << std::endl;

    if ( args.tokenizer_type == TokenizerType::Bpe ) {
        std::cout << "  Vocab size: " << args.vocab_size << std::endl;
        std::cout << "  Min freq:   " << args.min_frequency << std::endl;
    }

    std::cout << "  Byte-level: " << (args.byte_level ? "yes" : "no") << std::endl;

    if ( args.tokenizer_type == TokenizerType::Char ) {
        std::cout << "  Case-sens:  " << (args.case_sensitive ? "yes" : "no") << std::endl;
    }

    std::cout << std::endl;

    // Train based on type
    if ( args.tokenizer_type == TokenizerType::Bpe ) {
        // Create BPE config using fluent API
        BpeTrainerConfig config;
        config.withVocabSize( args.vocab_size )
            .withMinFrequency( args.min_frequency )
            .withByteLevel( args.byte_level );
            // FIXME: .withSpecialTokens( { "<|endoftext|>", "<|pad|>" } );

        // Create trainer and add corpus
        BpeTrainer trainer( config );
        trainer.addCorpusFromFile( args.input_file );

        // Train and save
        BpeVocabulary vocab = trainer.train();
        vocab.save( args.output_file );

        std::cout << "Training complete!" << std::endl;
        std::cout << "  Vocabulary size: " << vocab.getSize() << std::endl;
        std::cout << "  Saved to: " << args.output_file << std::endl;
    }
    else if ( args.tokenizer_type == TokenizerType::Char ) {
        // Create Char config using fluent API
        CharTrainerConfig config;
        config.withCaseSensitive( args.case_sensitive )
            .withByteLevel( args.byte_level );
            // FIXME:.withSpecialTokens( { "<PAD>", "<UNK>", "<BOS>", "<EOS>" } );

        // Create trainer and add corpus
        CharTrainer trainer( config );
        trainer.addCorpusFromFile( args.input_file );

        // Train and save
        CharVocabulary vocab = trainer.train();
        vocab.save( args.output_file );

        std::cout << "Training complete!" << std::endl;
        std::cout << "  Vocabulary size: " << vocab.size() << std::endl;
        std::cout << "  Saved to: " << args.output_file << std::endl;
    }
    else {
        throw std::runtime_error( "Unsupported tokenizer type" );
    }

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
    std::cout << "  Type:       " << tokenizerTypeToString( args.tokenizer_type ) << std::endl;
    std::cout << std::endl;

    // Load tokenizer based on type
    std::unique_ptr<Tokenizer> tokenizer;

    if ( args.tokenizer_type == TokenizerType::Bpe ) {
        BpeVocabulary vocab = BpeVocabulary::load( args.vocab_file );
        std::cout << "  Loaded vocabulary: " << vocab.getSize() << " tokens" << std::endl;
        tokenizer = std::make_unique<BpeTokenizer>( std::move( vocab ) );
    }
    else if ( args.tokenizer_type == TokenizerType::Char ) {
        CharVocabulary vocab = CharVocabulary::load( args.vocab_file );
        std::cout << "  Loaded vocabulary: " << vocab.size() << " tokens" << std::endl;
        tokenizer = std::make_unique<CharTokenizer>( std::move( vocab ) );
    }
    else {
        throw std::runtime_error( "Unsupported tokenizer type" );
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
        static_cast<std::streamsize>(tokens.size() * sizeof( TokenId )) );

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
    std::cout << "  Type:       " << tokenizerTypeToString( args.tokenizer_type ) << std::endl;
    std::cout << std::endl;

    // Load tokenizer based on type
    std::unique_ptr<Tokenizer> tokenizer;

    if ( args.tokenizer_type == TokenizerType::Bpe ) {
        BpeVocabulary vocab = BpeVocabulary::load( args.vocab_file );
        std::cout << "  Loaded vocabulary: " << vocab.getSize() << " tokens" << std::endl;
        tokenizer = std::make_unique<BpeTokenizer>( std::move( vocab ) );
    }
    else if ( args.tokenizer_type == TokenizerType::Char ) {
        CharVocabulary vocab = CharVocabulary::load( args.vocab_file );
        std::cout << "  Loaded vocabulary: " << vocab.getSize() << " tokens" << std::endl;
        tokenizer = std::make_unique<CharTokenizer>( std::move( vocab ) );
    }
    else {
        throw std::runtime_error( "Unsupported tokenizer type" );
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

    std::vector<TokenId> tokens( num_tokens );
    input.read( reinterpret_cast<char*>(tokens.data()),
        static_cast<std::streamsize>(num_tokens * sizeof( TokenId )) );

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