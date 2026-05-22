
#include <iostream>
#include <string>
#include <string_view>
#include <stdexcept>
#include <exception>

import Tokenize.Commands;

// ============================================================================
// Helper functions
// ============================================================================

namespace Tokenize
{

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

        if ( argc < 2 )
        {
            return args;
        }

        args.command = commandFromString( argv[ 1 ] );

        for ( int i = 2; i < argc; ++i )
        {
            std::string arg = argv[ i ];

            if ( arg == "--input" && i + 1 < argc )
            {
                args.input_file = argv[ ++i ];
            }
            else if ( arg == "--output" && i + 1 < argc )
            {
                args.output_file = argv[ ++i ];
            }
            else if ( arg == "--vocab" && i + 1 < argc )
            {
                args.vocab_file = argv[ ++i ];
            }
            else if ( arg == "--type" && i + 1 < argc )
            {
                args.tokenizer_type = tokenizerTypeFromString( argv[ ++i ] );
                if ( args.tokenizer_type == TokenizerType::Unknown )
                {
                    throw std::runtime_error( "Unknown tokenizer type: " + std::string( argv[ i ] ) );
                }
            }
            else if ( arg == "--vocab-size" && i + 1 < argc )
            {
                args.vocab_size = std::stoull( argv[ ++i ] );
            }
            else if ( arg == "--min-freq" && i + 1 < argc )
            {
                args.min_frequency = std::stoull( argv[ ++i ] );
            }
            else if ( arg == "--byte-level" )
            {
                args.byte_level = true;
            }
            else if ( arg == "--case-sensitive" )
            {
                args.case_sensitive = true;
            }
            else if ( arg == "--case-insensitive" )
            {
                args.case_sensitive = false;
            }
            else if ( arg == "--force" || arg == "-f" )
            {
                args.force = true;
            }
            else
            {
                throw std::runtime_error( "Unknown argument: " + arg );
            }
        }

        return args;
    }

}

// ============================================================================
// Main
// ============================================================================

int main( int argc, char** argv )
{
    using namespace Tokenize;
    try
    {
        Args args = parseArgs( argc, argv );

        switch ( args.command )
        {
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
    catch ( const std::exception& e )
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}