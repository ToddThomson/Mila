/**
 * \file
 * \brief Tokenize utility for training vocabularies and encoding/decoding text corpora.
 *
 * Supports three commands:
 * - train:  Build a vocabulary from a text corpus
 * - encode: Convert text to token IDs using a vocabulary
 * - decode: Convert token IDs back to text using a vocabulary
 *
 * Uses vocabulary factory methods and trainers for different types (char, bpe).
 */
module;
#include <iostream>
#include <string>
#include <string_view>
#include <stdexcept>
#include <exception>
#include <filesystem>
#include <fstream>
#include <vector>
#include <memory>

export module Tokenize.Commands;

import Mila;
export import Tokenize.Args;

// ============================================================================
// Command-line argument structure
// ============================================================================

namespace Tokenize
{
    namespace fs = std::filesystem;
    using namespace Mila::Data;

    // ============================================================================
    // Internal helpers
    // ============================================================================

    static std::string_view tokenizerTypeToString( TokenizerType type )
    {
        switch ( type )
        {
            case TokenizerType::Char: return "char";
            case TokenizerType::Bpe:  return "bpe";
            default:                  return "unknown";
        }
    }

    static void validateTrainArgs( const Args& args )
    {
        if ( args.input_file.empty() )
            throw std::runtime_error( "--input is required for train command" );
        if ( args.output_file.empty() )
            throw std::runtime_error( "--output is required for train command" );
    }

    static void validateEncodeArgs( const Args& args )
    {
        if ( args.vocab_file.empty() )
            throw std::runtime_error( "--vocab is required for encode command" );
        if ( args.input_file.empty() )
            throw std::runtime_error( "--input is required for encode command" );
        if ( args.output_file.empty() )
            throw std::runtime_error( "--output is required for encode command" );
    }

    static void validateDecodeArgs( const Args& args )
    {
        if ( args.vocab_file.empty() )
            throw std::runtime_error( "--vocab is required for decode command" );
        if ( args.input_file.empty() )
            throw std::runtime_error( "--input is required for decode command" );
        if ( args.output_file.empty() )
            throw std::runtime_error( "--output is required for decode command" );
    }

    // ============================================================================
    // Train command
    // ============================================================================

    export int trainCommand( const Args& args )
    {
        validateTrainArgs( args );

        if ( !args.force && fs::exists( args.output_file ) )
        {
            std::cout << "Vocabulary file already exists: " << args.output_file << std::endl;
            std::cout << "Use --force to rebuild" << std::endl;
            return 0;
        }

        std::cout << "Training vocabulary..." << std::endl;
        std::cout << "  Input:      " << args.input_file << std::endl;
        std::cout << "  Output:     " << args.output_file << std::endl;
        std::cout << "  Type:       " << tokenizerTypeToString( args.tokenizer_type ) << std::endl;

        if ( args.tokenizer_type == TokenizerType::Bpe )
        {
            std::cout << "  Vocab size: " << args.vocab_size << std::endl;
            std::cout << "  Min freq:   " << args.min_frequency << std::endl;
        }

        std::cout << "  Byte-level: " << (args.byte_level ? "yes" : "no") << std::endl;

        if ( args.tokenizer_type == TokenizerType::Char )
        {
            std::cout << "  Case-sens:  " << (args.case_sensitive ? "yes" : "no") << std::endl;
        }

        std::cout << std::endl;

        if ( args.tokenizer_type == TokenizerType::Bpe )
        {
            BpeVocabularyConfig config = BpeVocabularyConfig()
                .withVocabSize( args.vocab_size )
                .withMinFrequency( args.min_frequency )
                .withByteLevel( args.byte_level )
                .withSpecialTokens( SpecialTokens::standard() );

            BpeTrainer trainer( config );
            trainer.addCorpusFromFile( args.input_file );

            BpeVocabulary vocab = trainer.train();
            vocab.save( args.output_file );

            std::cout << "Training complete" << std::endl;
            std::cout << "  Vocabulary size: " << vocab.getSize() << std::endl;
            std::cout << "  Saved to: " << args.output_file << std::endl;
        }
        else if ( args.tokenizer_type == TokenizerType::Char )
        {
            CharVocabularyConfig config = CharVocabularyConfig()
                .withCaseSensitive( args.case_sensitive )
                .withByteLevel( args.byte_level )
                .withSpecialTokens( SpecialTokens::standard() );

            CharVocabulary vocab = CharVocabulary::trainFromFile( args.input_file, config );
            vocab.save( args.output_file );

            std::cout << "Training complete" << std::endl;
            std::cout << "  Vocabulary size: " << vocab.getSize() << std::endl;
            std::cout << "  Saved to: " << args.output_file << std::endl;
        }
        else
        {
            throw std::runtime_error( "Unsupported tokenizer type" );
        }

        return 0;
    }

    // ============================================================================
    // Encode command
    // ============================================================================

    export int encodeCommand( const Args& args )
    {
        validateEncodeArgs( args );

        std::cout << "Encoding corpus..." << std::endl;
        std::cout << "  Vocabulary: " << args.vocab_file << std::endl;
        std::cout << "  Input:      " << args.input_file << std::endl;
        std::cout << "  Output:     " << args.output_file << std::endl;
        std::cout << "  Type:       " << tokenizerTypeToString( args.tokenizer_type ) << std::endl;
        std::cout << std::endl;

        std::unique_ptr<Tokenizer> tokenizer;

        if ( args.tokenizer_type == TokenizerType::Bpe )
        {
            BpeVocabulary vocab = BpeVocabulary::load( args.vocab_file );
            std::cout << "  Loaded vocabulary: " << vocab.getSize() << " tokens" << std::endl;
            tokenizer = std::make_unique<BpeTokenizer>( std::move( vocab ) );
        }
        else if ( args.tokenizer_type == TokenizerType::Char )
        {
            CharVocabulary vocab = CharVocabulary::load( args.vocab_file );
            std::cout << "  Loaded vocabulary: " << vocab.getSize() << " tokens" << std::endl;
            tokenizer = std::make_unique<CharTokenizer>( std::move( vocab ) );
        }
        else
        {
            throw std::runtime_error( "Unsupported tokenizer type" );
        }

        std::ifstream input( args.input_file, std::ios::binary );
        if ( !input )
        {
            throw std::runtime_error( "Cannot open input file: " + args.input_file );
        }

        input.seekg( 0, std::ios::end );
        size_t file_size = static_cast<size_t>(input.tellg());
        input.seekg( 0, std::ios::beg );

        std::string text( file_size, '\0' );
        input.read( text.data(), static_cast<std::streamsize>(file_size) );

        if ( !input && !input.eof() )
        {
            throw std::runtime_error( "Error reading input file: " + args.input_file );
        }

        std::cout << "  Read " << text.size() << " characters" << std::endl;

        auto tokens = tokenizer->encode( text );
        std::cout << "  Encoded to " << tokens.size() << " tokens" << std::endl;

        std::ofstream output( args.output_file, std::ios::binary );
        if ( !output )
        {
            throw std::runtime_error( "Cannot open output file: " + args.output_file );
        }

        size_t num_tokens = tokens.size();
        output.write( reinterpret_cast<const char*>(&num_tokens), sizeof( num_tokens ) );
        output.write( reinterpret_cast<const char*>(tokens.data()),
            static_cast<std::streamsize>(tokens.size() * sizeof( TokenId )) );

        if ( !output )
        {
            throw std::runtime_error( "Error writing output file: " + args.output_file );
        }

        std::cout << "Encoding complete!" << std::endl;
        std::cout << "  Tokens saved to: " << args.output_file << std::endl;

        return 0;
    }

    // ============================================================================
    // Decode command
    // ============================================================================

    export int decodeCommand( const Args& args )
    {
        validateDecodeArgs( args );

        std::cout << "Decoding tokens..." << std::endl;
        std::cout << "  Vocabulary: " << args.vocab_file << std::endl;
        std::cout << "  Input:      " << args.input_file << std::endl;
        std::cout << "  Output:     " << args.output_file << std::endl;
        std::cout << "  Type:       " << tokenizerTypeToString( args.tokenizer_type ) << std::endl;
        std::cout << std::endl;

        std::unique_ptr<Tokenizer> tokenizer;

        if ( args.tokenizer_type == TokenizerType::Bpe )
        {
            BpeVocabulary vocab = BpeVocabulary::load( args.vocab_file );
            std::cout << "  Loaded vocabulary: " << vocab.getSize() << " tokens" << std::endl;
            tokenizer = std::make_unique<BpeTokenizer>( std::move( vocab ) );
        }
        else if ( args.tokenizer_type == TokenizerType::Char )
        {
            CharVocabulary vocab = CharVocabulary::load( args.vocab_file );
            std::cout << "  Loaded vocabulary: " << vocab.getSize() << " tokens" << std::endl;
            tokenizer = std::make_unique<CharTokenizer>( std::move( vocab ) );
        }
        else
        {
            throw std::runtime_error( "Unsupported tokenizer type" );
        }

        std::ifstream input( args.input_file, std::ios::binary );
        if ( !input )
        {
            throw std::runtime_error( "Cannot open input file: " + args.input_file );
        }

        size_t num_tokens;
        input.read( reinterpret_cast<char*>(&num_tokens), sizeof( num_tokens ) );
        if ( !input )
        {
            throw std::runtime_error( "Error reading token count from: " + args.input_file );
        }

        std::vector<TokenId> tokens( num_tokens );
        input.read( reinterpret_cast<char*>(tokens.data()),
            static_cast<std::streamsize>(num_tokens * sizeof( TokenId )) );

        if ( !input && !input.eof() )
        {
            throw std::runtime_error( "Error reading tokens from: " + args.input_file );
        }

        std::cout << "  Read " << num_tokens << " tokens" << std::endl;

        auto text = tokenizer->decode( tokens );
        std::cout << "  Decoded to " << text.size() << " characters" << std::endl;

        std::ofstream output( args.output_file, std::ios::binary );
        if ( !output )
        {
            throw std::runtime_error( "Cannot open output file: " + args.output_file );
        }

        output.write( text.data(), static_cast<std::streamsize>(text.size()) );

        if ( !output )
        {
            throw std::runtime_error( "Error writing output file: " + args.output_file );
        }

        std::cout << "Decoding complete!" << std::endl;
        std::cout << "  Text saved to: " << args.output_file << std::endl;

        return 0;
    }
}