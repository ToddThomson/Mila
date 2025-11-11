/**
 * @file CharPreprocessor.ixx
 * @brief Preprocessing pipeline for character-level language modeling.
 *
 * Handles vocabulary construction, tokenization, and serialization
 * for efficient data loading during training.
 */

module;
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <utility>

export module CharLM.Preprocessor;

import CharLM.Vocabulary;

namespace Mila::CharLM
{
    namespace fs = std::filesystem;

    /**
     * @brief Preprocessor for character-level text data.
     *
     * Performs one-time preprocessing of text data:
     * 1. Loads raw text file
     * 2. Builds character vocabulary
     * 3. Tokenizes entire corpus
     * 4. Saves vocabulary and tokens to disk
     *
     * Output files:
     * - <input>.vocab - Binary vocabulary file
     * - <input>.tokens - Binary tokenized data
     *
     * These files can be efficiently loaded by CharDataLoader without
     * repeating expensive preprocessing steps.
     */
    export class CharPreprocessor
    {
    public:
        /**
         * @brief Preprocesses text file for language modeling.
         *
         * Creates vocabulary and tokenized data files if they don't exist
         * or if force_rebuild is true. Automatically detects if preprocessing
         * is needed based on file timestamps.
         *
         * @param text_file Path to input text file
         * @param force_rebuild Force rebuild even if preprocessed files exist
         * @param add_special_tokens Add padding/unknown tokens to vocabulary
         * @return Pair of (vocab_size, num_tokens)
         */
        static std::pair<size_t, size_t> preprocess(
            const std::string& text_file,
            bool force_rebuild = false,
            bool add_special_tokens = false )
        {
            if (!fs::exists( text_file ))
            {
                throw std::runtime_error( "Text file not found: " + text_file );
            }

            std::string vocab_file = text_file + ".vocab";
            std::string tokens_file = text_file + ".tokens";

            // Check if preprocessing is needed
            if (!force_rebuild && isPreprocessed( text_file, vocab_file, tokens_file ))
            {
                std::cout << "Using existing preprocessed files for: " << text_file << std::endl;
                return getPreprocessedInfo( vocab_file, tokens_file );
            }

            std::cout << "Preprocessing: " << text_file << std::endl;

            // Load raw text
            std::string text = loadText( text_file );
            std::cout << "  Loaded " << text.size() << " characters" << std::endl;

            // Build vocabulary
            CharVocabulary vocab;
            size_t vocab_size = vocab.buildFromText( text, add_special_tokens );
            std::cout << "  Built vocabulary: " << vocab_size << " tokens" << std::endl;

            // Save vocabulary
            vocab.save( vocab_file );
            std::cout << "  Saved vocabulary to: " << vocab_file << std::endl;

            // Tokenize and save
            size_t num_tokens = tokenizeAndSave( text, vocab, tokens_file );
            std::cout << "  Tokenized and saved " << num_tokens << " tokens to: " 
                << tokens_file << std::endl;

            return { vocab_size, num_tokens };
        }

        /**
         * @brief Checks if preprocessing output files are up-to-date.
         *
         * Returns true if both vocab and tokens files exist and are newer
         * than the source text file.
         */
        static bool isPreprocessed(
            const std::string& text_file,
            const std::string& vocab_file,
            const std::string& tokens_file )
        {
            if (!fs::exists( vocab_file ) || !fs::exists( tokens_file ))
            {
                return false;
            }

            // Check if preprocessed files are newer than source
            auto text_time = fs::last_write_time( text_file );
            auto vocab_time = fs::last_write_time( vocab_file );
            auto tokens_time = fs::last_write_time( tokens_file );

            return vocab_time >= text_time && tokens_time >= text_time;
        }

        /**
         * @brief Gets information about preprocessed files without loading full data.
         */
        static std::pair<size_t, size_t> getPreprocessedInfo(
            const std::string& vocab_file,
            const std::string& tokens_file )
        {
            // Read vocab size
            std::ifstream vocab_stream( vocab_file, std::ios::binary );
            if (!vocab_stream)
            {
                throw std::runtime_error( "Cannot open vocab file: " + vocab_file );
            }

            size_t vocab_size;
            vocab_stream.read( reinterpret_cast<char*>(&vocab_size), sizeof( vocab_size ) );

            // Read token count
            std::ifstream tokens_stream( tokens_file, std::ios::binary );
            if (!tokens_stream)
            {
                throw std::runtime_error( "Cannot open tokens file: " + tokens_file );
            }

            size_t num_tokens;
            tokens_stream.read( reinterpret_cast<char*>(&num_tokens), sizeof( num_tokens ) );

            return { vocab_size, num_tokens };
        }

    private:
        /**
         * @brief Loads entire text file into memory.
         */
        static std::string loadText( const std::string& filename )
        {
            std::ifstream file( filename, std::ios::binary );
            if (!file)
            {
                throw std::runtime_error( "Cannot open text file: " + filename );
            }

            file.seekg( 0, std::ios::end );
            size_t file_size = file.tellg();
            file.seekg( 0, std::ios::beg );

            std::string text( file_size, '\0' );
            file.read( &text[0], file_size );

            if (!file && !file.eof())
            {
                throw std::runtime_error( "Error reading text file: " + filename );
            }

            return text;
        }

        /**
         * @brief Tokenizes text and saves to binary file.
         *
         * Format:
         *   Bytes 0-7: num_tokens (size_t)
         *   Bytes 8+: token_indices (int32_t array)
         *
         * @return Number of tokens written
         */
        static size_t tokenizeAndSave(
            const std::string& text,
            const CharVocabulary& vocab,
            const std::string& output_file )
        {
            std::ofstream file( output_file, std::ios::binary );
            if (!file)
            {
                throw std::runtime_error( "Cannot open tokens file for writing: " + output_file );
            }

            size_t num_tokens = text.size();

            // Write header
            file.write( reinterpret_cast<const char*>(&num_tokens), sizeof( num_tokens ) );

            // Tokenize and write in chunks for memory efficiency
            constexpr size_t CHUNK_SIZE = 1024 * 1024;  // 1MB chunks
            std::vector<int32_t> token_buffer;
            token_buffer.reserve( std::min( num_tokens, CHUNK_SIZE ) );

            for (size_t i = 0; i < text.size(); ++i)
            {
                token_buffer.push_back( vocab.charToIndex( text[i] ) );

                if (token_buffer.size() >= CHUNK_SIZE || i == text.size() - 1)
                {
                    file.write(
                        reinterpret_cast<const char*>( token_buffer.data() ),
                        token_buffer.size() * sizeof( int32_t ) );

                    token_buffer.clear();
                }
            }

            if (!file)
            {
                throw std::runtime_error( "Error writing tokens file: " + output_file );
            }

            return num_tokens;
        }
    };
}
