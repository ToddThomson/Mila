/**
 * @file PreprocessText.cpp
 * @brief Standalone utility for preprocessing text files for CharLM.
 *
 * Usage:
 *   PreprocessText <input_text_file> [--force]
 *
 * Creates:
 *   <input_text_file>.vocab  - Binary vocabulary file
 *   <input_text_file>.tokens - Binary tokenized data
 */

#include <iostream>
#include <string>
#include <stdexcept>

import CharLM.Preprocessor;

void printUsage()
{
    std::cout << "Usage: PreprocessText <input_text_file> [--force]" << std::endl;
    std::cout << std::endl;
    std::cout << "Preprocesses text file for character-level language modeling." << std::endl;
    std::cout << "Creates vocabulary and tokenized data files." << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --force    Force rebuild even if preprocessed files exist" << std::endl;
    std::cout << "  --special  Add special tokens (PAD, UNK) to vocabulary" << std::endl;
    std::cout << "  --help     Show this help message" << std::endl;
}

int main( int argc, char** argv )
{
    try
    {
        if (argc < 2)
        {
            printUsage();
            return 1;
        }

        std::string input_file;
        bool force_rebuild = false;
        bool add_special = false;

        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];

            if (arg == "--help" || arg == "-h")
            {
                printUsage();
                return 0;
            }
            else if (arg == "--force" || arg == "-f")
            {
                force_rebuild = true;
            }
            else if (arg == "--special" || arg == "-s")
            {
                add_special = true;
            }
            else if (input_file.empty())
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

        if (input_file.empty())
        {
            std::cerr << "Error: No input file specified" << std::endl;
            printUsage();
            return 1;
        }

        std::cout << "Preprocessing: " << input_file << std::endl;
        std::cout << "Force rebuild: " << (force_rebuild ? "yes" : "no") << std::endl;
        std::cout << "Special tokens: " << (add_special ? "yes" : "no") << std::endl;
        std::cout << std::endl;

        auto [vocab_size, num_tokens] = Mila::CharLM::CharPreprocessor::preprocess(
            input_file,
            force_rebuild,
            add_special );

        std::cout << std::endl;
        std::cout << "Preprocessing complete!" << std::endl;
        std::cout << "  Vocabulary size: " << vocab_size << std::endl;
        std::cout << "  Total tokens: " << num_tokens << std::endl;
        std::cout << "  Output files:" << std::endl;
        std::cout << "    " << input_file << ".vocab" << std::endl;
        std::cout << "    " << input_file << ".tokens" << std::endl;

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
