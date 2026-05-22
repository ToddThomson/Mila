module;
#include <string>

export module Tokenize.Args;

namespace Tokenize
{
    export enum class Command
    {
        Unknown,
        Train,
        Encode,
        Decode,
        Help
    };

    export enum class TokenizerType
    {
        Unknown,
        Char,
        Bpe
    };

    export struct Args
    {
        Command command = Command::Unknown;
        std::string input_file;
        std::string output_file;
        std::string vocab_file;
        TokenizerType tokenizer_type = TokenizerType::Char;
        size_t vocab_size = 30000;
        size_t min_frequency = 2;
        bool byte_level = false;
        bool case_sensitive = true;
        bool force = false;
    };
}