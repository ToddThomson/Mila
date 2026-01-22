module;
#include <cstddef>

export module Data.BpeTokenizerTrainer:Config;

namespace Mila::Data
{
    export struct BpeTokenizerTrainerConfig {
        std::size_t vocab_size = 10000;
        std::size_t min_frequency = 2;
        bool add_prefix_space = false;
    };
}