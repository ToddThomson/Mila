module;
#include <string>
#include <span>

export module Data.BpeTokenizerTrainer;
export import :Config;

import Data.Vocabulary;

namespace Mila::Data
{
    class BPETokenizerTrainer {
    public:

        explicit BPETokenizerTrainer( Config config ) : config_( config ) {}

        // Train on corpus, return vocabulary
        Vocabulary train( std::span<const std::string> texts );

    private:
        BpeConfig config_;
        // BPE training implementation
    };
}