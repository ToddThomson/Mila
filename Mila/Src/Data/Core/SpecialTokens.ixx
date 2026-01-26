module;
#include <string>

export module Data.SpecialTokens;

namespace Mila::Data
{
    /**
     * @brief Configuration for special tokens used across tokenizer types.
     *
     * Special tokens are reserved vocabulary entries with specific semantic meaning
     * used during training, inference, or data processing.
     */
    export struct SpecialTokens {
        bool enabled = true;
        std::string pad_token = "<|pad|>";   ///< Padding token for sequence alignment
        std::string unk_token = "<|unk|>";   ///< Unknown token for OOV words
        std::string bos_token = "<|bos|>";   ///< Beginning of sequence marker
        std::string eos_token = "<|eos|>";   ///< End of sequence marker
    };
}