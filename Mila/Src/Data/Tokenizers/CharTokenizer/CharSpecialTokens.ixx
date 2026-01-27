module;
#include <string>

export module Data.CharSpecialTokens;

namespace Mila::Data
{
    /**
     * @brief Configuration for special tokens used across tokenizer types.
     *
     * Special tokens are reserved vocabulary entries with specific semantic meaning
     * used during training, inference, or data processing.
     */
    export struct CharSpecialTokens {
        bool use_pad = true;
        bool use_unk = true;
        bool use_bos = true;
        bool use_eos = true;
        bool use_mask = false;  // Opt-in for MLM tasks

        size_t count() const {
            return (use_pad ? 1 : 0) + (use_unk ? 1 : 0) +
                (use_bos ? 1 : 0) + (use_eos ? 1 : 0) +
                (use_mask ? 1 : 0);
        }
    };
}