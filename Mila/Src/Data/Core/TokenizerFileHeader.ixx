module;
#include <cstdint>

export module Data.TokenizerFileHeader;

namespace Mila::Data
{
    constexpr uint32_t TOKENIZED_FILE_MAGIC_NUMBER = 0x4D494C41; // "MILA"
    constexpr uint32_t TOKENIZED_FILE_VERSION = 1;

    export struct TokenizedFileHeader {
        uint32_t magic_number;      // e.g., 0x4D494C41 ("MILA")
        uint32_t version;           // format version
        uint16_t tokenizer_type;    // 0=char, 1=bpe, etc.
        uint64_t num_tokens;
        uint32_t vocab_size;        // useful for validation
        // padding to align to cache line if needed
    };
}