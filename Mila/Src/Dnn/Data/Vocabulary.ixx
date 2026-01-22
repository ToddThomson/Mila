module;
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <filesystem>

export module Data.Vocabulary;

namespace Mila::Dnn::Data
{
    // Universal vocabulary format (used by all tokenizer types)
    export struct Vocabulary {
        // Token <-> ID mappings
        std::unordered_map<std::string, uint32_t> token_to_id;
        std::vector<std::string> id_to_token;

        // BPE-specific data (empty for char-level)
        std::vector<std::pair<std::string, std::string>> merges;

        // Special tokens
        std::optional<uint32_t> bos_token_id;
        std::optional<uint32_t> eos_token_id;
        std::optional<uint32_t> pad_token_id;
        std::optional<uint32_t> unk_token_id;

        // Metadata
        std::string tokenizer_type;
        std::unordered_map<std::string, std::string> metadata;
    };

    // Serialization (binary format)
    void save_vocab( const Vocabulary& vocab, const std::filesystem::path& path );
    Vocabulary load_vocab( const std::filesystem::path& path );
}