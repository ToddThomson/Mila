/**
 * @file Gpt2Tokenizer.cpp
 * @brief Unit tests for the Gpt2Tokenizer loader and runtime.
 *
 * Exercises binary loading, encode/decode, special tokens, and token lookup.
 */

#include <gtest/gtest.h>

#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <cstdint>
#include <memory>

import Mila;

using namespace Mila::Dnn::Data;

namespace Tests::Dnn::Data
{

static void write_u32(std::ofstream& os, uint32_t v)
{
    os.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

static void write_bytes(std::ofstream& os, std::string_view s)
{
    os.write(s.data(), static_cast<std::streamsize>(s.size()));
}

/**
 * Ensure loading a non-existent tokenizer file returns nullptr.
 */
TEST(Gpt2TokenizerTest, MissingFile)
{
    auto tokenizer = Gpt2Tokenizer::fromFile("this_file_does_not_exist_tokenizer.bin");
    ASSERT_EQ(tokenizer, nullptr);
}

/**
 * Create a minimal binary tokenizer file, load it, and validate:
 *  - vocab size
 *  - special tokens (BOS/EOS/PAD)
 *  - encode / decode behavior
 *  - tokenToString / isValidToken behavior
 */
TEST(Gpt2TokenizerTest, LoadEncodeDecodeAndSpecialTokens)
{
    const std::filesystem::path tmp =
        std::filesystem::temp_directory_path() / "mila_test_gpt2_tokenizer.bin";

    {
        std::ofstream os(tmp, std::ios::binary);
        ASSERT_TRUE(os.good()) << "Failed to open temporary tokenizer file for writing";

        // Minimal vocabulary: three tokens
        // ids: 0 -> "<bos>", 1 -> "<eos>", 2 -> "A"
        const std::vector<std::pair<std::string, uint32_t>> vocab = {
            { "<bos>", 0u },
            { "<eos>", 1u },
            { "A",     2u }
        };

        const uint32_t vocabSize = static_cast<uint32_t>(vocab.size());
        const uint32_t numMerges = 0u; // no merges for simplicity

        // Header
        write_u32(os, vocabSize);
        write_u32(os, numMerges);

        // Vocabulary entries: token_len, token_bytes, token_id
        for (const auto& [tok, id] : vocab) {
            write_u32(os, static_cast<uint32_t>(tok.size()));
            write_bytes(os, tok);
            write_u32(os, id);
        }

        // No BPE merges to write (numMerges == 0)

        // Special tokens: has_eos, eos_id, has_bos, bos_id, has_pad, pad_id
        write_u32(os, 1u); // has_eos
        write_u32(os, 1u); // eos id

        write_u32(os, 1u); // has_bos
        write_u32(os, 0u); // bos id

        write_u32(os, 0u); // has_pad (absent)
        // (no pad id follows)

        os.flush();
        os.close();
    }

    // Load tokenizer
    auto tokenizer = Gpt2Tokenizer::fromFile(tmp.string());
    ASSERT_NE(tokenizer, nullptr);

    // Basic properties
    EXPECT_EQ(tokenizer->getVocabSize(), 3u);

    // Special token ids
    ASSERT_TRUE(tokenizer->getBosTokenId().has_value());
    ASSERT_TRUE(tokenizer->getEosTokenId().has_value());
    EXPECT_FALSE(tokenizer->getPadTokenId().has_value());

    EXPECT_EQ(*tokenizer->getBosTokenId(), 0u);
    EXPECT_EQ(*tokenizer->getEosTokenId(), 1u);

    // tokenToString and isValidToken
    EXPECT_EQ(tokenizer->tokenToString(0u), "<bos>");
    EXPECT_EQ(tokenizer->tokenToString(1u), "<eos>");
    EXPECT_EQ(tokenizer->tokenToString(2u), "A");

    EXPECT_TRUE(tokenizer->isValidToken(0u));
    EXPECT_TRUE(tokenizer->isValidToken(1u));
    EXPECT_TRUE(tokenizer->isValidToken(2u));
    EXPECT_FALSE(tokenizer->isValidToken(9999u));

    // Encode single character "A" -> should map to id 2
    auto encoded = tokenizer->encode("A");
    ASSERT_EQ(encoded.size(), 1u);
    EXPECT_EQ(encoded[0], 2u);

    // tokenToString for unknown token id should return "<UNK>"
    EXPECT_EQ(tokenizer->tokenToString(0xFFFFFFFFu), std::string("<UNK>"));

    // Cleanup
    std::error_code ec;
    std::filesystem::remove(tmp, ec);
    if (ec) {
        // do not fail test on cleanup problems, but log
        ADD_FAILURE() << "Failed to remove temp tokenizer file: " << ec.message();
    }
}

} // namespace Tests::Dnn::Data