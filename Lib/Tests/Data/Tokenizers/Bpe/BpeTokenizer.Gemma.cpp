/*!
 * @file BpeTokenizer.Gemma.cpp
 * @brief Integration tests for BpeTokenizer loading the Gemma 4 SentencePiece BPE binary.
 *
 * Gated on the converted tokenizer binary being present (skips otherwise), like the
 * Llama 3.2 suite. Produce it with:
 *   python Tools/Converters/Gemma/convert_tokenizer.py --model google/gemma-4-12b-it \
 *       --output <TEST_DATA_DIR>/models/gemma/gemma_tokenizer.bin
 *
 * The ground-truth ids below are the HuggingFace reference printed by that converter
 * for the same string (add_special_tokens=False), so a green Encode_PlainText test is
 * byte-for-byte SentencePiece BPE parity with HF on this input.
 */

#include <gtest/gtest.h>
#include <filesystem>
#include <vector>
#include <string>

import Mila;

namespace Mila::Data::Tests
{
    using Mila::Data::BpeTokenizer;
    using Mila::Data::TokenId;

    namespace fs = std::filesystem;

    static fs::path gemma_tokenizer_path()
    {
        fs::path dataDir = TEST_DATA_DIR;
        return dataDir / "models" / "gemma" / "gemma_tokenizer.bin";
    }

    // ---- Load / skip --------------------------------------------------------

    TEST( BpeTokenizerGemma, Load_SucceedsAndHasCorrectVocabSize )
    {
        auto p = gemma_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Gemma tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadGemma( p );

        // Gemma 4 uses a 262,144-token SentencePiece BPE vocabulary.
        size_t vsz = tokenizer->getVocabSize();
        EXPECT_EQ( vsz, 262144u );

        EXPECT_TRUE( tokenizer->isValidToken( static_cast<TokenId>(0u) ) );
        EXPECT_TRUE( tokenizer->isValidToken( static_cast<TokenId>(vsz - 1) ) );
        EXPECT_FALSE( tokenizer->isValidToken( static_cast<TokenId>(vsz) ) );
    }

    TEST( BpeTokenizerGemma, LoadGemma_InvalidPath_Throws )
    {
        fs::path bad = fs::path( "nonexistent" ) / "file" / "gemma_tokenizer.bin";
        EXPECT_THROW( BpeTokenizer::loadGemma( bad ), std::runtime_error );
    }

    // ---- Special token metadata (Gemma: <pad>=0, <eos>=1, <bos>=2, <unk>=3) --

    TEST( BpeTokenizerGemma, SpecialTokenIds_MatchGemmaConvention )
    {
        auto p = gemma_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Gemma tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadGemma( p );

        auto bos = tokenizer->getBosTokenId();
        auto eos = tokenizer->getEosTokenId();
        auto pad = tokenizer->getPadTokenId();

        ASSERT_TRUE( bos.has_value() );
        ASSERT_TRUE( eos.has_value() );
        ASSERT_TRUE( pad.has_value() );
        EXPECT_EQ( *bos, static_cast<TokenId>(2u) );
        EXPECT_EQ( *eos, static_cast<TokenId>(1u) );
        EXPECT_EQ( *pad, static_cast<TokenId>(0u) );
    }

    // ---- Encode parity with HuggingFace -------------------------------------

    TEST( BpeTokenizerGemma, Encode_PlainText_MatchesHuggingFaceOutput )
    {
        auto p = gemma_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Gemma tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadGemma( p );

        // Ground truth from convert_tokenizer.py (add_special_tokens=False):
        //   ['The', '_capital', '_of', '_France', '_is', '_Paris', '.']  (_ = U+2581 metaspace)
        const std::string text = "The capital of France is Paris.";
        const std::vector<TokenId> expected = { 818, 5279, 529, 7001, 563, 9079, 236761 };

        auto ids = tokenizer->encode( text );

        ASSERT_EQ( ids.size(), expected.size() );

        for ( size_t i = 0; i < ids.size(); ++i )
        {
            EXPECT_EQ( ids[ i ], expected[ i ] ) << "Token mismatch at position " << i;
        }
    }

    // What an agentic prompt carries and a single-spaced sentence does not: indentation of 4, 8 and 12 spaces, two
    // spaces inside a line, tabs, trailing spaces, runs of blank lines, tool-call strings delimited by <|"|> (one
    // empty), and the special tokens past the turn and tool markers, then non-ASCII text.
    const std::string kAgenticText =
        "<|turn>user\n"
        "Patch it<turn|>\n"
        "<|turn>model\n"
        "<|tool_call>call:patch{new_string:<|\"|>class Inventory:\n"
        "    def __init__(self):\n"
        "        self._quantity = {}\n"
        "\n"
        "    def add(self, sku, quantity):\n"
        "        if quantity <= 0:\n"
        "            raise ValueError(\"quantity  must be positive\")\n"
        "\t\treturn  x   \n"
        "<|\"|>,old_string:<|\"|><|\"|>,path:<|\"|>stockroom/inventory.py<|\"|>}<tool_call|><|tool_response>\n"
        "{\"ok\":  true,\n"
        "\n"
        "\n"
        "  \"n\": 3}  <mask> <|image> caf\xC3\xA9 \xF0\x9F\x99\x82\n";

    TEST( BpeTokenizerGemma, Encode_AgenticText_MatchesHuggingFaceOutput )
    {
        auto p = gemma_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Gemma tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadGemma( p );

        // HuggingFace `tokenizers` on google/gemma-4-26B-A4B-it-qat-q4_0-unquantized's tokenizer.json,
        // encode( kAgenticText, add_special_tokens=False ) -- the ids llama.cpp gives too. 140, 144 and 148 are 4, 8
        // and 12 spaces as one piece, 52 the string delimiter <|"|>.
        const std::vector<TokenId> expected = {
            105, 2364, 107, 73842, 625, 106, 107, 105, 4368, 107, 48, 6639, 236787, 10517, 236782, 1951,
            236779, 2383, 236787, 52, 1999, 52761, 236787, 107, 140, 2063, 5062, 2949, 10210, 1508, 1473, 107,
            144, 1508, 4917, 22111, 578, 8637, 108, 140, 2063, 1138, 236769, 1508, 236764, 101541, 236764, 10489,
            1473, 107, 144, 584, 10489, 6605, 236743, 236771, 236787, 107, 148, 24296, 38271, 885, 22111, 138,
            23600, 577, 4414, 1373, 107, 255969, 2060, 138, 236781, 139, 107, 52, 236764, 947, 236779, 2383,
            236787, 52, 52, 236764, 2337, 236787, 52, 14973, 3352, 236786, 46589, 236761, 2551, 52, 236783, 49,
            50, 107, 14937, 680, 1083, 138, 3397, 236764, 109, 138, 236775, 236749, 1083, 236743, 236800, 236783,
            138, 4, 236743, 255999, 33443, 57235, 107 };

        const auto ids = tokenizer->encode( kAgenticText );

        ASSERT_EQ( ids.size(), expected.size() );

        for ( size_t i = 0; i < ids.size(); ++i )
        {
            EXPECT_EQ( ids[ i ], expected[ i ] ) << "Token mismatch at position " << i;
        }

        EXPECT_EQ( tokenizer->decode( ids ), kAgenticText );
    }

    TEST( BpeTokenizerGemma, EncodeDecode_Roundtrip_RestoresText )
    {
        auto p = gemma_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Gemma tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadGemma( p );

        const std::string text = "The capital of France is Paris.";
        auto enc = tokenizer->encode( text );

        ASSERT_FALSE( enc.empty() );
        EXPECT_EQ( tokenizer->decode( enc ), text );
    }

    // ---- Special tokens are matched atomically ------------------------------

    TEST( BpeTokenizerGemma, Encode_BosSpecialToken_ReturnsDirectId )
    {
        auto p = gemma_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Gemma tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadGemma( p );

        auto ids = tokenizer->encode( "<bos>" );
        ASSERT_EQ( ids.size(), 1u );
        EXPECT_EQ( ids[ 0 ], static_cast<TokenId>(2u) );
    }

    TEST( BpeTokenizerGemma, Encode_SpecialTokens_AreSingleAtomicTokens )
    {
        auto p = gemma_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Gemma tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadGemma( p );

        // Every token Gemma 4's tokenizer.json marks special (its added_tokens), with its id there. Stated literally
        // rather than imported from the loader so the test is an independent expectation, not a restatement of the
        // code under test.
        const std::vector<std::pair<std::string, TokenId>> special_tokens = {
            { "<pad>", 0 }, { "<eos>", 1 }, { "<bos>", 2 }, { "<unk>", 3 }, { "<mask>", 4 },
            { "<|tool>", 46 }, { "<tool|>", 47 }, { "<|tool_call>", 48 }, { "<tool_call|>", 49 },
            { "<|tool_response>", 50 }, { "<tool_response|>", 51 }, { "<|\"|>", 52 },
            { "<|think|>", 98 }, { "<|channel>", 100 }, { "<channel|>", 101 }, { "<|turn>", 105 }, { "<turn|>", 106 },
            { "<|image>", 255999 }, { "<|audio>", 256000 }, { "<|image|>", 258880 }, { "<|audio|>", 258881 },
            { "<image|>", 258882 }, { "<audio|>", 258883 }, { "<|video|>", 258884 } };

        for ( const auto& [token, id] : special_tokens )
        {
            auto ids = tokenizer->encode( token );
            ASSERT_EQ( ids.size(), 1u ) << "Special token not matched atomically: " << token;
            EXPECT_EQ( ids[ 0 ], id ) << token;
        }
    }

    TEST( BpeTokenizerGemma, Encode_Gemma3TurnMarkers_AreNotInTheVocabulary )
    {
        auto p = gemma_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Gemma tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadGemma( p );

        // Pins the Gemma 3 -> Gemma 4 protocol change that the previous version of
        // this test asserted backwards: Gemma 4 replaced <start_of_turn>/<end_of_turn>
        // with <|turn>/<turn|>, so the Gemma 3 markers encode as subword pieces.
        EXPECT_GT( tokenizer->encode( "<start_of_turn>" ).size(), 1u );
        EXPECT_GT( tokenizer->encode( "<end_of_turn>" ).size(), 1u );
    }
}
