/*!
 * @file BpeTokenizer.Llama32.cpp
 * @brief Integration tests for BpeTokenizer loading the Llama 3.2 binary tokenizer format.
 */

#include <gtest/gtest.h>
#include <filesystem>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <utility>

import Mila;

namespace Mila::Data::Tests
{
    using Mila::Data::BpeTokenizer;
    using Mila::Data::TokenId;

    namespace fs = std::filesystem;

    static fs::path llama32_tokenizer_path()
    {
        fs::path dataDir = TEST_DATA_DIR;
        return dataDir / "models" / "llama" / "llama32_tokenizer.bin";
    }

    static std::string escape_for_debug( const std::string& s )
    {
        std::ostringstream oss;

        for ( unsigned char c : s )
        {
            if ( c == ' ' )
            {
                oss << "<sp>";
            }
            else if ( std::isprint( c ) )
            {
                oss << c;
            }
            else
            {
                oss << "\\x" << std::hex << std::setw( 2 ) << std::setfill( '0' )
                    << static_cast<int>(c) << std::dec << std::setfill( ' ' );
            }
        }

        return oss.str();
    }

    // ---- Load / skip --------------------------------------------------------

    TEST( BpeTokenizerLlama32, SkipWhenTokenizerMissing )
    {
        auto p = llama32_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Llama 3.2 tokenizer binary not present at: " << p.string();
        }

        ASSERT_TRUE( fs::is_regular_file( p ) );
        ASSERT_GT( fs::file_size( p ), 0u );
    }

    TEST( BpeTokenizerLlama32, LoadLlama32_SucceedsAndHasCorrectVocabSize )
    {
        auto p = llama32_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Llama 3.2 tokenizer binary not present at: " << p.string();
        }

        ASSERT_NO_THROW(
            {
                std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadLlama32( p );

        // Llama 3.2 uses a 128,256-token TikToken BPE vocabulary.
        size_t vsz = tokenizer->getVocabSize();
        EXPECT_EQ( vsz, 128256u );

        EXPECT_TRUE( tokenizer->isValidToken( static_cast<TokenId>(0u) ) );
        EXPECT_TRUE( tokenizer->isValidToken( static_cast<TokenId>(vsz - 1) ) );
        EXPECT_FALSE( tokenizer->isValidToken( static_cast<TokenId>(vsz) ) );
            } );
    }

    TEST( BpeTokenizerLlama32, LoadLlama32_InvalidPath_Throws )
    {
        fs::path bad = fs::path( "nonexistent" ) / "file" / "llama32_tokenizer.bin";
        EXPECT_THROW( BpeTokenizer::loadLlama32( bad ), std::runtime_error );
    }

    // ---- Vocabulary metadata ------------------------------------------------

    TEST( BpeTokenizerLlama32, GetBosTokenId_ReturnsExpectedId )
    {
        auto p = llama32_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Llama 3.2 tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadLlama32( p );

        auto bos = tokenizer->getBosTokenId();
        ASSERT_TRUE( bos.has_value() );
        EXPECT_EQ( *bos, static_cast<TokenId>(128000u) );
    }

    TEST( BpeTokenizerLlama32, GetEosTokenId_ReturnsExpectedId )
    {
        auto p = llama32_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Llama 3.2 tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadLlama32( p );

        auto eos = tokenizer->getEosTokenId();
        ASSERT_TRUE( eos.has_value() );
        EXPECT_EQ( *eos, static_cast<TokenId>(128001u) );
    }

    TEST( BpeTokenizerLlama32, GetPadTokenId_ReturnsNullopt )
    {
        auto p = llama32_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Llama 3.2 tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadLlama32( p );

        // Llama 3.2 has no dedicated PAD token; uses byte-level fallback instead.
        EXPECT_FALSE( tokenizer->getPadTokenId().has_value() );
    }

    // ---- Encode / Decode ----------------------------------------------------

    TEST( BpeTokenizerLlama32, EncodeDecode_Roundtrip_ShortText )
    {
        auto p = llama32_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Llama 3.2 tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadLlama32( p );

        std::string text = "Hello, world!";
        auto enc = tokenizer->encode( text );

        ASSERT_FALSE( enc.empty() );

        for ( auto id : enc )
        {
            EXPECT_TRUE( tokenizer->isValidToken( id ) );
        }

        EXPECT_EQ( tokenizer->decode( enc ), text );
    }

    TEST( BpeTokenizerLlama32, Encode_BosSpecialToken_ReturnsDirectId )
    {
        auto p = llama32_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Llama 3.2 tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadLlama32( p );

        auto ids = tokenizer->encode( "<|begin_of_text|>" );
        ASSERT_EQ( ids.size(), 1u );
        EXPECT_EQ( ids[ 0 ], static_cast<TokenId>(128000u) );
    }

    TEST( BpeTokenizerLlama32, Encode_EosSpecialToken_ReturnsDirectId )
    {
        auto p = llama32_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Llama 3.2 tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadLlama32( p );

        auto ids = tokenizer->encode( "<|end_of_text|>" );
        ASSERT_EQ( ids.size(), 1u );
        EXPECT_EQ( ids[ 0 ], static_cast<TokenId>(128001u) );
    }

    TEST( BpeTokenizerLlama32, EncodeDecode_Roundtrip_WithBosAndEos )
    {
        auto p = llama32_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Llama 3.2 tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadLlama32( p );

        const std::string text = "<|begin_of_text|>Hello, world!<|end_of_text|>";
        auto enc = tokenizer->encode( text );

        ASSERT_GE( enc.size(), 3u );
        EXPECT_EQ( enc.front(), static_cast<TokenId>(128000u) );
        EXPECT_EQ( enc.back(), static_cast<TokenId>(128001u) );
        EXPECT_EQ( tokenizer->decode( enc ), text );
    }

    TEST( BpeTokenizerLlama32, Encode_PlainText_MatchesHuggingFaceOutput )
    {
        auto p = llama32_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Llama 3.2 tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadLlama32( p );

        // Ground truth from convert_llama_tokenizer.py test output (add_special_tokens=False).
        // Verifies that Mila's BPE implementation matches the HuggingFace TikToken reference.
        const std::string text = "Hello, world! This is a test.";
        const std::vector<TokenId> expected = { 9906, 11, 1917, 0, 1115, 374, 264, 1296, 13 };

        auto ids = tokenizer->encode( text );

        ASSERT_EQ( ids.size(), expected.size() );

        for ( size_t i = 0; i < ids.size(); ++i )
        {
            EXPECT_EQ( ids[ i ], expected[ i ] ) << "Token mismatch at position " << i;
        }
    }

    TEST( BpeTokenizerLlama32, Encode_LongNumber_ChunksToMaxThreeDigits )
    {
        auto p = llama32_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Llama 3.2 tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadLlama32( p );

        // The Llama 3 pre-tokenization regex splits digit runs into chunks of at most
        // 3 digits before BPE runs. "12345678" (8 digits) -> ["123","456","78"] = 3 chunks,
        // so the token count must be >= 3 regardless of BPE merges within each chunk.
        auto ids = tokenizer->encode( "12345678" );

        ASSERT_GE( ids.size(), 3u );

        for ( auto id : ids )
        {
            EXPECT_TRUE( tokenizer->isValidToken( id ) );
        }
    }

    TEST( BpeTokenizerLlama32, Decode_KnownTokenIds_MatchesHuggingFaceText )
    {
        auto p = llama32_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Llama 3.2 tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadLlama32( p );

        // Ground truth from convert_llama_tokenizer.py (add_special_tokens=False).
        const std::vector<TokenId> ids = { 9906, 11, 1917, 0, 1115, 374, 264, 1296, 13 };
        const std::string expected = "Hello, world! This is a test.";

        EXPECT_EQ( tokenizer->decode( ids ), expected );
    }

    // HuggingFace's ids for meta-llama/Llama-3.1-8B-Instruct (add_special_tokens=False, transformers 5.12.1).
    // Each word after the first is one that the longest match at each position splits differently: " seminary"
    // is 54675 88 that way, " cradle" 46141 91485, " turrets" 65486 82.
    TEST( BpeTokenizerLlama32, Encode_MatchesHuggingFaceIds )
    {
        auto p = llama32_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Llama 3.2 tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadLlama32( p );

        const std::pair<std::string, std::vector<TokenId>> cases[] = {
            { "Hello, world! This is a test.", { 9906, 11, 1917, 0, 1115, 374, 264, 1296, 13 } },
            { " seminary", { 5347, 3367 } },
            { " CARMEN ARIZA", { 356, 18394, 965, 6395, 2913, 32 } },
            { " Tatch", { 350, 759 } },
            { " turrets", { 259, 88434 } },
            { " cradle", { 1589, 72455 } },
            { " The seminary in Seville kept its turrets.", { 578, 5347, 3367, 304, 1369, 8078, 8774, 1202, 259, 88434, 13 } },
        };

        for ( const auto& [ text, expected ] : cases )
        {
            EXPECT_EQ( tokenizer->encode( text ), expected ) << "\"" << text << "\"";
        }
    }

    // A contraction matches whatever its case, so after a line break a quote before a capital splits as "'D": the
    // pattern's (?i:...), which the ASCII fallback has to spell out. HuggingFace's ids, as above.
    TEST( BpeTokenizerLlama32, Encode_QuoteBeforeCapital_MatchesHuggingFaceIds )
    {
        auto p = llama32_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Llama 3.2 tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadLlama32( p );

        const std::pair<std::string, std::vector<TokenId>> cases[] = {
            { "Crass.\n\n'Didn't he", { 16384, 395, 382, 28805, 307, 77, 956, 568 } },
            { "that.'\n\n'They calls 'em", { 9210, 22438, 17773, 36661, 6880, 364, 336 } },
            { "\n'There's", { 198, 17773, 6881, 596 } },
            { " IT'S WE'LL THEY'RE", { 8871, 13575, 20255, 6, 4178, 63593, 95253 } },
        };

        for ( const auto& [ text, expected ] : cases )
        {
            EXPECT_EQ( tokenizer->encode( text ), expected ) << "\"" << text << "\"";
        }
    }

    // What an agentic prompt carries and a single-spaced sentence does not: indentation of 4, 8 and 12 spaces, two
    // spaces inside a line, tabs, trailing spaces, runs of blank lines, a tool call as escaped JSON and its result.
    // ASCII only: non-ASCII text is split by an ASCII approximation of the pattern and does not yet match (Vnext.md,
    // "Llama, Qwen and GPT-2 tokenize non-ASCII text differently from HuggingFace").
    const std::string kAgenticText =
        "<|start_header_id|>user<|end_header_id|>\n"
        "\n"
        "Patch it<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        "\n"
        "<|python_tag|>{\"name\": \"patch\", \"parameters\": {\"path\": \"stockroom/inventory.py\", \"new_string\": "
        "\"class Inventory:\\n    def __init__(self):\\n        self._quantity = {}\\n\\n    def add(self, sku, "
        "quantity):\\n        if quantity <= 0:\\n            raise ValueError(\\\"quantity  must be positive\\\")\\n"
        "\\t\\treturn  x   \\n\"}}<|eom_id|><|start_header_id|>ipython<|end_header_id|>\n"
        "\n"
        "class Inventory:\n"
        "    def __init__(self):\n"
        "        self._quantity = {}\n"
        "\n"
        "    def add(self, sku, quantity):\n"
        "        if quantity <= 0:\n"
        "            raise ValueError(\"quantity  must be positive\")\n"
        "\t\treturn  x   \n"
        "{\"ok\":  true,\n"
        "\n"
        "\n"
        "  \"n\": 3}  \n"
        "<|eot_id|>";

    TEST( BpeTokenizerLlama32, Encode_AgenticText_MatchesHuggingFaceIds )
    {
        auto p = llama32_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Llama 3.2 tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadLlama32( p );

        // HuggingFace `tokenizers` on meta-llama/Llama-3.2-3B-Instruct's tokenizer.json,
        // encode( kAgenticText, add_special_tokens=False ).
        const std::vector<TokenId> expected = {
            128006, 882, 128007, 271, 44722, 433, 128009, 128006, 78191, 128007, 271, 128010, 5018, 609, 794, 330,
            3482, 498, 330, 14105, 794, 5324, 2398, 794, 330, 13787, 3039, 18480, 7371, 7345, 498, 330,
            943, 3991, 794, 330, 1058, 29226, 7338, 77, 262, 711, 1328, 2381, 3889, 726, 90149, 77,
            286, 659, 1462, 13832, 284, 57157, 77, 1734, 262, 711, 923, 1214, 11, 53422, 11, 12472,
            90149, 77, 286, 422, 12472, 2717, 220, 15, 7338, 77, 310, 4933, 15764, 37114, 13832, 220,
            2011, 387, 6928, 63805, 59, 77, 5061, 5061, 693, 220, 865, 256, 1144, 77, 32075, 128008,
            128006, 23799, 4690, 128007, 271, 1058, 29226, 512, 262, 711, 1328, 2381, 3889, 726, 997, 286,
            659, 1462, 13832, 284, 10294, 262, 711, 923, 1214, 11, 53422, 11, 12472, 997, 286, 422,
            12472, 2717, 220, 15, 512, 310, 4933, 15764, 446, 13832, 220, 2011, 387, 6928, 1158, 197,
            862, 220, 865, 5996, 5018, 564, 794, 220, 837, 58575, 220, 330, 77, 794, 220, 18,
            92, 2355, 128009 };

        const auto ids = tokenizer->encode( kAgenticText );

        ASSERT_EQ( ids.size(), expected.size() );

        for ( size_t i = 0; i < ids.size(); ++i )
        {
            EXPECT_EQ( ids[ i ], expected[ i ] ) << "Token mismatch at position " << i;
        }

        EXPECT_EQ( tokenizer->decode( ids ), kAgenticText );
    }

    TEST( BpeTokenizerLlama32, Encode_AddedTokens_AreSingleAtomicTokens )
    {
        auto p = llama32_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Llama 3.2 tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadLlama32( p );

        // Every token Llama 3.2's tokenizer.json lists in added_tokens, with its id there. Stated literally rather than
        // read from the loader so the test is an independent expectation; reserved tokens 3 to 247 are 128008 + N.
        std::vector<std::pair<std::string, TokenId>> added_tokens = {
            { "<|begin_of_text|>", 128000 }, { "<|end_of_text|>", 128001 },
            { "<|reserved_special_token_0|>", 128002 }, { "<|reserved_special_token_1|>", 128003 },
            { "<|finetune_right_pad_id|>", 128004 }, { "<|reserved_special_token_2|>", 128005 },
            { "<|start_header_id|>", 128006 }, { "<|end_header_id|>", 128007 }, { "<|eom_id|>", 128008 },
            { "<|eot_id|>", 128009 }, { "<|python_tag|>", 128010 } };

        for ( int n = 3; n <= 247; ++n )
        {
            added_tokens.emplace_back(
                "<|reserved_special_token_" + std::to_string( n ) + "|>", static_cast<TokenId>(128008 + n) );
        }

        ASSERT_EQ( added_tokens.size(), 256u );

        for ( const auto& [token, id] : added_tokens )
        {
            auto ids = tokenizer->encode( token );
            ASSERT_EQ( ids.size(), 1u ) << "Added token not matched atomically: " << token;
            EXPECT_EQ( ids[ 0 ], id ) << token;
        }
    }

    TEST( BpeTokenizerLlama32, EncodeDecode_Roundtrip_ShortText_Diagnostic )
    {
        auto p = llama32_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Llama 3.2 tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadLlama32( p );

        std::string text = "Hello, world!";
        auto enc = tokenizer->encode( text );

        ASSERT_FALSE( enc.empty() );

        for ( auto id : enc )
        {
            EXPECT_TRUE( tokenizer->isValidToken( id ) );
        }

        std::string decoded = tokenizer->decode( enc );

        if ( decoded != text )
        {
            std::ostringstream msg;
            msg << "Roundtrip mismatch\n";
            msg << "  original      : \"" << text << "\"\n";
            msg << "  original (esc): \"" << escape_for_debug( text ) << "\"\n";
            msg << "  decoded       : \"" << decoded << "\"\n";
            msg << "  decoded (esc) : \"" << escape_for_debug( decoded ) << "\"\n";
            msg << "  token ids (" << enc.size() << "):";

            for ( size_t i = 0; i < enc.size(); ++i )
            {
                msg << (i ? "," : "") << enc[ i ];
            }

            msg << "\n  tokens:\n";

            for ( size_t i = 0; i < enc.size(); ++i )
            {
                auto id = enc[ i ];
                auto tokstr = tokenizer->tokenToString( id );
                msg << "    [" << i << "] id=" << id
                    << " raw=\"" << tokstr
                    << "\" esc=\"" << escape_for_debug( tokstr ) << "\"\n";
            }

            msg << "  original bytes: ";

            for ( unsigned char c : text )
            {
                msg << std::hex << std::setw( 2 ) << std::setfill( '0' )
                    << static_cast<int>(c) << " ";
            }

            msg << std::dec << "\n";

            FAIL() << msg.str();
        }

        EXPECT_EQ( decoded, text );
    }
}