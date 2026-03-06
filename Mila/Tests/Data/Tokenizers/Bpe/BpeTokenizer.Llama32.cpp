/*!
 * @file BpeTokenizer.Llama32.cpp
 * @brief Integration tests for BpeTokenizer loading the Llama 3.2 binary tokenizer format.
 */

#include <gtest/gtest.h>

import Mila;

#include <filesystem>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

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
        // Tests the decode direction only; encode does not yet match HF because
        // Llama 3.2 TikToken requires max-munch (longest-prefix) matching rather
        // than explicit BPE merge rules.
        const std::vector<TokenId> ids = { 9906, 11, 1917, 0, 1115, 374, 264, 1296, 13 };
        const std::string expected = "Hello, world! This is a test.";

        EXPECT_EQ( tokenizer->decode( ids ), expected );
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