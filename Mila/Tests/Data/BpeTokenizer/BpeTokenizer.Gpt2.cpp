#include <gtest/gtest.h>

import Mila;

#include <filesystem>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

namespace Data::Tokenizers::BpeTokenizer_Gpt2_Tests
{
    using namespace Mila::Data;
    namespace fs = std::filesystem;

    // Helper: path to GPT-2 tokenizer under TEST_DATA_DIR (set via CMake target_compile_definitions)
    static fs::path gpt2_tokenizer_path()
    {
        fs::path dataDir = TEST_DATA_DIR; // defined by Mila\Tests/CMakeLists.txt
        return dataDir / "models" / "gpt2" / "gpt2_tokenizer.bin";
    }

    static std::string escape_for_debug( const std::string& s )
    {
        std::ostringstream oss;
        for ( unsigned char c : s ) {
            if ( c == ' ' ) {
                oss << "<sp>";
            }
            else if ( std::isprint( c ) ) {
                oss << c;
            }
            else {
                oss << "\\x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(c)
                    << std::dec << std::setfill(' ');
            }
        }
        return oss.str();
    }

    TEST( BpeTokenizerGpt2, SkipWhenTokenizerMissing )
    {
        auto p = gpt2_tokenizer_path();
        if ( !fs::exists( p ) ) {
            GTEST_SKIP() << "GPT-2 tokenizer binary not present at: " << p.string();
        }

        // If present, just ensure the file is readable
        ASSERT_TRUE( fs::is_regular_file( p ) );
        ASSERT_GT( fs::file_size( p ), 0u );
    }

    TEST( BpeTokenizerGpt2, LoadGpt2_SucceedsAndHasLargeVocab )
    {
        auto p = gpt2_tokenizer_path();
        if ( !fs::exists( p ) ) {
            GTEST_SKIP() << "GPT-2 tokenizer binary not present at: " << p.string();
        }

        // Act
        ASSERT_NO_THROW( {
            BpeTokenizer tokenizer = BpeTokenizer::loadGpt2( p );

        // Basic expectations for GPT-2 style vocabularies: very large (>50k)
        size_t vsz = tokenizer.getVocabSize();
        EXPECT_GT( vsz, 50000u );

        // A few token ids should be valid (beginning and end of range)
        EXPECT_TRUE( tokenizer.isValidToken( static_cast<Mila::Dnn::Data::TokenId>(0u) ) );
        EXPECT_TRUE( tokenizer.isValidToken( static_cast<Mila::Dnn::Data::TokenId>(vsz - 1) ) );
            } );
    }

    TEST( BpeTokenizerGpt2, EncodeDecode_SinglePrintableToken )
    {
        auto p = gpt2_tokenizer_path();
        if ( !fs::exists( p ) ) {
            GTEST_SKIP() << "GPT-2 tokenizer binary not present at: " << p.string();
        }

        BpeTokenizer tokenizer = BpeTokenizer::loadGpt2( p );

        std::string piece = "!";
        auto enc = tokenizer.encode( piece );

        // Expect at least one token produced; tokenToString of first id should match piece
        ASSERT_GE( enc.size(), 1u );
        auto firstId = enc.front();
        EXPECT_EQ( tokenizer.tokenToString( firstId ), piece );

        // Decode roundtrip should equal original for byte-level GPT-2 tokenizer
        std::string decoded = tokenizer.decode( enc );
        EXPECT_EQ( decoded, piece );
    }

    TEST( BpeTokenizerGpt2, EncodeDecode_Roundtrip_ShortText )
    {
        auto p = gpt2_tokenizer_path();
        if ( !fs::exists( p ) ) {
            GTEST_SKIP() << "GPT-2 tokenizer binary not present at: " << p.string();
        }

        BpeTokenizer tokenizer = BpeTokenizer::loadGpt2( p );

        std::string text = "Hello, world!";
        auto enc = tokenizer.encode( text );

        // Ensure encoding produced tokens and all token ids are valid
        ASSERT_FALSE( enc.empty() );
        for ( auto id : enc ) {
            EXPECT_TRUE( tokenizer.isValidToken( id ) );
        }

        // Decode should reconstruct the original text for byte-level BPE
        std::string decoded = tokenizer.decode( enc );
        EXPECT_EQ( decoded, text );
    }

    TEST( BpeTokenizerGpt2, LoadGpt2_InvalidPath_Throws )
    {
        // Use a clearly invalid path and ensure loadGpt2 throws
        fs::path bad = fs::path( "nonexistent" ) / "file" / "gpt2_tokenizer.bin";
        EXPECT_THROW( BpeTokenizer::loadGpt2( bad ), std::runtime_error );
    }

    TEST( BpeTokenizerGpt2, EncodeDecode_Roundtrip_ShortText_Diagnostic )
    {
        auto p = gpt2_tokenizer_path();
        if ( !fs::exists( p ) ) {
            GTEST_SKIP() << "GPT-2 tokenizer binary not present at: " << p.string();
        }

        BpeTokenizer tokenizer = BpeTokenizer::loadGpt2( p );

        std::string text = "Hello, world!";

        auto enc = tokenizer.encode( text );

        ASSERT_FALSE( enc.empty() );

        for ( auto id : enc ) {
            EXPECT_TRUE( tokenizer.isValidToken( id ) );
        }

        std::string decoded = tokenizer.decode( enc );

        if ( decoded != text ) {
            std::ostringstream msg;

            msg << "Roundtrip mismatch\n";
            msg << "  original      : \"" << text << "\"\n";
            msg << "  original (esc): \"" << escape_for_debug( text ) << "\"\n";
            msg << "  decoded       : \"" << decoded << "\"\n";
            msg << "  decoded (esc) : \"" << escape_for_debug( decoded ) << "\"\n";
            msg << "  token ids (" << enc.size() << "):";
            for ( size_t i = 0; i < enc.size(); ++i ) {
                msg << (i ? "," : "") << enc[i];
            }
            msg << "\n";

            msg << "  tokens:\n";
            for ( size_t i = 0; i < enc.size(); ++i ) {
                auto id = enc[i];
                auto tokstr = tokenizer.tokenToString( id );
                msg << "    [" << i << "] id=" << id << " raw=\"" << tokstr
                    << "\" esc=\"" << escape_for_debug( tokstr ) << "\"\n";
            }

            // Also include raw bytes hex of original to rule out weird characters
            msg << "  original bytes: ";
            for ( unsigned char c : text ) {
                msg << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(c) << " ";
            }
            msg << std::dec << "\n";

            // Emit the collected diagnostic info as a failure
            FAIL() << msg.str();
        }

        EXPECT_EQ( decoded, text );
    }
}