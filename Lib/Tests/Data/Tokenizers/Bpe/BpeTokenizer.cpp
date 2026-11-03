/*!
 * @file Gpt4Tokenizer.cpp
 * @brief Unit tests for BpeTokenizer with Llama 3.2 binary format vocabulary loading.
 *
 * Constructs a minimal in-memory Llama 3.2-format vocabulary and exercises
 * construction, vocabulary metadata, encode (plain text, special-token pre-pass,
 * mixed), decode, and encode/decode roundtrips via BpeTokenizer::loadLlama32.
 */

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <span>
#include <filesystem>
#include <fstream>
#include <cstdint>
#include <optional>
#include <memory>
#include <stdexcept>

import Mila;

namespace Mila::Data::Tests
{
    using Mila::Data::BpeTokenizer;
    using Mila::Data::TokenId;

    // Minimal test vocabulary layout (Llama 3.2 binary format):
    //   ID 0-4 : 'H', 'e', 'l', 'o', '!'
    //   ID 5   : "<|begin_of_text|>"  (BOS)
    //   ID 6   : "<|end_of_text|>"    (EOS)
    // All characters are printable ASCII, so the GPT-2 byte encoder maps them
    // to identical single-character strings. No BPE merge rules are present.

    static constexpr uint32_t kTestVocabSize = 7;
    static constexpr uint32_t kBosId = 5;
    static constexpr uint32_t kEosId = 6;
    static constexpr const char* kBosStr = "<|begin_of_text|>";
    static constexpr const char* kEosStr = "<|end_of_text|>";

    static std::filesystem::path writeLlama32TestVocab()
    {
        auto path = std::filesystem::temp_directory_path() / "mila_test_gpt4_vocab.bin";
        std::ofstream f( path, std::ios::binary );

        if ( !f )
        {
            throw std::runtime_error( "Cannot create test vocabulary file: " + path.string() );
        }

        auto write_u32 = [&]( uint32_t v )
            {
                f.write( reinterpret_cast<const char*>(&v), sizeof( v ) );
            };

        auto write_u8 = [&]( uint8_t v )
            {
                f.write( reinterpret_cast<const char*>(&v), sizeof( v ) );
            };

        auto write_f32 = [&]( float v )
            {
                f.write( reinterpret_cast<const char*>(&v), sizeof( v ) );
            };

        auto write_token = [&]( const std::string& token, uint32_t id )
            {
                write_u32( static_cast<uint32_t>(token.size()) );
                f.write( token.data(), static_cast<std::streamsize>(token.size()) );
                write_f32( 0.0f );
                write_u32( id );
            };

        write_u32( kTestVocabSize );
        write_u8( 1 );  // use_byte_fallback

        write_token( "H", 0 );
        write_token( "e", 1 );
        write_token( "l", 2 );
        write_token( "o", 3 );
        write_token( "!", 4 );
        write_token( kBosStr, 5 );
        write_token( kEosStr, 6 );

        // BOS, EOS present; PAD and UNK absent.
        write_u32( 1 );  write_u32( kBosId );
        write_u32( 1 );  write_u32( kEosId );
        write_u32( 0 );
        write_u32( 0 );

        return path;
    }

    class BpeTokenizerLlama32Test : public ::testing::Test
    {
    protected:

        void SetUp() override
        {
            vocab_path_ = writeLlama32TestVocab();
            tokenizer_ = BpeTokenizer::loadLlama32( vocab_path_ );
        }

        void TearDown() override
        {
            std::filesystem::remove( vocab_path_ );
        }

        std::filesystem::path vocab_path_;
        std::shared_ptr<BpeTokenizer> tokenizer_;
    };

    // ---- Construction -------------------------------------------------------

    TEST_F( BpeTokenizerLlama32Test, LoadLlama32_ValidFile_ReturnsNonNull )
    {
        EXPECT_NE( tokenizer_, nullptr );
    }

    TEST( BpeTokenizerLlama32ConstructionTest, LoadLlama32_MissingFile_ThrowsRuntimeError )
    {
        EXPECT_THROW(
            BpeTokenizer::loadLlama32( "nonexistent_directory/vocab.bin" ),
            std::runtime_error );
    }

    TEST_F( BpeTokenizerLlama32Test, Constructor_DirectVocab_VocabIsAccessible )
    {
        EXPECT_EQ( tokenizer_->getVocab().getSize(), static_cast<size_t>(kTestVocabSize) );
    }

    // ---- Vocabulary Metadata ------------------------------------------------

    TEST_F( BpeTokenizerLlama32Test, GetVocabSize_ReturnsExpectedCount )
    {
        EXPECT_EQ( tokenizer_->getVocabSize(), static_cast<size_t>(kTestVocabSize) );
    }

    TEST_F( BpeTokenizerLlama32Test, GetBosTokenId_ReturnsExpectedId )
    {
        auto bos = tokenizer_->getBosTokenId();
        ASSERT_TRUE( bos.has_value() );
        EXPECT_EQ( *bos, static_cast<TokenId>(kBosId) );
    }

    TEST_F( BpeTokenizerLlama32Test, GetEosTokenId_ReturnsExpectedId )
    {
        auto eos = tokenizer_->getEosTokenId();
        ASSERT_TRUE( eos.has_value() );
        EXPECT_EQ( *eos, static_cast<TokenId>(kEosId) );
    }

    TEST_F( BpeTokenizerLlama32Test, GetPadTokenId_LlamaStyle_ReturnsNullopt )
    {
        // Llama-style config has use_pad = false; no PAD token is registered.
        EXPECT_FALSE( tokenizer_->getPadTokenId().has_value() );
    }

    TEST_F( BpeTokenizerLlama32Test, IsValidToken_InRangeIds_ReturnsTrue )
    {
        EXPECT_TRUE( tokenizer_->isValidToken( 0 ) );
        EXPECT_TRUE( tokenizer_->isValidToken( 4 ) );
        EXPECT_TRUE( tokenizer_->isValidToken( static_cast<TokenId>(kBosId) ) );
        EXPECT_TRUE( tokenizer_->isValidToken( static_cast<TokenId>(kEosId) ) );
    }

    TEST_F( BpeTokenizerLlama32Test, IsValidToken_OutOfRangeId_ReturnsFalse )
    {
        EXPECT_FALSE( tokenizer_->isValidToken( static_cast<TokenId>(kTestVocabSize) ) );
        EXPECT_FALSE( tokenizer_->isValidToken( 9999 ) );
    }

    TEST_F( BpeTokenizerLlama32Test, IsValidToken_NegativeId_ReturnsFalse )
    {
        EXPECT_FALSE( tokenizer_->isValidToken( -1 ) );
    }

    TEST_F( BpeTokenizerLlama32Test, TokenToString_ValidIds_ReturnExpectedStrings )
    {
        EXPECT_EQ( tokenizer_->tokenToString( 0 ), "H" );
        EXPECT_EQ( tokenizer_->tokenToString( 1 ), "e" );
        EXPECT_EQ( tokenizer_->tokenToString( 2 ), "l" );
        EXPECT_EQ( tokenizer_->tokenToString( 3 ), "o" );
        EXPECT_EQ( tokenizer_->tokenToString( 4 ), "!" );
        EXPECT_EQ( tokenizer_->tokenToString( static_cast<TokenId>(kBosId) ), kBosStr );
        EXPECT_EQ( tokenizer_->tokenToString( static_cast<TokenId>(kEosId) ), kEosStr );
    }

    TEST_F( BpeTokenizerLlama32Test, TokenToString_InvalidId_ReturnsEmptyString )
    {
        EXPECT_EQ( tokenizer_->tokenToString( static_cast<TokenId>(kTestVocabSize) ), "" );
        EXPECT_EQ( tokenizer_->tokenToString( 9999 ), "" );
        EXPECT_EQ( tokenizer_->tokenToString( -1 ), "" );
    }

    // ---- Encode -------------------------------------------------------------

    TEST_F( BpeTokenizerLlama32Test, Encode_EmptyString_ReturnsEmptyVector )
    {
        EXPECT_TRUE( tokenizer_->encode( "" ).empty() );
    }

    TEST_F( BpeTokenizerLlama32Test, Encode_PlainAsciiText_ReturnsExpectedIds )
    {
        // "Hello!" splits into ["Hello", "!"] by the Llama3 ASCII fallback regex.
        // Each character byte-encodes to itself (printable ASCII), no BPE merges apply.
        auto ids = tokenizer_->encode( "Hello!" );
        ASSERT_EQ( ids.size(), 6u );
        EXPECT_EQ( ids[ 0 ], 0 );  // H
        EXPECT_EQ( ids[ 1 ], 1 );  // e
        EXPECT_EQ( ids[ 2 ], 2 );  // l
        EXPECT_EQ( ids[ 3 ], 2 );  // l
        EXPECT_EQ( ids[ 4 ], 3 );  // o
        EXPECT_EQ( ids[ 5 ], 4 );  // !
    }

    TEST_F( BpeTokenizerLlama32Test, Encode_SingleKnownChar_ReturnsSingleId )
    {
        auto ids = tokenizer_->encode( "H" );
        ASSERT_EQ( ids.size(), 1u );
        EXPECT_EQ( ids[ 0 ], 0 );
    }

    TEST_F( BpeTokenizerLlama32Test, Encode_DuplicateChar_ReturnsRepeatedId )
    {
        auto ids = tokenizer_->encode( "ll" );
        ASSERT_EQ( ids.size(), 2u );
        EXPECT_EQ( ids[ 0 ], 2 );
        EXPECT_EQ( ids[ 1 ], 2 );
    }

    TEST_F( BpeTokenizerLlama32Test, Encode_UnknownByteToken_FallsBackToIdZero )
    {
        // 'W' is absent from the test vocabulary; encodeSegment maps unknown
        // tokens to ID 0 (the nullopt branch of "id ? *id : 0").
        auto ids = tokenizer_->encode( "W" );
        ASSERT_EQ( ids.size(), 1u );
        EXPECT_EQ( ids[ 0 ], 0 );
    }

    TEST_F( BpeTokenizerLlama32Test, Encode_BosSpecialTokenOnly_ReturnsDirectId )
    {
        auto ids = tokenizer_->encode( kBosStr );
        ASSERT_EQ( ids.size(), 1u );
        EXPECT_EQ( ids[ 0 ], static_cast<TokenId>(kBosId) );
    }

    TEST_F( BpeTokenizerLlama32Test, Encode_EosSpecialTokenOnly_ReturnsDirectId )
    {
        auto ids = tokenizer_->encode( kEosStr );
        ASSERT_EQ( ids.size(), 1u );
        EXPECT_EQ( ids[ 0 ], static_cast<TokenId>(kEosId) );
    }

    TEST_F( BpeTokenizerLlama32Test, Encode_SpecialTokenAtStart_PrependsSpecialId )
    {
        // "<|begin_of_text|>Hello!" -> [5, 0, 1, 2, 2, 3, 4]
        auto ids = tokenizer_->encode( std::string( kBosStr ) + "Hello!" );
        ASSERT_EQ( ids.size(), 7u );
        EXPECT_EQ( ids[ 0 ], static_cast<TokenId>(kBosId) );
        EXPECT_EQ( ids[ 1 ], 0 );
        EXPECT_EQ( ids[ 2 ], 1 );
        EXPECT_EQ( ids[ 3 ], 2 );
        EXPECT_EQ( ids[ 4 ], 2 );
        EXPECT_EQ( ids[ 5 ], 3 );
        EXPECT_EQ( ids[ 6 ], 4 );
    }

    TEST_F( BpeTokenizerLlama32Test, Encode_SpecialTokenAtEnd_AppendsSpecialId )
    {
        // "Hello!<|end_of_text|>" -> [0, 1, 2, 2, 3, 4, 6]
        auto ids = tokenizer_->encode( "Hello!" + std::string( kEosStr ) );
        ASSERT_EQ( ids.size(), 7u );
        EXPECT_EQ( ids[ 0 ], 0 );
        EXPECT_EQ( ids[ 5 ], 4 );
        EXPECT_EQ( ids[ 6 ], static_cast<TokenId>(kEosId) );
    }

    TEST_F( BpeTokenizerLlama32Test, Encode_SpecialTokensSurroundingText_FullSequence )
    {
        // "<|begin_of_text|>Hello!<|end_of_text|>" -> [5, 0, 1, 2, 2, 3, 4, 6]
        auto ids = tokenizer_->encode(
            std::string( kBosStr ) + "Hello!" + std::string( kEosStr ) );
        ASSERT_EQ( ids.size(), 8u );
        EXPECT_EQ( ids.front(), static_cast<TokenId>(kBosId) );
        EXPECT_EQ( ids[ 1 ], 0 );
        EXPECT_EQ( ids[ 2 ], 1 );
        EXPECT_EQ( ids[ 3 ], 2 );
        EXPECT_EQ( ids[ 4 ], 2 );
        EXPECT_EQ( ids[ 5 ], 3 );
        EXPECT_EQ( ids[ 6 ], 4 );
        EXPECT_EQ( ids.back(), static_cast<TokenId>(kEosId) );
    }

    TEST_F( BpeTokenizerLlama32Test, Encode_AdjacentSpecialTokens_BothResolvedDirectly )
    {
        // "<|begin_of_text|><|end_of_text|>" -> [5, 6]
        auto ids = tokenizer_->encode( std::string( kBosStr ) + std::string( kEosStr ) );
        ASSERT_EQ( ids.size(), 2u );
        EXPECT_EQ( ids[ 0 ], static_cast<TokenId>(kBosId) );
        EXPECT_EQ( ids[ 1 ], static_cast<TokenId>(kEosId) );
    }

    TEST_F( BpeTokenizerLlama32Test, Encode_SpecialTokenRepeated_ProducesCorrectCount )
    {
        auto ids = tokenizer_->encode( std::string( kBosStr ) + std::string( kBosStr ) );
        ASSERT_EQ( ids.size(), 2u );
        EXPECT_EQ( ids[ 0 ], static_cast<TokenId>(kBosId) );
        EXPECT_EQ( ids[ 1 ], static_cast<TokenId>(kBosId) );
    }

    // ---- Decode -------------------------------------------------------------

    TEST_F( BpeTokenizerLlama32Test, Decode_EmptySpan_ReturnsEmptyString )
    {
        std::vector<TokenId> ids;
        EXPECT_EQ( tokenizer_->decode( ids ), "" );
    }

    TEST_F( BpeTokenizerLlama32Test, Decode_SingleCharToken_ReturnsChar )
    {
        std::vector<TokenId> ids = { 0 };
        EXPECT_EQ( tokenizer_->decode( ids ), "H" );
    }

    TEST_F( BpeTokenizerLlama32Test, Decode_MultipleCharTokens_ReturnsDecodedString )
    {
        std::vector<TokenId> ids = { 0, 1, 2, 2, 3, 4 };
        EXPECT_EQ( tokenizer_->decode( ids ), "Hello!" );
    }

    TEST_F( BpeTokenizerLlama32Test, Decode_BosToken_ReturnsSpecialTokenString )
    {
        std::vector<TokenId> ids = { static_cast<TokenId>(kBosId) };
        EXPECT_EQ( tokenizer_->decode( ids ), std::string( kBosStr ) );
    }

    TEST_F( BpeTokenizerLlama32Test, Decode_EosToken_ReturnsSpecialTokenString )
    {
        std::vector<TokenId> ids = { static_cast<TokenId>(kEosId) };
        EXPECT_EQ( tokenizer_->decode( ids ), std::string( kEosStr ) );
    }

    TEST_F( BpeTokenizerLlama32Test, Decode_InvalidTokenId_ReturnsQuestionMark )
    {
        std::vector<TokenId> ids = { 9999 };
        EXPECT_EQ( tokenizer_->decode( ids ), "?" );
    }

    TEST_F( BpeTokenizerLlama32Test, Decode_NegativeTokenId_ReturnsQuestionMark )
    {
        std::vector<TokenId> ids = { -1 };
        EXPECT_EQ( tokenizer_->decode( ids ), "?" );
    }

    TEST_F( BpeTokenizerLlama32Test, Decode_MixedValidAndInvalidIds_SubstitutesQuestionMark )
    {
        // [0, 9999, 1] -> "H?e"
        std::vector<TokenId> ids = { 0, 9999, 1 };
        EXPECT_EQ( tokenizer_->decode( ids ), "H?e" );
    }

    TEST_F( BpeTokenizerLlama32Test, Decode_FullSequenceWithSpecialTokens_ReturnsCorrectString )
    {
        // [5, 0, 1, 2, 2, 3, 4, 6] -> "<|begin_of_text|>Hello!<|end_of_text|>"
        std::vector<TokenId> ids = {
            static_cast<TokenId>(kBosId), 0, 1, 2, 2, 3, 4,
            static_cast<TokenId>(kEosId)
        };
        EXPECT_EQ( tokenizer_->decode( ids ),
            std::string( kBosStr ) + "Hello!" + std::string( kEosStr ) );
    }

    // ---- Encode / Decode Roundtrip ------------------------------------------

    TEST_F( BpeTokenizerLlama32Test, EncodeDecodeRoundtrip_PlainText_PreservesContent )
    {
        const std::string text = "Hello!";
        auto ids = tokenizer_->encode( text );
        EXPECT_EQ( tokenizer_->decode( ids ), text );
    }

    TEST_F( BpeTokenizerLlama32Test, EncodeDecodeRoundtrip_BosEosSurroundedText_PreservesContent )
    {
        const std::string text =
            std::string( kBosStr ) + "Hello!" + std::string( kEosStr );
        auto ids = tokenizer_->encode( text );
        EXPECT_EQ( tokenizer_->decode( ids ), text );
    }

    TEST_F( BpeTokenizerLlama32Test, EncodeDecodeRoundtrip_AdjacentSpecialTokens_PreservesContent )
    {
        const std::string text = std::string( kBosStr ) + std::string( kEosStr );
        auto ids = tokenizer_->encode( text );
        EXPECT_EQ( tokenizer_->decode( ids ), text );
    }

    TEST_F( BpeTokenizerLlama32Test, EncodeDecodeRoundtrip_EachKnownChar_PreservesContent )
    {
        for ( const std::string& ch : { "H", "e", "l", "o", "!" } )
        {
            auto ids = tokenizer_->encode( ch );
            EXPECT_EQ( tokenizer_->decode( ids ), ch ) << "Roundtrip failed for char: " << ch;
        }
    }
}