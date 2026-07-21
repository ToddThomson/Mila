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

    TEST( BpeTokenizerGemma, Encode_StartOfTurn_IsSingleAtomicToken )
    {
        auto p = gemma_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Gemma tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadGemma( p );

        // REVIEW: This test is failing because the Gemma 4 tokenizer binary does not contain a <start_of_turn> token.
        // Gemma 4 use <|turn> and <turn|> as special tokens for turn boundaries. The test should be updated to use the correct special tokens.

        // Registered from the loaded vocab, so it must match as one token, not subword.
        auto ids = tokenizer->encode( "<start_of_turn>" );
        EXPECT_EQ( ids.size(), 1u );
    }
}
