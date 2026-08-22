/*!
 * @file BpeTokenizer.Qwen.cpp
 * @brief Integration tests for BpeTokenizer loading the Qwen 3.8 byte-level BPE binary.
 *
 * Gated on the converted tokenizer binary being present (skips otherwise), like the
 * Llama 3.2 and Gemma suites. Produce it with:
 *   python Tools/Converters/Qwen/convert_tokenizer.py --model Qwen/Qwen3.8-27B \
 *       --output <TEST_DATA_DIR>/models/qwen/qwen38_tokenizer.bin
 *
 * The ground-truth ids below are the checkpoint reference printed by that converter
 * for the same string, so a green Encode_PlainText test is byte-for-byte parity with
 * the published tokenizer on this input.
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

    static fs::path qwen_tokenizer_path()
    {
        fs::path dataDir = TEST_DATA_DIR;
        return dataDir / "models" / "qwen" / "qwen38_tokenizer.bin";
    }

    // ---- Load / skip --------------------------------------------------------

    TEST( BpeTokenizerQwen, Load_SucceedsAndHasCorrectVocabSize )
    {
        auto p = qwen_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Qwen tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadQwen( p );

        // 248,044 learned pieces plus 33 added control tokens. This is deliberately
        // NOT the model's vocab_size of 248,320 -- the checkpoint pads the embedding
        // table, so the last 243 rows have no piece and must never be sampled.
        size_t vsz = tokenizer->getVocabSize();
        EXPECT_EQ( vsz, 248077u );

        EXPECT_TRUE( tokenizer->isValidToken( static_cast<TokenId>(0u) ) );
        EXPECT_TRUE( tokenizer->isValidToken( static_cast<TokenId>(vsz - 1) ) );
        EXPECT_FALSE( tokenizer->isValidToken( static_cast<TokenId>(vsz) ) );
    }

    TEST( BpeTokenizerQwen, LoadQwen_InvalidPath_Throws )
    {
        fs::path bad = fs::path( "nonexistent" ) / "file" / "qwen38_tokenizer.bin";
        EXPECT_THROW( BpeTokenizer::loadQwen( bad ), std::runtime_error );
    }

    // ---- Special token metadata ---------------------------------------------

    TEST( BpeTokenizerQwen, SpecialTokenIds_MatchQwenConvention )
    {
        auto p = qwen_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Qwen tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadQwen( p );

        // EOS is the TURN end, not the document terminator -- getting these two the
        // wrong way round produces a model that never stops on a chat turn.
        auto eos = tokenizer->getEosTokenId();
        auto pad = tokenizer->getPadTokenId();

        ASSERT_TRUE( eos.has_value() );
        ASSERT_TRUE( pad.has_value() );
        EXPECT_EQ( *eos, static_cast<TokenId>(248046u) );   // <|im_end|>
        EXPECT_EQ( *pad, static_cast<TokenId>(248044u) );   // <|endoftext|>

        // Qwen has no BOS: add_bos_token is false in the checkpoint.
        EXPECT_FALSE( tokenizer->getBosTokenId().has_value() );
    }

    // ---- Encode parity with the checkpoint ----------------------------------

    TEST( BpeTokenizerQwen, Encode_PlainText_MatchesCheckpointOutput )
    {
        auto p = qwen_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Qwen tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadQwen( p );

        // Ground truth from convert_tokenizer.py:
        //   ['The', 'Ġcapital', 'Ġof', 'ĠFrance', 'Ġis', 'ĠParis', '.']
        const std::string text = "The capital of France is Paris.";
        const std::vector<TokenId> expected = { 760, 6511, 314, 9338, 369, 11751, 13 };

        auto ids = tokenizer->encode( text );

        ASSERT_EQ( ids.size(), expected.size() );

        for ( size_t i = 0; i < ids.size(); ++i )
        {
            EXPECT_EQ( ids[ i ], expected[ i ] ) << "Token mismatch at position " << i;
        }
    }

    TEST( BpeTokenizerQwen, Encode_Digits_AreOneTokenEach )
    {
        auto p = qwen_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Qwen tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadQwen( p );

        // Pins per-digit tokenization as a property of the loaded vocabulary. It is
        // NOT a check on the split pattern: this vocabulary holds no multi-digit piece
        // and no digit-digit merge rule (verified against the checkpoint), so the merge
        // loop cannot join digits whichever pattern grouped them. What this catches is a
        // future vocabulary that gained multi-digit pieces without the pattern changing.
        //   ['4', '2', 'Ġitems', 'Ġcost', 'Ġ$', '3', '.', '5', '0']
        const std::string text = "42 items cost $3.50";
        const std::vector<TokenId> expected = { 19, 17, 3470, 2695, 393, 18, 13, 20, 15 };

        auto ids = tokenizer->encode( text );

        ASSERT_EQ( ids.size(), expected.size() );

        for ( size_t i = 0; i < ids.size(); ++i )
        {
            EXPECT_EQ( ids[ i ], expected[ i ] ) << "Token mismatch at position " << i;
        }
    }

    TEST( BpeTokenizerQwen, EncodeDecode_Roundtrip_RestoresText )
    {
        auto p = qwen_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Qwen tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadQwen( p );

        const std::string text = "The capital of France is Paris.";
        auto enc = tokenizer->encode( text );

        ASSERT_FALSE( enc.empty() );
        EXPECT_EQ( tokenizer->decode( enc ), text );
    }

    // ---- Control tokens are matched atomically ------------------------------

    TEST( BpeTokenizerQwen, Encode_ControlTokens_AreSingleAtomicTokens )
    {
        auto p = qwen_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Qwen tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadQwen( p );

        // The set BpeVocabulary::loadQwen registers from the checkpoint vocabulary.
        // Stated literally rather than imported from the loader so the test is an
        // independent expectation, not a restatement of the code under test.
        const std::vector<std::string> control_tokens = {
            "<|endoftext|>", "<|im_start|>", "<|im_end|>",
            "<think>", "</think>",
            "<tool_call>", "</tool_call>",
            "<tool_response>", "</tool_response>" };

        for ( const auto& token : control_tokens )
        {
            auto ids = tokenizer->encode( token );
            EXPECT_EQ( ids.size(), 1u ) << "Control token not matched atomically: " << token;
        }
    }

    TEST( BpeTokenizerQwen, Encode_ChatMlTurn_ResolvesToStopTokenIds )
    {
        auto p = qwen_tokenizer_path();

        if ( !fs::exists( p ) )
        {
            GTEST_SKIP() << "Qwen tokenizer binary not present at: " << p.string();
        }

        std::shared_ptr<BpeTokenizer> tokenizer = BpeTokenizer::loadQwen( p );

        // Pins the two ids QwenModel hardcodes as its stop set against the vocabulary
        // they are supposed to come from.
        auto turn_open = tokenizer->encode( "<|im_start|>" );
        auto turn_close = tokenizer->encode( "<|im_end|>" );
        auto end_of_text = tokenizer->encode( "<|endoftext|>" );

        ASSERT_EQ( turn_open.size(), 1u );
        ASSERT_EQ( turn_close.size(), 1u );
        ASSERT_EQ( end_of_text.size(), 1u );
        EXPECT_EQ( turn_open[ 0 ], static_cast<TokenId>(248045u) );
        EXPECT_EQ( turn_close[ 0 ], static_cast<TokenId>(248046u) );
        EXPECT_EQ( end_of_text[ 0 ], static_cast<TokenId>(248044u) );
    }
}
