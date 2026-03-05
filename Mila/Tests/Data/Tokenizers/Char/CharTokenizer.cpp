#include <gtest/gtest.h>

import Mila;

#include <filesystem>
#include <sstream>
#include <vector>
#include <chrono>
#include <random>

namespace Data::Tokenizers::Tests
{
    using namespace Mila::Data;
    namespace fs = std::filesystem;

    static fs::path make_temp_file( const std::string& stem = "char_tokenizer_test" )
    {
        auto dir = fs::temp_directory_path();
        auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::mt19937_64 rng( static_cast<unsigned long long>(now) );
        std::uniform_int_distribution<unsigned long long> dist;
        auto rand = dist( rng );

        return dir / (stem + "_" + std::to_string( now ) + "_" + std::to_string( rand ) + ".bin");
    }

    TEST( CharTokenizerTests, EncodeDecode_Roundtrip_Basic )
    {
        std::string corpus = "hello world";

        CharVocabularyConfig cfg;
        SpecialTokens st = {};
        st.use_pad = false;
        st.use_unk = false;
        st.use_bos = false;
        st.use_eos = false;
        cfg.withSpecialTokens( st );

        std::istringstream stream( corpus );
        CharTrainer trainer( cfg );
        trainer.addCorpusFromStream( stream );

        CharVocabulary vocab = trainer.train();
        CharTokenizer tokenizer( std::move( vocab ) );

        std::string input = "hello";
        auto encoded = tokenizer.encode( input );
        auto decoded = tokenizer.decode( std::span<const Mila::Dnn::Data::TokenId>( encoded.data(), encoded.size() ) );

        EXPECT_EQ( decoded, input );
        EXPECT_EQ( tokenizer.getVocabSize(), tokenizer.getVocabSize() ); // trivial sanity check
    }

    TEST( CharTokenizerTests, Decode_InvalidId_ProducesQuestionMark )
    {
        std::string corpus = "ab";

        CharVocabularyConfig cfg;
        SpecialTokens st = {};
        st.use_pad = false;
        st.use_unk = false;
        st.use_bos = false;
        st.use_eos = false;
        cfg.withSpecialTokens( st );

        std::istringstream stream( corpus );
        CharTrainer trainer( cfg );
        trainer.addCorpusFromStream( stream );

        CharVocabulary vocab = trainer.train();
        CharTokenizer tokenizer( std::move( vocab ) );

        // Prepare tokens: a valid id followed by an invalid out-of-range id
        std::vector<Mila::Dnn::Data::TokenId> tokens;
        tokens.push_back( 0u );
        tokens.push_back( static_cast<Mila::Dnn::Data::TokenId>(tokenizer.getVocabSize() + 10) );

        std::string decoded = tokenizer.decode( tokens );

        ASSERT_GE( decoded.size(), 2u );
        EXPECT_NE( decoded[ 0 ], '?' );
        EXPECT_EQ( decoded[ 1 ], '?' );
    }

    TEST( CharTokenizerTests, LoadFromFile_EncodesCorrectly )
    {
        std::string corpus = "aa aa aa bb bb";

        CharVocabularyConfig cfg;
        SpecialTokens st = {};
        st.use_pad = true;
        st.use_unk = true;
        st.use_bos = false;
        st.use_eos = false;
        cfg.withSpecialTokens( st );

        std::istringstream stream( corpus );
        CharTrainer trainer( cfg );
        trainer.addCorpusFromStream( stream );

        CharVocabulary vocab = trainer.train();

        auto tmp = make_temp_file( "char_tok_load" );
        if ( fs::exists( tmp ) ) fs::remove( tmp );

        vocab.save( tmp );
        ASSERT_TRUE( fs::exists( tmp ) );

        CharTokenizer fromMem( CharVocabulary::load( tmp ) );
        CharTokenizer fromFile = CharTokenizer::load( tmp );

        std::string text = "aabb";
        auto enc_mem = fromMem.encode( text );
        auto enc_file = fromFile.encode( text );

        EXPECT_EQ( enc_mem.size(), enc_file.size() );

        fs::remove( tmp );
    }

    TEST( CharTokenizerTests, TokenToString_ReturnsCorrectString )
    {
        std::string corpus = "test";

        CharVocabularyConfig cfg;
        SpecialTokens st = {};
        st.use_pad = true;
        st.use_unk = false;
        cfg.withSpecialTokens( st );

        std::istringstream stream( corpus );
        CharTrainer trainer( cfg );
        trainer.addCorpusFromStream( stream );

        CharVocabulary vocab = trainer.train();
        CharTokenizer tokenizer( std::move( vocab ) );

        auto valid = tokenizer.tokenToString( 0 );
        EXPECT_FALSE( valid.empty() );

        auto invalid = tokenizer.tokenToString( static_cast<Mila::Dnn::Data::TokenId>(tokenizer.getVocabSize() + 100) );
        EXPECT_TRUE( invalid.empty() );
    }

    TEST( CharTokenizerTests, IsValidToken_ChecksRange )
    {
        std::string corpus = "abc";

        CharVocabularyConfig cfg;
        SpecialTokens st = {};
        st.use_pad = false;
        st.use_unk = false;
        cfg.withSpecialTokens( st );

        std::istringstream stream( corpus );
        CharTrainer trainer( cfg );
        trainer.addCorpusFromStream( stream );

        CharVocabulary vocab = trainer.train();
        CharTokenizer tokenizer( std::move( vocab ) );

        EXPECT_TRUE( tokenizer.isValidToken( 0u ) );
        EXPECT_TRUE( tokenizer.isValidToken( static_cast<Mila::Dnn::Data::TokenId>(tokenizer.getVocabSize() - 1) ) );
        EXPECT_FALSE( tokenizer.isValidToken( static_cast<Mila::Dnn::Data::TokenId>(tokenizer.getVocabSize()) ) );
    }

    TEST( CharTokenizerTests, Encode_EmptyString_ReturnsEmptyVector )
    {
        CharVocabularyConfig cfg;
        SpecialTokens st = {};
        st.use_pad = false;
        st.use_unk = false;
        cfg.withSpecialTokens( st );

        CharVocabulary vocab = CharTrainer( cfg ).train();
        CharTokenizer tokenizer( std::move( vocab ) );

        auto encoded = tokenizer.encode( "" );
        EXPECT_TRUE( encoded.empty() );
    }

    TEST( CharTokenizerTests, Decode_EmptyVector_ReturnsEmptyString )
    {
        CharVocabularyConfig cfg;
        SpecialTokens st = {};
        st.use_pad = false;
        st.use_unk = false;
        cfg.withSpecialTokens( st );

        CharVocabulary vocab = CharTrainer( cfg ).train();
        CharTokenizer tokenizer( std::move( vocab ) );

        std::vector<Mila::Dnn::Data::TokenId> empty;
        auto decoded = tokenizer.decode( empty );
        EXPECT_TRUE( decoded.empty() );
    }

    TEST( CharTokenizerTests, Encode_UnknownChar_EmitsZeroFallback )
    {
        // When vocabulary does not provide a mapping (tokenToId returns nullopt),
        // CharTokenizer should emit 0u for that input byte.
        std::string corpus = "ab";

        CharVocabularyConfig cfg;
        SpecialTokens st = {};
        st.use_pad = false;
        st.use_unk = false;
        cfg.withSpecialTokens( st );

        std::istringstream stream( corpus );
        CharTrainer trainer( cfg );
        trainer.addCorpusFromStream( stream );

        CharVocabulary vocab = trainer.train();
        CharTokenizer tokenizer( std::move( vocab ) );

        std::string input = "azb";
        auto encoded = tokenizer.encode( input );

        // middle character 'z' is unknown and should map to 0u
        ASSERT_EQ( encoded.size(), 3u );
        EXPECT_EQ( encoded[ 1 ], static_cast<Mila::Dnn::Data::TokenId>(0u) );
    }
}