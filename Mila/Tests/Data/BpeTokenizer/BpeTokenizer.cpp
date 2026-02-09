#include <gtest/gtest.h>

import Mila;

#include <filesystem>
#include <sstream>
#include <vector>
#include <chrono>
#include <random>

namespace Data::Tokenizers::BpeTokenizer_Tests
{
    using namespace Mila::Data;
    namespace fs = std::filesystem;

    static fs::path make_temp_file( const std::string& stem = "bpe_tokenizer_test" )
    {
        auto dir = fs::temp_directory_path();
        auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::mt19937_64 rng( static_cast<unsigned long long>( now ) );
        std::uniform_int_distribution<unsigned long long> dist;
        auto rand = dist( rng );

        return dir / (stem + "_" + std::to_string( now ) + "_" + std::to_string( rand ) + ".bin");
    }

    TEST( BpeTokenizerTests, EncodeDecode_Roundtrip_ByteLevel )
    {
        std::string corpus = "hello world, and hello world again! Must have some merges!";
        
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 300 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::none() );

        BpeVocabulary vocab = BpeVocabulary::train( corpus, cfg );
        BpeTokenizer tokenizer( std::move( vocab ) );

        std::string input = "hello";

        auto encoded = tokenizer.encode( input );
        auto decoded = tokenizer.decode( std::span<const Mila::Dnn::Data::TokenId>( encoded.data(), encoded.size() ) );

        EXPECT_EQ( decoded, input );
        EXPECT_GT( tokenizer.getVocabSize(), 256u );

        if ( !encoded.empty() )
        {
            auto firstId = encoded.front();
            EXPECT_FALSE( tokenizer.tokenToString( firstId ).empty() );
            EXPECT_TRUE( tokenizer.isValidToken( firstId ) );
        }

        EXPECT_FALSE( tokenizer.isValidToken( static_cast<Mila::Dnn::Data::TokenId>( tokenizer.getVocabSize() + 100 ) ) );
    }

    TEST( BpeTokenizerTests, Encode_WithSpecialTokens_IncludesTokenIds )
    {
        std::string corpus = "test corpus";
        
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 300 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::standard() );

        BpeVocabulary vocab = BpeVocabulary::train( corpus, cfg );
        BpeTokenizer tokenizer( std::move( vocab ) );

        std::string input = "test";
        auto encoded = tokenizer.encode( input );

        EXPECT_FALSE( encoded.empty() );
        EXPECT_GT( tokenizer.getVocabSize(), 256u );
    }

    TEST( BpeTokenizerTests, Decode_InvalidId_ProducesQuestionMark )
    {
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 260 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::none() );

        BpeVocabulary vocab = BpeVocabulary::train( std::string(), cfg );
        BpeTokenizer tokenizer( std::move( vocab ) );

        std::vector<Mila::Dnn::Data::TokenId> tokens;
        tokens.push_back( 0u );
        tokens.push_back( static_cast<Mila::Dnn::Data::TokenId>( tokenizer.getVocabSize() + 5 ) );

        std::string decoded = tokenizer.decode( tokens );

        ASSERT_GE( decoded.size(), 2u );
        EXPECT_NE( decoded[ 0 ], '?' );
        EXPECT_EQ( decoded[ 1 ], '?' );
    }

    TEST( BpeTokenizerTests, LoadFromFile_EncodesCorrectly )
    {
        std::string corpus = "aa aa aa bb bb";
        
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 300 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::standard() )
            .withMinFrequency( 1 );

        BpeVocabulary vocab = BpeVocabulary::train( corpus, cfg );

        auto tmp = make_temp_file( "bpe_tok_load" );
        if ( fs::exists( tmp ) ) fs::remove( tmp );

        vocab.save( tmp );
        ASSERT_TRUE( fs::exists( tmp ) );

        BpeTokenizer fromMem( BpeVocabulary::train( corpus, cfg ) );
        BpeTokenizer fromFile = BpeTokenizer::load( tmp );

        std::string text = "aabb";
        auto enc_mem = fromMem.encode( text );
        auto enc_file = fromFile.encode( text );

        EXPECT_EQ( enc_mem.size(), enc_file.size() );

        fs::remove( tmp );
    }

    TEST( BpeTokenizerTests, TokenToString_ReturnsCorrectString )
    {
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 300 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::standard() );

        BpeVocabulary vocab = BpeVocabulary::train( "test corpus", cfg );
        BpeTokenizer tokenizer( std::move( vocab ) );

        auto pad_tok = tokenizer.tokenToString( 256 );
        EXPECT_FALSE( pad_tok.empty() );

        auto invalid_tok = tokenizer.tokenToString( static_cast<Mila::Dnn::Data::TokenId>( tokenizer.getVocabSize() + 100 ) );
        EXPECT_TRUE( invalid_tok.empty() );
    }

    TEST( BpeTokenizerTests, IsValidToken_ChecksRange )
    {
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 256 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::none() );

        BpeVocabulary vocab = BpeVocabulary::train( std::string(), cfg );
        BpeTokenizer tokenizer( std::move( vocab ) );

        EXPECT_TRUE( tokenizer.isValidToken( 0u ) );
        EXPECT_TRUE( tokenizer.isValidToken( 255u ) );
        EXPECT_FALSE( tokenizer.isValidToken( 256u ) );
        EXPECT_FALSE( tokenizer.isValidToken( 1000u ) );
    }

    TEST( BpeTokenizerTests, GetVocabSize_ReturnsCorrectSize )
    {
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 300 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::standard() );

        BpeVocabulary vocab = BpeVocabulary::train( "test", cfg );
        size_t expected_size = vocab.getSize();

        BpeTokenizer tokenizer( std::move( vocab ) );

        EXPECT_EQ( tokenizer.getVocabSize(), expected_size );
    }

    TEST( BpeTokenizerTests, Encode_EmptyString_ReturnsEmptyVector )
    {
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 256 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::none() );

        BpeVocabulary vocab = BpeVocabulary::train( std::string(), cfg );
        BpeTokenizer tokenizer( std::move( vocab ) );

        auto encoded = tokenizer.encode( "" );
        EXPECT_TRUE( encoded.empty() );
    }

    TEST( BpeTokenizerTests, Decode_EmptyVector_ReturnsEmptyString )
    {
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 256 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::none() );

        BpeVocabulary vocab = BpeVocabulary::train( std::string(), cfg );
        BpeTokenizer tokenizer( std::move( vocab ) );

        std::vector<Mila::Dnn::Data::TokenId> empty;
        auto decoded = tokenizer.decode( empty );
        EXPECT_TRUE( decoded.empty() );
    }

    TEST( BpeTokenizerTests, EncodeDecode_Roundtrip_WithMerges )
    {
        std::string corpus = "aaa aaa aaa bbb bbb bbb";
        
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 270 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::none() )
            .withMinFrequency( 1 );

        BpeVocabulary vocab = BpeVocabulary::train( corpus, cfg );
        BpeTokenizer tokenizer( std::move( vocab ) );

        std::string input = "aaabbb";
        auto encoded = tokenizer.encode( input );
        auto decoded = tokenizer.decode( encoded );

        EXPECT_EQ( decoded, input );
    }

    TEST( BpeTokenizerTests, Encode_LongText_Succeeds )
    {
        std::string corpus = "the quick brown fox jumps over the lazy dog ";
        
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 350 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::standard() );

        BpeVocabulary vocab = BpeVocabulary::train( corpus, cfg );
        BpeTokenizer tokenizer( std::move( vocab ) );

        std::string long_text;
        for ( int i = 0; i < 100; ++i )
        {
            long_text += corpus;
        }

        auto encoded = tokenizer.encode( long_text );
        EXPECT_FALSE( encoded.empty() );
        EXPECT_LT( encoded.size(), long_text.size() );
    }

    TEST( BpeTokenizerTests, EncodeDecode_Roundtrip_UnicodeText )
    {
        std::string corpus = "Hello 世界 🌍, and hello 世界 🌍 again!";
        
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 300 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::none() );

        BpeVocabulary vocab = BpeVocabulary::train( corpus, cfg );
        BpeTokenizer tokenizer( std::move( vocab ) );

        std::string input = "Hello 世界";
        auto encoded = tokenizer.encode( input );
        auto decoded = tokenizer.decode( encoded );

        EXPECT_EQ( decoded, input );
    }

    TEST( BpeTokenizerTests, LoadGpt2_FromTestDataDir )
    {
        fs::path dataDir = TEST_DATA_DIR;
        fs::path tokenizerPath = dataDir / "models" / "gpt2" / "gpt2_tokenizer.bin";

        if ( !fs::exists( tokenizerPath ) )
        {
            GTEST_SKIP() << "GPT-2 tokenizer binary not present at: " << tokenizerPath.string();
        }

        ASSERT_NO_THROW(
        {
            BpeTokenizer tokenizer = BpeTokenizer::loadGpt2( tokenizerPath );
            EXPECT_GT( tokenizer.getVocabSize(), 0u );
            
            std::string test_text = "Hello world";
            auto encoded = tokenizer.encode( test_text );
            EXPECT_FALSE( encoded.empty() );
            
            auto decoded = tokenizer.decode( encoded );
            EXPECT_FALSE( decoded.empty() );
        });
    }

    TEST( BpeTokenizerTests, MoveConstructor_TransfersOwnership )
    {
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 300 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::standard() );

        BpeVocabulary vocab = BpeVocabulary::train( "test", cfg );
        size_t expected_size = vocab.getSize();

        BpeTokenizer tokenizer1( std::move( vocab ) );
        BpeTokenizer tokenizer2( std::move( tokenizer1 ) );

        EXPECT_EQ( tokenizer2.getVocabSize(), expected_size );
        
        auto encoded = tokenizer2.encode( "test" );
        EXPECT_FALSE( encoded.empty() );
    }

    TEST( BpeTokenizerTests, Encode_WithMLMTokens_IncludesMask )
    {
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 300 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::forMLM() );

        BpeVocabulary vocab = BpeVocabulary::train( "test corpus", cfg );
        
        auto mask_id = vocab.getSpecialTokenId( 'm' );
        ASSERT_TRUE( mask_id.has_value() );

        BpeTokenizer tokenizer( std::move( vocab ) );

        EXPECT_TRUE( tokenizer.isValidToken( static_cast<Mila::Dnn::Data::TokenId>( *mask_id ) ) );
        EXPECT_FALSE( tokenizer.tokenToString( static_cast<Mila::Dnn::Data::TokenId>( *mask_id ) ).empty() );
    }
}