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

    // Helper: create a unique temporary path for test artifacts.
    static fs::path make_temp_file( const std::string& stem = "bpe_tokenizer_test" )
    {
        auto dir = fs::temp_directory_path();
        auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::mt19937_64 rng( static_cast<unsigned long long>(now) );
        std::uniform_int_distribution<unsigned long long> dist;
        auto rand = dist( rng );

        return dir / (stem + "_" + std::to_string( now ) + "_" + std::to_string( rand ) + ".bin");
    }

    TEST( BpeTokenizerTests, EncodeDecode_Roundtrip_ByteLevel )
    {
        // Arrange
        std::string corpus = "hello world";
        std::istringstream stream( corpus );

        BpeTrainerConfig cfg;
        cfg.withVocabSize( 300 )
            .withByteLevel( true )
            .withSpecialTokens( BpeSpecialTokens::none() );

        BpeTrainer trainer( cfg );
        trainer.addCorpusFromStream( stream );
        BpeVocabulary vocab = trainer.train();

        BpeTokenizer tokenizer( vocab );

        std::string input = "hello";
        // Act
        auto encoded = tokenizer.encode( input );
        auto decoded = tokenizer.decode( std::span<const Mila::Dnn::Data::TokenId>( encoded.data(), encoded.size() ) );

        // Assert: byte-level vocabulary covers all bytes, decode should equal original
        EXPECT_EQ( decoded, input );
        EXPECT_EQ( tokenizer.getVocabSize(), vocab.getSize() );

        // tokenToString and isValidToken checks
        if ( !encoded.empty() ) {
            auto firstId = encoded.front();
            EXPECT_EQ( tokenizer.tokenToString( firstId ), vocab.idToToken( firstId ).value() );
            EXPECT_TRUE( tokenizer.isValidToken( firstId ) );
        }

        // invalid id should be invalid
        EXPECT_FALSE( tokenizer.isValidToken( static_cast<Mila::Dnn::Data::TokenId>(vocab.getSize() + 100) ) );
    }

    TEST( BpeTokenizerTests, Encode_NonByteLevel_MissingChar_EmitsFallbackZero )
    {
        // Arrange: non-byte-level vocab built from limited corpus => missing characters possible
        std::string corpus = "ab";
        std::istringstream stream( corpus );

        BpeTrainerConfig cfg;
        cfg.withVocabSize( 10 )
            .withByteLevel( false )                 // character-level base vocabulary
            .withSpecialTokens( BpeSpecialTokens::none() );

        BpeTrainer trainer( cfg );
        trainer.addCorpusFromStream( stream );
        BpeVocabulary vocab = trainer.train();

        BpeTokenizer tokenizer( vocab );

        // Input contains 'a' (present) and 'c' (absent)
        std::string input = "ac";

        // Act
        auto encoded = tokenizer.encode( input );

        // Assert: first token corresponds to 'a', second is fallback 0u because 'c' missing
        ASSERT_GE( encoded.size(), 2u );
        auto id_a = vocab.tokenToId( std::string( 1, 'a' ) );
        ASSERT_TRUE( id_a.has_value() );
        EXPECT_EQ( encoded[ 0 ], *id_a );

        // second token: tokenToId('c') should be nullopt; encode should have emitted 0u fallback
        auto id_c = vocab.tokenToId( std::string( 1, 'c' ) );
        EXPECT_FALSE( id_c.has_value() );
        EXPECT_EQ( encoded[ 1 ], static_cast<Mila::Dnn::Data::TokenId>(0u) );
    }

    TEST( BpeTokenizerTests, Decode_InvalidId_ProducesQuestionMark )
    {
        // Arrange
        BpeTrainerConfig cfg;
        cfg.withVocabSize( 256 )
            .withByteLevel( true )
            .withSpecialTokens( BpeSpecialTokens::none() );

        BpeTrainer trainer( cfg );
        
        // train on empty corpus -> base byte vocabulary
        BpeVocabulary vocab = trainer.train();

        BpeTokenizer tokenizer( vocab );

        // Prepare tokens containing an out-of-range id
        std::vector<Mila::Dnn::Data::TokenId> tokens;
        tokens.push_back( 0u ); // valid
        tokens.push_back( static_cast<Mila::Dnn::Data::TokenId>(vocab.getSize() + 5) ); // invalid

        // Act
        std::string decoded = tokenizer.decode( tokens );

        // Assert: decoded contains valid token string for first id and '?' for invalid id
        ASSERT_GE( decoded.size(), 2u );
        EXPECT_NE( decoded[ 0 ], '?' );
        EXPECT_EQ( decoded[ 1 ], '?' );
    }

    TEST( BpeTokenizerTests, LoadFromFile_And_EncodeMatchesInMemoryTokenizer )
    {
        // Arrange: build a vocabulary and save to disk, then load tokenizer via convenience method
        std::string corpus = "aa aa aa bb bb";
        std::istringstream stream( corpus );

        BpeTrainerConfig cfg;
        cfg.withVocabSize( 300 )
            .withByteLevel( true )
            .withSpecialTokens( BpeSpecialTokens::standard() )
            .withMinFrequency( 1 );

        BpeTrainer trainer( cfg );
        trainer.addCorpusFromStream( stream );
        BpeVocabulary vocab = trainer.train();

        auto tmp = make_temp_file( "bpe_tok_load" );
        if ( fs::exists( tmp ) ) fs::remove( tmp );

        ASSERT_NO_THROW( vocab.save( tmp ) );
        ASSERT_TRUE( fs::exists( tmp ) );

        // Act
        BpeTokenizer fromMem( vocab );
        BpeTokenizer fromFile = BpeTokenizer::load( tmp );

        std::string text = "aabb";
        auto enc_mem = fromMem.encode( text );
        auto enc_file = fromFile.encode( text );

        // Assert: both tokenizers should encode to same ids (vocab equivalence)
        EXPECT_EQ( enc_mem, enc_file );

        // Cleanup
        fs::remove( tmp );
    }

    TEST( BpeTokenizerTests, TokenToString_IsValidToken_GetVocabSize )
    {
        // Arrange
        BpeTrainerConfig cfg;
        cfg.withVocabSize( 256 )
            .withByteLevel( true )
            .withSpecialTokens( BpeSpecialTokens::none() );

        BpeTrainer trainer( cfg );
        BpeVocabulary vocab = trainer.train();

        BpeTokenizer tokenizer( vocab );

        // pick some ids to check
        uint32_t vid = 1;
        auto maybeTok = vocab.idToToken( vid );
        ASSERT_TRUE( maybeTok.has_value() );

        // Act / Assert
        EXPECT_EQ( tokenizer.tokenToString( static_cast<Mila::Dnn::Data::TokenId>(vid) ), *maybeTok );
        EXPECT_TRUE( tokenizer.isValidToken( static_cast<Mila::Dnn::Data::TokenId>(vid) ) );
        EXPECT_EQ( tokenizer.getVocabSize(), vocab.getSize() );
    }

    TEST( BpeTokenizerTests, LoadGpt2_FromTestDataDir )
    {
        // The tests CMakeLists defines TEST_DATA_DIR as a compile-time string via
        // target_compile_definitions(... TEST_DATA_DIR="${TEST_DATA_DIR}")
        fs::path dataDir = TEST_DATA_DIR;
        fs::path tokenizerPath = dataDir / "models" / "gpt2" / "gpt2_tokenizer.bin";

        if ( !fs::exists( tokenizerPath ) ) {
            GTEST_SKIP() << "GPT-2 tokenizer binary not present at: " << tokenizerPath.string();
        }

        // Act / Assert: ensure loading doesn't throw and produces a non-empty vocabulary
        ASSERT_NO_THROW({
            BpeTokenizer tokenizer = BpeTokenizer::loadGpt2( tokenizerPath );
            EXPECT_GT( tokenizer.getVocabSize(), 0u );
        });
    }
}