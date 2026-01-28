#include <gtest/gtest.h>

import Mila;

#include <filesystem>
#include <sstream>
#include <fstream>
#include <chrono>
#include <random>

namespace Data::Tokenizers::BpeTokenizer_Tests
{
    using namespace Mila::Data;
    namespace fs = std::filesystem;

    // Helper: create a unique temporary path for test artifacts.
    static fs::path make_temp_file( const std::string& stem = "bpe_trainer_test" )
    {
        auto dir = fs::temp_directory_path();
        auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::mt19937_64 rng( static_cast<unsigned long long>(now) );
        std::uniform_int_distribution<unsigned long long> dist;
        auto rand = dist( rng );

        return dir / (stem + "_" + std::to_string( now ) + "_" + std::to_string( rand ) + ".txt");
    }

    TEST( BpeTrainerTests, Constructor_DefaultConfig_Succeeds )
    {
        // Default BpeTrainerConfig is valid per implementation; constructing should not throw.
        EXPECT_NO_THROW( BpeTrainer trainer{} );
    }

    TEST( BpeTrainerTests, Constructor_InvalidConfig_Throws )
    {
        // Create a config that is invalid: target vocab smaller than base size.
        BpeTrainerConfig cfg;
        cfg.withVocabSize( 100 )      // smaller than base 256 (+ specials)
            .withByteLevel( true )
            .withSpecialTokens( BpeSpecialTokens::none() );

        EXPECT_THROW( BpeTrainer trainer( cfg ), std::invalid_argument );
    }

    TEST( BpeTrainerTests, AddCorpusFromStream_Train_PerformsExpectedMerges )
    {
        // Arrange: corpus with frequent "a a" pairs
        std::string corpus = "aa aa aa";
        std::istringstream stream( corpus );

        BpeTrainerConfig cfg;
        cfg.withVocabSize( 257 )              // allow one merge beyond base bytes
            .withByteLevel( true )
            .withSpecialTokens( BpeSpecialTokens::none() );

        BpeTrainer trainer( cfg );

        // Act
        trainer.addCorpusFromStream( stream );
        BpeVocabulary vocab = trainer.train();

        // Assert: at least one merge performed and merged token "aa" present
        auto merges = vocab.getMergeRules();
        ASSERT_FALSE( merges.empty() );
        EXPECT_EQ( merges.front().first, std::string( 1, 'a' ) );
        EXPECT_EQ( merges.front().second, std::string( 1, 'a' ) );

        auto merged_id = vocab.tokenToId( std::string( "aa" ) );
        ASSERT_TRUE( merged_id.has_value() );
        EXPECT_EQ( vocab.idToToken( *merged_id ).value(), "aa" );
    }

    TEST( BpeTrainerTests, Train_ClearsCorpus_AfterFirstTrain )
    {
        // Arrange
        std::string corpus = "aa aa aa";
        std::istringstream stream( corpus );

        BpeTrainerConfig cfg;
        cfg.withVocabSize( 257 )
            .withByteLevel( true )
            .withSpecialTokens( BpeSpecialTokens::none() );

        BpeTrainer trainer( cfg );

        // Act: add corpus and train once (should perform merges)
        trainer.addCorpusFromStream( stream );
        BpeVocabulary first = trainer.train();

        // Train again without adding corpus: internal corpus_ should have been cleared,
        // so second training runs on empty input and should produce no merges.
        BpeVocabulary second = trainer.train();

        // Assert
        EXPECT_GT( first.getSize(), 256u );      // first run added merges
        EXPECT_EQ( second.getMergeRules().size(), 0u ); // second run no merges on empty corpus
    }

    TEST( BpeTrainerTests, AddCorpusFromFile_ValidAndInvalidPath )
    {
        // Arrange: write temp corpus file
        auto tmp = make_temp_file( "bpe_trainer_corpus" );
        {
            std::ofstream out( tmp, std::ios::binary );
            out << "aa aa aa\nbb bb";
        }
        ASSERT_TRUE( fs::exists( tmp ) );

        BpeTrainerConfig cfg;
        cfg.withVocabSize( 260 )
            .withByteLevel( true )
            .withSpecialTokens( BpeSpecialTokens::none() );

        BpeTrainer trainer( cfg );

        // Act: should not throw for valid file
        EXPECT_NO_THROW( trainer.addCorpusFromFile( tmp ) );
        BpeVocabulary vocab = trainer.train();

        // Assert: merges or tokens present
        EXPECT_GE( vocab.getSize(), 256u );

        // Cleanup
        fs::remove( tmp );

        // Invalid path should throw
        auto bad = make_temp_file( "bpe_trainer_nonexistent" );
        if ( fs::exists( bad ) ) fs::remove( bad );
        EXPECT_THROW( trainer.addCorpusFromFile( bad ), std::runtime_error );
    }

    TEST( BpeTrainerTests, Train_PassesSpecialTokensThroughConfig )
    {
        // Arrange: empty corpus but with special tokens configured
        BpeTrainerConfig cfg;
        cfg.withVocabSize( 260 )
            .withByteLevel( true )
            .withSpecialTokens( BpeSpecialTokens::standard() );

        BpeTrainer trainer( cfg );

        // Act
        BpeVocabulary vocab = trainer.train();

        // Assert: special token ids available and map back to configured strings
        auto pad_id = vocab.getSpecialTokenId( 'p' );
        auto unk_id = vocab.getSpecialTokenId( 'u' );
        auto bos_id = vocab.getSpecialTokenId( 'b' );
        auto eos_id = vocab.getSpecialTokenId( 'e' );

        ASSERT_TRUE( pad_id.has_value() );
        ASSERT_TRUE( unk_id.has_value() );
        ASSERT_TRUE( bos_id.has_value() );
        ASSERT_TRUE( eos_id.has_value() );

        EXPECT_EQ( vocab.idToToken( *pad_id ).value(), BpeSpecialTokens::standard().pad_token );
        EXPECT_EQ( vocab.idToToken( *unk_id ).value(), BpeSpecialTokens::standard().unk_token );
        EXPECT_EQ( vocab.idToToken( *bos_id ).value(), BpeSpecialTokens::standard().bos_token );
        EXPECT_EQ( vocab.idToToken( *eos_id ).value(), BpeSpecialTokens::standard().eos_token );
    }
}