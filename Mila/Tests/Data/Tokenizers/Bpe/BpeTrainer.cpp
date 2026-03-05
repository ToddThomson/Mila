#include <gtest/gtest.h>

import Mila;

#include <filesystem>
#include <sstream>
#include <chrono>
#include <random>
#include <fstream>

namespace Data::Tokenizers::BpeTrainer_Tests
{
    using namespace Mila::Data;
    namespace fs = std::filesystem;

    static fs::path make_temp_file( const std::string& stem = "bpe_trainer_test" )
    {
        auto dir = fs::temp_directory_path();
        auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::mt19937_64 rng( static_cast<unsigned long long>( now ) );
        std::uniform_int_distribution<unsigned long long> dist;
        auto rand = dist( rng );

        return dir / (stem + "_" + std::to_string( now ) + "_" + std::to_string( rand ) + ".bin");
    }

    TEST( BpeTrainerTests, Constructor_ValidConfig_Succeeds )
    {
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 1000 )
            .withByteLevel( true );

        EXPECT_NO_THROW( BpeTrainer trainer( cfg ) );
    }

    TEST( BpeTrainerTests, Constructor_InvalidConfig_Throws )
    {
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 0 );

        EXPECT_THROW( BpeTrainer trainer( cfg ), std::invalid_argument );
    }

    TEST( BpeTrainerTests, AddCorpusFromStream_ValidStream_AccumulatesText )
    {
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 300 )
            .withByteLevel( true );

        BpeTrainer trainer( cfg );

        std::istringstream stream1( "hello world" );
        trainer.addCorpusFromStream( stream1 );

        EXPECT_EQ( trainer.getCorpusSize(), 11u );

        std::istringstream stream2( " test" );
        trainer.addCorpusFromStream( stream2 );

        EXPECT_EQ( trainer.getCorpusSize(), 16u );
    }

    TEST( BpeTrainerTests, AddCorpusFromFile_ValidFile_AccumulatesText )
    {
        auto tmp = make_temp_file( "corpus" );
        {
            std::ofstream out( tmp );
            out << "test corpus content";
        }

        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 300 )
            .withByteLevel( true );

        BpeTrainer trainer( cfg );
        trainer.addCorpusFromFile( tmp );

        EXPECT_EQ( trainer.getCorpusSize(), 19u );

        fs::remove( tmp );
    }

    TEST( BpeTrainerTests, AddCorpusFromFile_InvalidPath_Throws )
    {
        auto bad = make_temp_file( "nonexistent" );
        if ( fs::exists( bad ) ) fs::remove( bad );

        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 300 )
            .withByteLevel( true );

        BpeTrainer trainer( cfg );

        EXPECT_THROW( trainer.addCorpusFromFile( bad ), std::runtime_error );
    }

    TEST( BpeTrainerTests, Train_EmptyCorpus_Throws )
    {
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 300 )
            .withByteLevel( true );

        BpeTrainer trainer( cfg );

        EXPECT_THROW( trainer.train(), std::runtime_error );
    }

    TEST( BpeTrainerTests, Train_ValidCorpus_ReturnsVocabulary )
    {
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 300 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::standard() );

        BpeTrainer trainer( cfg );

        std::istringstream stream( "test test test corpus corpus" );
        trainer.addCorpusFromStream( stream );

        BpeVocabulary vocab = trainer.train();

        EXPECT_GT( vocab.getSize(), 256u );
        EXPECT_FALSE( vocab.getMergeRules().empty() );
    }

    TEST( BpeTrainerTests, Train_ClearsCorpusAfterTraining )
    {
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 300 )
            .withByteLevel( true );

        BpeTrainer trainer( cfg );

        std::istringstream stream( "test corpus" );
        trainer.addCorpusFromStream( stream );

        EXPECT_GT( trainer.getCorpusSize(), 0u );

        BpeVocabulary vocab = trainer.train();

        EXPECT_EQ( trainer.getCorpusSize(), 0u );
    }

    TEST( BpeTrainerTests, Train_MultipleFiles_CombinesCorpus )
    {
        auto tmp1 = make_temp_file( "corpus1" );
        auto tmp2 = make_temp_file( "corpus2" );

        {
            std::ofstream out1( tmp1 );
            out1 << "aaa aaa aaa";
        }
        {
            std::ofstream out2( tmp2 );
            out2 << "bbb bbb bbb";
        }

        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 270 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::none() );

        BpeTrainer trainer( cfg );
        trainer.addCorpusFromFile( tmp1 );
        trainer.addCorpusFromFile( tmp2 );

        BpeVocabulary vocab = trainer.train();

        EXPECT_GT( vocab.getSize(), 256u );

        auto aa_tok = vocab.tokenToId( "aa" );
        auto bb_tok = vocab.tokenToId( "bb" );

        EXPECT_TRUE( aa_tok.has_value() || vocab.getMergeRules().size() > 0 );

        fs::remove( tmp1 );
        fs::remove( tmp2 );
    }

    TEST( BpeTrainerTests, GetCorpusSize_ReturnsAccumulatedSize )
    {
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 300 )
            .withByteLevel( true );

        BpeTrainer trainer( cfg );

        EXPECT_EQ( trainer.getCorpusSize(), 0u );

        std::istringstream stream( "12345" );
        trainer.addCorpusFromStream( stream );

        EXPECT_EQ( trainer.getCorpusSize(), 5u );
    }

    TEST( BpeTrainerTests, ClearCorpus_RemovesAccumulatedText )
    {
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 300 )
            .withByteLevel( true );

        BpeTrainer trainer( cfg );

        std::istringstream stream( "test corpus" );
        trainer.addCorpusFromStream( stream );

        EXPECT_GT( trainer.getCorpusSize(), 0u );

        trainer.clearCorpus();

        EXPECT_EQ( trainer.getCorpusSize(), 0u );
    }

    TEST( BpeTrainerTests, GetConfig_ReturnsStoredConfiguration )
    {
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 500 )
            .withMinFrequency( 3 )
            .withByteLevel( true )
            .withMaxMerges( 50 );

        BpeTrainer trainer( cfg );

        const auto& stored_cfg = trainer.getConfig();

        EXPECT_EQ( stored_cfg.getVocabSize(), 500u );
        EXPECT_EQ( stored_cfg.getMinFrequency(), 3u );
        EXPECT_TRUE( stored_cfg.isByteLevel() );
        EXPECT_EQ( stored_cfg.getMaxMerges(), 50u );
    }

    TEST( BpeTrainerTests, Train_ProducesVocabularyWithSameConfig )
    {
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 350 )
            .withMinFrequency( 2 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::forMLM() );

        BpeTrainer trainer( cfg );

        std::istringstream stream( "training corpus text" );
        trainer.addCorpusFromStream( stream );

        BpeVocabulary vocab = trainer.train();

        const auto& vocab_cfg = vocab.getConfig();

        EXPECT_EQ( vocab_cfg.getVocabSize(), cfg.getVocabSize() );
        EXPECT_EQ( vocab_cfg.getMinFrequency(), cfg.getMinFrequency() );
        EXPECT_EQ( vocab_cfg.isByteLevel(), cfg.isByteLevel() );
    }

    TEST( BpeTrainerTests, Train_LargeCorpus_SucceedsWithProgress )
    {
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 512 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::standard() );

        BpeTrainer trainer( cfg );

        std::string large_corpus;
        for ( int i = 0; i < 100; ++i )
        {
            large_corpus += "the quick brown fox jumps over the lazy dog ";
        }

        std::istringstream stream( large_corpus );
        trainer.addCorpusFromStream( stream );

        EXPECT_NO_THROW( BpeVocabulary vocab = trainer.train() );
    }

    TEST( BpeTrainerTests, Train_WithMinFrequency_RespectsThreshold )
    {
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 300 )
            .withMinFrequency( 10 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::none() );

        BpeTrainer trainer( cfg );

        std::istringstream stream( "a b c d e" );
        trainer.addCorpusFromStream( stream );

        BpeVocabulary vocab = trainer.train();

        EXPECT_EQ( vocab.getMergeRules().size(), 0u );
    }

    TEST( BpeTrainerTests, Train_WithMaxMerges_LimitsTraining )
    {
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 500 )
            .withMaxMerges( 2 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::none() );

        BpeTrainer trainer( cfg );

        std::istringstream stream( "aaaa bbbb cccc dddd" );
        trainer.addCorpusFromStream( stream );

        BpeVocabulary vocab = trainer.train();

        EXPECT_LE( vocab.getMergeRules().size(), 2u );
    }

    TEST( BpeTrainerTests, MultipleTrainCalls_RequireCorpusReaccumulation )
    {
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 300 )
            .withByteLevel( true );

        BpeTrainer trainer( cfg );

        std::istringstream stream1( "first training" );
        trainer.addCorpusFromStream( stream1 );

        BpeVocabulary vocab1 = trainer.train();

        EXPECT_THROW( trainer.train(), std::runtime_error );

        std::istringstream stream2( "second training" );
        trainer.addCorpusFromStream( stream2 );

        EXPECT_NO_THROW( BpeVocabulary vocab2 = trainer.train() );
    }

    TEST( BpeTrainerTests, AddCorpusFromStream_LargeBuffer_HandlesCorrectly )
    {
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 300 )
            .withByteLevel( true );

        BpeTrainer trainer( cfg );

        std::string large_text( 100000, 'x' );
        std::istringstream stream( large_text );

        EXPECT_NO_THROW( trainer.addCorpusFromStream( stream ) );
        EXPECT_EQ( trainer.getCorpusSize(), 100000u );
    }
}