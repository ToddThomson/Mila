#include <gtest/gtest.h>

import Mila;

#include <filesystem>
#include <sstream>
#include <fstream>
#include <chrono>
#include <random>
#include <algorithm>

namespace Data::Tokenizers::BpeTokenizer_Tests
{
    using namespace Mila::Data;
    namespace fs = std::filesystem;

    static fs::path make_temp_file( const std::string& stem = "bpe_vocab_test" )
    {
        auto dir = fs::temp_directory_path();
        auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::mt19937_64 rng( static_cast<unsigned long long>( now ) );
        std::uniform_int_distribution<unsigned long long> dist;
        auto rand = dist( rng );

        return dir / (stem + "_" + std::to_string( now ) + "_" + std::to_string( rand ) + ".bin");
    }

    TEST( BpeVocabularyTests, ByteLevelBaseVocabulary_TargetEqualsBase_NoSpecialTokens )
    {
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 256 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::none() );

        BpeVocabulary vocab = BpeVocabulary::train( std::string(), cfg );

        EXPECT_EQ( vocab.getSize(), 256u );

        auto tok0 = std::string( 1, static_cast<char>( 0 ) );
        auto id0 = vocab.tokenToId( tok0 );
        ASSERT_TRUE( id0.has_value() );
        EXPECT_EQ( *id0, 0u );
        auto t0 = vocab.idToToken( 0 );
        ASSERT_TRUE( t0.has_value() );
        EXPECT_EQ( *t0, tok0 );

        auto tokA = std::string( 1, 'A' );
        auto idA = vocab.tokenToId( tokA );
        ASSERT_TRUE( idA.has_value() );
        EXPECT_EQ( vocab.idToToken( *idA ).value(), tokA );
    }

    TEST( BpeVocabularyTests, AddsSpecialTokens_WhenEnabled )
    {
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 260 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::standard() );

        BpeVocabulary vocab = BpeVocabulary::train( std::string(), cfg );

        EXPECT_GE( vocab.getSize(), 256u );

        auto pad_id = vocab.getSpecialTokenId( 'p' );
        auto unk_id = vocab.getSpecialTokenId( 'u' );
        auto bos_id = vocab.getSpecialTokenId( 'b' );
        auto eos_id = vocab.getSpecialTokenId( 'e' );

        ASSERT_TRUE( pad_id.has_value() );
        ASSERT_TRUE( unk_id.has_value() );
        ASSERT_TRUE( bos_id.has_value() );
        ASSERT_TRUE( eos_id.has_value() );

        EXPECT_EQ( vocab.idToToken( *pad_id ).value(), SpecialTokens::standard().pad_token );
        EXPECT_EQ( vocab.idToToken( *unk_id ).value(), SpecialTokens::standard().unk_token );
        EXPECT_EQ( vocab.idToToken( *bos_id ).value(), SpecialTokens::standard().bos_token );
        EXPECT_EQ( vocab.idToToken( *eos_id ).value(), SpecialTokens::standard().eos_token );

        EXPECT_EQ( vocab.tokenToId( SpecialTokens::standard().pad_token ).value(), *pad_id );
        EXPECT_EQ( vocab.tokenToId( SpecialTokens::standard().unk_token ).value(), *unk_id );
    }

    TEST( BpeVocabularyTests, PerformsOneMerge_ForFrequentPair )
    {
        std::string corpus = "aa aa aa";
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 257 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::none() );

        BpeVocabulary vocab = BpeVocabulary::train( corpus, cfg );

        EXPECT_GT( vocab.getSize(), 256u );

        auto merges = vocab.getMergeRules();
        ASSERT_FALSE( merges.empty() );

        auto merged_pair = merges.front();
        EXPECT_EQ( merged_pair.first, std::string( 1, 'a' ) );
        EXPECT_EQ( merged_pair.second, std::string( 1, 'a' ) );

        auto merged_tok = std::string( "aa" );
        auto merged_id = vocab.tokenToId( merged_tok );
        ASSERT_TRUE( merged_id.has_value() );
        EXPECT_EQ( vocab.idToToken( *merged_id ).value(), merged_tok );
    }

    TEST( BpeVocabularyTests, MinFrequencyPreventsMerges )
    {
        std::string corpus = "aa aa aa";
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 257 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::none() )
            .withMinFrequency( 10 );

        BpeVocabulary vocab = BpeVocabulary::train( corpus, cfg );

        EXPECT_EQ( vocab.getMergeRules().size(), 0u );
        EXPECT_EQ( vocab.getSize(), 256u );
    }

    TEST( BpeVocabularyTests, MaxMergesLimitsNumberOfMerges )
    {
        std::string corpus = "aaaa bbbb aaaa bbbb aaaa bbbb";
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 260 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::none() )
            .withMaxMerges( 1 )
            .withMinFrequency( 1 );

        BpeVocabulary vocab = BpeVocabulary::train( corpus, cfg );

        EXPECT_LE( vocab.getMergeRules().size(), 1u );
        EXPECT_LE( vocab.getSize(), 257u );
    }

    TEST( BpeVocabularyTests, SaveLoad_Roundtrip_PreservesVocabAndMerges )
    {
        std::string corpus = "aa aa aa bb bb";
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 260 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::standard() )
            .withMinFrequency( 1 );

        BpeVocabulary vocab = BpeVocabulary::train( corpus, cfg );

        auto tmp = make_temp_file( "bpe_roundtrip" );
        if ( fs::exists( tmp ) ) fs::remove( tmp );

        ASSERT_NO_THROW( vocab.save( tmp ) );
        ASSERT_TRUE( fs::exists( tmp ) );

        BpeVocabulary loaded = BpeVocabulary::load( tmp );

        EXPECT_EQ( loaded.getSize(), vocab.getSize() );

        auto p1 = vocab.getSpecialTokenId( 'p' );
        auto p2 = loaded.getSpecialTokenId( 'p' );
        ASSERT_TRUE( p1.has_value() );
        ASSERT_TRUE( p2.has_value() );
        EXPECT_EQ( *p1, *p2 );
        EXPECT_EQ( vocab.idToToken( *p1 ), loaded.idToToken( *p2 ) );

        const auto& m1 = vocab.getMergeRules();
        const auto& m2 = loaded.getMergeRules();
        EXPECT_EQ( m1.size(), m2.size() );
        for ( size_t i = 0; i < m1.size(); ++i )
        {
            EXPECT_EQ( m1[ i ].first, m2[ i ].first );
            EXPECT_EQ( m1[ i ].second, m2[ i ].second );
        }

        fs::remove( tmp );
    }

    TEST( BpeVocabularyTests, SaveLoad_Roundtrip_PreservesConfiguration )
    {
        std::string corpus = "test corpus for config preservation";
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 300 )
            .withByteLevel( true )
            .withMinFrequency( 2 )
            .withMaxMerges( 10 )
            .withSpecialTokens( SpecialTokens::standard() );

        BpeVocabulary vocab = BpeVocabulary::train( corpus, cfg );

        auto tmp = make_temp_file( "bpe_config_roundtrip" );
        if ( fs::exists( tmp ) ) fs::remove( tmp );

        vocab.save( tmp );
        BpeVocabulary loaded = BpeVocabulary::load( tmp );

        const auto& orig_cfg = vocab.getConfig();
        const auto& loaded_cfg = loaded.getConfig();

        EXPECT_EQ( loaded_cfg.getVocabSize(), orig_cfg.getVocabSize() );
        EXPECT_EQ( loaded_cfg.getMinFrequency(), orig_cfg.getMinFrequency() );
        EXPECT_EQ( loaded_cfg.isByteLevel(), orig_cfg.isByteLevel() );
        EXPECT_EQ( loaded_cfg.getMaxMerges(), orig_cfg.getMaxMerges() );

        fs::remove( tmp );
    }

    TEST( BpeVocabularyTests, TokenToId_ReturnsUnkWhenConfigured_OrNulloptOtherwise )
    {
        BpeVocabularyConfig cfg1 = BpeVocabularyConfig()
            .withVocabSize( 260 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::standard() );

        BpeVocabulary v1 = BpeVocabulary::train( std::string(), cfg1 );

        auto unk_id = v1.getSpecialTokenId( 'u' );
        ASSERT_TRUE( unk_id.has_value() );
        auto maybe = v1.tokenToId( "<|this-does-not-exist|>" );
        ASSERT_TRUE( maybe.has_value() );
        EXPECT_EQ( *maybe, *unk_id );

        BpeVocabularyConfig cfg2 = BpeVocabularyConfig()
            .withVocabSize( 256 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::none() );

        BpeVocabulary v2 = BpeVocabulary::train( std::string(), cfg2 );

        auto none = v2.tokenToId( "<|nope|>" );
        EXPECT_FALSE( none.has_value() );
    }

    TEST( BpeVocabularyTests, IdToToken_OutOfRange_ReturnsNullopt )
    {
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 256 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::none() );

        BpeVocabulary vocab = BpeVocabulary::train( std::string(), cfg );

        auto out = vocab.idToToken( static_cast<uint32_t>( vocab.getSize() + 100 ) );
        EXPECT_FALSE( out.has_value() );
    }

    TEST( BpeVocabularyTests, Load_InvalidPath_Throws )
    {
        auto bad = make_temp_file( "bpe_nonexistent" );
        if ( fs::exists( bad ) ) fs::remove( bad );

        EXPECT_THROW( BpeVocabulary::load( bad ), std::runtime_error );
    }

    TEST( BpeVocabularyTests, TrainFromFile_ValidCorpus_Succeeds )
    {
        auto tmp_corpus = make_temp_file( "test_corpus" );
        {
            std::ofstream out( tmp_corpus );
            out << "test test test corpus corpus";
        }

        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 262 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::standard() );

        BpeVocabulary vocab = BpeVocabulary::trainFromFile( tmp_corpus, cfg );

        EXPECT_GT( vocab.getSize(), 256u );
        EXPECT_FALSE( vocab.getMergeRules().empty() );

        fs::remove( tmp_corpus );
    }

    TEST( BpeVocabularyTests, TrainFromFile_InvalidPath_Throws )
    {
        auto bad = make_temp_file( "nonexistent_corpus" );
        if ( fs::exists( bad ) ) fs::remove( bad );

        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 260 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::standard() );

        EXPECT_THROW( BpeVocabulary::trainFromFile( bad, cfg ), std::runtime_error );
    }

    TEST( BpeVocabularyTests, GetConfig_ReturnsStoredConfiguration )
    {
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 500 )
            .withMinFrequency( 3 )
            .withByteLevel( true )
            .withMaxMerges( 50 )
            .withSpecialTokens( SpecialTokens::forMLM() );

        BpeVocabulary vocab = BpeVocabulary::train( "test corpus", cfg );

        const auto& stored_cfg = vocab.getConfig();

        EXPECT_EQ( stored_cfg.getVocabSize(), 500u );
        EXPECT_EQ( stored_cfg.getMinFrequency(), 3u );
        EXPECT_TRUE( stored_cfg.isByteLevel() );
        EXPECT_EQ( stored_cfg.getMaxMerges(), 50u );
    }

    TEST( BpeVocabularyTests, SpecialTokens_MLM_IncludesMaskToken )
    {
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 265 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::forMLM() );

        BpeVocabulary vocab = BpeVocabulary::train( std::string(), cfg );

        auto mask_id = vocab.getSpecialTokenId( 'm' );
        ASSERT_TRUE( mask_id.has_value() );

        auto mask_token = vocab.idToToken( *mask_id );
        ASSERT_TRUE( mask_token.has_value() );
        EXPECT_EQ( *mask_token, SpecialTokens::forMLM().mask_token );
    }

    TEST( BpeVocabularyTests, SpecialTokens_Classification_IncludesSepAndCls )
    {
        BpeVocabularyConfig cfg = BpeVocabularyConfig()
            .withVocabSize( 265 )
            .withByteLevel( true )
            .withSpecialTokens( SpecialTokens::forClassification() );

        BpeVocabulary vocab = BpeVocabulary::train( std::string(), cfg );

        auto sep_id = vocab.getSpecialTokenId( 's' );
        auto cls_id = vocab.getSpecialTokenId( 'c' );

        ASSERT_TRUE( sep_id.has_value() );
        ASSERT_TRUE( cls_id.has_value() );

        EXPECT_EQ( vocab.idToToken( *sep_id ).value(), SpecialTokens::forClassification().sep_token );
        EXPECT_EQ( vocab.idToToken( *cls_id ).value(), SpecialTokens::forClassification().cls_token );
    }
}