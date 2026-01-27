#include <gtest/gtest.h>

import Mila;

#include <filesystem>
#include <sstream>
#include <chrono>
#include <random>
#include <algorithm>

namespace Data::BpeTokenizer::Tests
{

    using namespace Mila::Data;
    namespace fs = std::filesystem;

    // Helper: create a unique temporary path for test artifacts.
    static fs::path make_temp_file( const std::string& stem = "bpe_vocab_test" )
    {
        auto dir = fs::temp_directory_path();
        auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::mt19937_64 rng( static_cast<unsigned long long>(now) );
        std::uniform_int_distribution<unsigned long long> dist;
        auto rand = dist( rng );

        return dir / (stem + "_" + std::to_string( now ) + "_" + std::to_string( rand ) + ".bin");
    }

    TEST( BpeVocabularyTests, ByteLevelBaseVocabulary_TargetEqualsBase_NoSpecialTokens )
    {
        // Arrange
        BpeTrainerConfig cfg;
        cfg.withVocabSize( 256 );
        cfg.withByteLevel( true );
        cfg.withSpecialTokens( BpeSpecialTokens::none() );

        BpeVocabulary vocab;

        // Act
        size_t final_size = vocab.buildFromText( std::string(), cfg );

        // Assert: should initialize with 256 byte tokens and return immediately (no merges)
        EXPECT_EQ( final_size, 256u );
        EXPECT_EQ( vocab.getSize(), 256u );

        // Verify a few known byte tokens map to expected ids/strings
        auto tok0 = std::string( 1, static_cast<char>(0) );
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
        // Arrange
        BpeTrainerConfig cfg;
        cfg.withVocabSize( 260 );
        cfg.withByteLevel( true );
        cfg.withSpecialTokens( BpeSpecialTokens::standard() );

        BpeVocabulary vocab;

        // Act
        size_t final_size = vocab.buildFromText( std::string(), cfg );

        // Assert: base 256 + 4 core special tokens => at least 260 (or early return if target satisfied)
        EXPECT_GE( final_size, 256u );

        // special token ids should be available
        auto pad_id = vocab.getSpecialTokenId( 'p' );
        auto unk_id = vocab.getSpecialTokenId( 'u' );
        auto bos_id = vocab.getSpecialTokenId( 'b' );
        auto eos_id = vocab.getSpecialTokenId( 'e' );

        ASSERT_TRUE( pad_id.has_value() );
        ASSERT_TRUE( unk_id.has_value() );
        ASSERT_TRUE( bos_id.has_value() );
        ASSERT_TRUE( eos_id.has_value() );

        // Their id->token strings should match the configured special strings
        EXPECT_EQ( vocab.idToToken( *pad_id ).value(), BpeSpecialTokens::standard().pad_token );
        EXPECT_EQ( vocab.idToToken( *unk_id ).value(), BpeSpecialTokens::standard().unk_token );
        EXPECT_EQ( vocab.idToToken( *bos_id ).value(), BpeSpecialTokens::standard().bos_token );
        EXPECT_EQ( vocab.idToToken( *eos_id ).value(), BpeSpecialTokens::standard().eos_token );

        // tokenToId should return the special ids
        EXPECT_EQ( vocab.tokenToId( BpeSpecialTokens::standard().pad_token ).value(), *pad_id );
        EXPECT_EQ( vocab.tokenToId( BpeSpecialTokens::standard().unk_token ).value(), *unk_id );
    }

    TEST( BpeVocabularyTests, PerformsOneMerge_ForFrequentPair )
    {
        // Arrange: create a corpus with a frequent adjacent pair "a","a"
        std::string corpus = "aa aa aa";
        BpeTrainerConfig cfg;
        cfg.withVocabSize( 257 );           // ask for one extra token beyond base bytes
        cfg.withByteLevel( true );
        cfg.withSpecialTokens( BpeSpecialTokens::none() );
        // default min_frequency == 2 which is satisfied by three occurrences

        BpeVocabulary vocab;

        // Act
        size_t final_size = vocab.buildFromText( corpus, cfg );

        // Assert: final size increased by at least one (merge performed)
        EXPECT_GT( final_size, 256u );

        // There should be at least one merge rule and merged token present
        auto merges = vocab.getMergeRules();
        ASSERT_FALSE( merges.empty() );

        auto merged_pair = merges.front();
        EXPECT_EQ( merged_pair.first, std::string( 1, 'a' ) );
        EXPECT_EQ( merged_pair.second, std::string( 1, 'a' ) );

        // The merged string "aa" must be present in the vocabulary
        auto merged_tok = std::string( "aa" );
        auto merged_id = vocab.tokenToId( merged_tok );
        ASSERT_TRUE( merged_id.has_value() );
        EXPECT_EQ( vocab.idToToken( *merged_id ).value(), merged_tok );
    }

    TEST( BpeVocabularyTests, MinFrequencyPreventsMerges )
    {
        // Arrange: same corpus but require higher min_frequency so no merges happen
        std::string corpus = "aa aa aa";
        BpeTrainerConfig cfg;
        cfg.withVocabSize( 257 );
        cfg.withByteLevel( true );
        cfg.withSpecialTokens( BpeSpecialTokens::none() );
        cfg.withMinFrequency( 10 ); // larger than any pair count

        BpeVocabulary vocab;

        // Act
        size_t final_size = vocab.buildFromText( corpus, cfg );

        // Assert: no merges performed
        EXPECT_EQ( vocab.getMergeRules().size(), 0u );
        // size should equal base 256 (no merges and no specials)
        EXPECT_EQ( final_size, 256u );
    }

    TEST( BpeVocabularyTests, MaxMergesLimitsNumberOfMerges )
    {
        // Arrange: create corpus that could produce multiple merges ("a a a a b b b b")
        std::string corpus = "aaaa bbbb aaaa bbbb aaaa bbbb";
        BpeTrainerConfig cfg;
        cfg.withVocabSize( 260 );        // allow a few merges
        cfg.withByteLevel( true );
        cfg.withSpecialTokens( BpeSpecialTokens::none() );
        cfg.withMaxMerges( 1 );          // limit to a single merge
        cfg.withMinFrequency( 1 );       // allow low-frequency merges for the test

        BpeVocabulary vocab;

        // Act
        size_t final_size = vocab.buildFromText( corpus, cfg );

        // Assert: at most one merge rule recorded
        EXPECT_LE( vocab.getMergeRules().size(), 1u );
        // final size should reflect at most one added token
        EXPECT_LE( final_size, 257u );
    }

    TEST( BpeVocabularyTests, SaveLoad_Roundtrip_PreservesVocabAndMerges )
    {
        // Arrange
        std::string corpus = "aa aa aa bb bb";
        BpeTrainerConfig cfg;
        cfg.withVocabSize( 260 );
        cfg.withByteLevel( true );
        cfg.withSpecialTokens( BpeSpecialTokens::standard() );
        cfg.withMinFrequency( 1 );

        BpeVocabulary vocab;
        vocab.buildFromText( corpus, cfg );

        auto tmp = make_temp_file( "bpe_roundtrip" );
        if ( fs::exists( tmp ) ) fs::remove( tmp );

        // Act
        ASSERT_NO_THROW( vocab.save( tmp ) );
        ASSERT_TRUE( fs::exists( tmp ) );

        BpeVocabulary loaded = BpeVocabulary::load( tmp );

        // Assert sizes preserved
        EXPECT_EQ( loaded.getSize(), vocab.getSize() );

        // Special token ids preserved
        auto p1 = vocab.getSpecialTokenId( 'p' );
        auto p2 = loaded.getSpecialTokenId( 'p' );
        ASSERT_TRUE( p1.has_value() );
        ASSERT_TRUE( p2.has_value() );
        EXPECT_EQ( *p1, *p2 );
        EXPECT_EQ( vocab.idToToken( *p1 ), loaded.idToToken( *p2 ) );

        // Merge rules preserved (order and content)
        const auto& m1 = vocab.getMergeRules();
        const auto& m2 = loaded.getMergeRules();
        EXPECT_EQ( m1.size(), m2.size() );
        for ( size_t i = 0; i < m1.size(); ++i ) {
            EXPECT_EQ( m1[ i ].first, m2[ i ].first );
            EXPECT_EQ( m1[ i ].second, m2[ i ].second );
        }

        // Cleanup
        fs::remove( tmp );
    }

    TEST( BpeVocabularyTests, TokenToId_ReturnsUnkWhenConfigured_OrNulloptOtherwise )
    {
        // Arrange: with special tokens
        BpeTrainerConfig cfg1;
        cfg1.withVocabSize( 260 );
        cfg1.withByteLevel( true );
        cfg1.withSpecialTokens( BpeSpecialTokens::standard() );

        BpeVocabulary v1;
        v1.buildFromText( std::string(), cfg1 );

        // Unknown token should map to UNK id when present
        auto unk_id = v1.getSpecialTokenId( 'u' );
        ASSERT_TRUE( unk_id.has_value() );
        auto maybe = v1.tokenToId( "<|this-does-not-exist|>" );
        ASSERT_TRUE( maybe.has_value() );
        EXPECT_EQ( *maybe, *unk_id );

        // Arrange: without special tokens
        BpeTrainerConfig cfg2;
        cfg2.withVocabSize( 256 );
        cfg2.withByteLevel( true );
        cfg2.withSpecialTokens( BpeSpecialTokens::none() );

        BpeVocabulary v2;
        v2.buildFromText( std::string(), cfg2 );

        // Unknown token should return nullopt when UNK not configured
        auto none = v2.tokenToId( "<|nope|>" );
        EXPECT_FALSE( none.has_value() );
    }

    TEST( BpeVocabularyTests, IdToToken_OutOfRange_ReturnsNullopt )
    {
        BpeTrainerConfig cfg;
        cfg.withVocabSize( 256 );
        cfg.withByteLevel( true );
        cfg.withSpecialTokens( BpeSpecialTokens::none() );

        BpeVocabulary vocab;
        vocab.buildFromText( std::string(), cfg );

        auto out = vocab.idToToken( static_cast<uint32_t>(vocab.getSize() + 100) );
        EXPECT_FALSE( out.has_value() );
    }

    TEST( BpeVocabularyTests, Load_InvalidPath_Throws )
    {
        auto bad = make_temp_file( "bpe_nonexistent" );
        if ( fs::exists( bad ) ) fs::remove( bad );

        EXPECT_THROW( BpeVocabulary::load( bad ), std::runtime_error );
    }
}