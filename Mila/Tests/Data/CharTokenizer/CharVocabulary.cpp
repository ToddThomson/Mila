#include <gtest/gtest.h>

import Mila;

#include <sstream>
#include <filesystem>
#include <chrono>
#include <random>

namespace Data::CharTokenizer::Tests
{

    using namespace Mila::Data;
    namespace fs = std::filesystem;

    // Helper: create a unique temporary path for test artifacts.
    static fs::path make_temp_file( const std::string& stem = "char_vocab_test" )
    {
        auto dir = fs::temp_directory_path();
        auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::mt19937_64 rng( static_cast<unsigned long long>(now) );
        std::uniform_int_distribution<unsigned long long> dist;
        auto rand = dist( rng );

        return dir / (stem + "_" + std::to_string( now ) + "_" + std::to_string( rand ) + ".bin");
    }

    TEST( CharVocabularyTests, BuildFromText_BasicOrderingAndSize_WithCharVocabularyConfig )
    {
        // Arrange
        std::string corpus = "bca";

        CharVocabularyConfig cfg;
        CharSpecialTokens st = {};
        st.use_pad = false;
        st.use_unk = false;
        st.use_bos = false;
        st.use_eos = false;
        cfg.withSpecialTokens( st );

        std::istringstream stream( corpus );
        CharTrainer trainer( cfg );
        trainer.addCorpusFromStream( stream );

        // Act
        CharVocabulary vocab = trainer.train();

        // Assert
        EXPECT_EQ( vocab.getSize(), 3u );

        EXPECT_EQ( vocab.charToIndex( 'a' ), 0 );
        EXPECT_EQ( vocab.charToIndex( 'b' ), 1 );
        EXPECT_EQ( vocab.charToIndex( 'c' ), 2 );

        EXPECT_EQ( vocab.indexToChar( 0 ), 'a' );
        EXPECT_EQ( vocab.indexToChar( 1 ), 'b' );
        EXPECT_EQ( vocab.indexToChar( 2 ), 'c' );
    }

    TEST( CharVocabularyTests, BuildFromText_CRLFNormalization )
    {
        // Arrange
        std::string corpus = "first\r\nsecond\r\n";

        CharVocabularyConfig cfg;
        CharSpecialTokens st = {};
        st.use_pad = false;
        st.use_unk = false;
        st.use_bos = false;
        st.use_eos = false;
        cfg.withSpecialTokens( st );

        std::istringstream stream( corpus );
        CharTrainer trainer( cfg );
        trainer.addCorpusFromStream( stream );

        // Act
        CharVocabulary vocab = trainer.train();

        // '\n' should be present
        auto idx_newline = vocab.charToIndex( '\n' );

        EXPECT_GE( vocab.getSize(), 2u );
        EXPECT_EQ( vocab.indexToChar( idx_newline ), '\n' );
    }

    TEST( CharVocabularyTests, BuildFromText_WithCharSpecialTokens_ReservesTokens )
    {
        // Arrange
        std::string corpus = "ab";

        CharVocabularyConfig cfg;
        CharSpecialTokens st = {};
        st.use_pad = true;
        st.use_unk = true;
        st.use_bos = true;
        st.use_eos = true;
        cfg.withSpecialTokens( st );

        std::istringstream stream( corpus );
        CharTrainer trainer( cfg );
        trainer.addCorpusFromStream( stream );

        // Act
        CharVocabulary vocab = trainer.train();

        // Assert: reserved ids exist for pad and unk (padTokenId and unkTokenId >= 0)
        EXPECT_TRUE( vocab.hasSpecialTokens() );
        EXPECT_GE( vocab.padTokenId(), 0 );
        EXPECT_GE( vocab.unkTokenId(), 0 );

        // Reserved placeholder bytes should map to reserved ids
        EXPECT_EQ( vocab.charToIndex( '\0' ), vocab.padTokenId() );
        EXPECT_EQ( vocab.charToIndex( '\1' ), vocab.unkTokenId() );

        // Unknown token lookup: when UNK is enabled tokenToId returns unk id
        auto unk_lookup = vocab.tokenToId( "z" );
        ASSERT_TRUE( unk_lookup.has_value() );
        EXPECT_EQ( *unk_lookup, static_cast<uint32_t>(vocab.unkTokenId()) );
    }

    TEST( CharVocabularyTests, TokenToId_NoSpecialTokens_ReturnsNulloptForUnknown )
    {
        // Arrange
        std::string corpus = "ab";

        CharVocabularyConfig cfg;
        CharSpecialTokens st = {};
        st.use_pad = false;
        st.use_unk = false;
        st.use_bos = false;
        st.use_eos = false;
        cfg.withSpecialTokens( st );

        std::istringstream stream( corpus );
        CharTrainer trainer( cfg );
        trainer.addCorpusFromStream( stream );

        CharVocabulary vocab = trainer.train();

        // No UNK configured => tokenToId should return nullopt for missing token
        auto missing = vocab.tokenToId( "z" );
        EXPECT_FALSE( missing.has_value() );
    }

    TEST( CharVocabularyTests, SaveLoad_Roundtrip_PreservesState_WithSpecialTokens )
    {
        // Arrange
        std::string corpus = "xyz\n";

        CharVocabularyConfig cfg;
        CharSpecialTokens st = {};
        st.use_pad = true;
        st.use_unk = true;
        st.use_bos = true;
        st.use_eos = true;
        cfg.withSpecialTokens( st );

        std::istringstream stream( corpus );
        CharTrainer trainer( cfg );
        trainer.addCorpusFromStream( stream );

        CharVocabulary vocab = trainer.train();

        auto tmp = make_temp_file( "vocab_roundtrip" );
        if ( fs::exists( tmp ) ) fs::remove( tmp );

        // Act
        ASSERT_NO_THROW( vocab.save( tmp ) );
        ASSERT_TRUE( fs::exists( tmp ) );

        CharVocabulary loaded = CharVocabulary::load( tmp );

        // Assert basic properties preserved
        EXPECT_EQ( loaded.getSize(), vocab.getSize() );
        EXPECT_EQ( loaded.hasSpecialTokens(), vocab.hasSpecialTokens() );
        EXPECT_EQ( loaded.padTokenId(), vocab.padTokenId() );
        EXPECT_EQ( loaded.unkTokenId(), vocab.unkTokenId() );

        // Verify a few id->token round trips
        for ( uint32_t i = 0; i < std::min<uint32_t>( 10u, static_cast<uint32_t>( vocab.getSize() ) ); ++i )
        {
            auto a = vocab.idToToken( i );
            auto b = loaded.idToToken( i );
            EXPECT_EQ( a.has_value(), b.has_value() );
            if ( a && b )
            {
                EXPECT_EQ( *a, *b );
            }
        }

        // Cleanup
        fs::remove( tmp );
    }

    TEST( CharVocabularyTests, Load_InvalidPath_Throws )
    {
        auto bad = make_temp_file( "nonexistent_load" );
        if ( fs::exists( bad ) ) fs::remove( bad );

        EXPECT_THROW( CharVocabulary::load( bad ), std::runtime_error );
    }

    TEST( CharVocabularyTests, IndexToChar_OutOfRange_ReturnsQuestionMark )
    {
        // Arrange
        std::string corpus = "ab";

        CharVocabularyConfig cfg;
        CharSpecialTokens st = {};
        st.use_pad = false;
        st.use_unk = false;
        st.use_bos = false;
        st.use_eos = false;
        cfg.withSpecialTokens( st );

        std::istringstream stream( corpus );
        CharTrainer trainer( cfg );
        trainer.addCorpusFromStream( stream );

        CharVocabulary vocab = trainer.train();

        // indexToChar should return '?' for out-of-range index
        EXPECT_EQ( vocab.indexToChar( static_cast<int>(vocab.getSize() + 100) ), '?' );
    }

    TEST( CharVocabularyTests, CharToIndex_NoUnk_ReturnsZeroForUnknown )
    {
        // Arrange
        std::string corpus = "a";

        CharVocabularyConfig cfg;
        CharSpecialTokens st = {};
        st.use_pad = false;
        st.use_unk = false;
        st.use_bos = false;
        st.use_eos = false;
        cfg.withSpecialTokens( st );

        std::istringstream stream( corpus );
        CharTrainer trainer( cfg );
        trainer.addCorpusFromStream( stream );

        CharVocabulary vocab = trainer.train();

        // 'z' not present; without UNK configured, charToIndex should return 0
        int idx = vocab.charToIndex( 'z' );
        EXPECT_EQ( idx, 0 );
    }
}