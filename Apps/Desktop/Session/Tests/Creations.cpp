/**
 * @file Creations.cpp
 * @brief What a chat made is read from its stored calls, and a written file is compared with the disk by its record
 *        (AssistantInterface.md 4.9 and 8).
 */

#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <string>
#include <system_error>

import nlohmann.json;
import Desktop.Creations;
import Desktop.ConversationSummary;

namespace Mila::Tests::Apps::Desktop
{
    using namespace Mila::Desktop;

    namespace
    {
        class CreationsTest : public ::testing::Test
        {
        protected:

            void SetUp() override
            {
                static std::atomic<int> counter{ 0 };

                folder_ = std::filesystem::temp_directory_path() / std::format( "mila_creations_test_{}", counter++ );
                std::filesystem::remove_all( folder_ );
                std::filesystem::create_directories( folder_ );
            }

            void TearDown() override
            {
                std::error_code ignored;
                std::filesystem::remove_all( folder_, ignored );
            }

            std::filesystem::path write( const std::string& name, const std::string& text ) const
            {
                const std::filesystem::path file = folder_ / name;
                std::ofstream( file, std::ios::binary ) << text;

                return file;
            }

            std::string folderText() const
            {
                const std::u8string text = folder_.u8string();

                return std::string( text.begin(), text.end() );
            }

            std::filesystem::path folder_;
        };

        nlohmann::json call( const std::string& name, const std::string& path, const std::string& time,
            nlohmann::json exposes = true )
        {
            return { { "call_id", std::format( "call-{}", time ) }, { "name", name }, { "summary", std::format( "{} {}", name, path ) },
                { "arguments", nlohmann::json( { { "path", path } } ).dump() }, { "time", time }, { "decision", nullptr },
                { "result", "done" }, { "is_error", false }, { "exposes", std::move( exposes ) } };
        }

        nlohmann::json reply( nlohmann::json calls, const std::string& content = "Done." )
        {
            return { { "role", "assistant" }, { "content", content }, { "tool_calls", std::move( calls ) } };
        }

        nlohmann::json chat( nlohmann::json messages, nlohmann::json folder = nullptr )
        {
            return { { "id", "conversation-1" }, { "title", "Notes" }, { "project", "project-1" }, { "folder", std::move( folder ) },
                { "messages", std::move( messages ) } };
        }
    }

    TEST_F( CreationsTest, AWrittenFileIsTheSameUntilItsBytesChange )
    {
        const std::filesystem::path file = write( "notes.txt", "first" );
        const nlohmann::json record = fileRecord( file );

        ASSERT_TRUE( record.is_object() );
        EXPECT_EQ( record[ "size" ], 5 );
        EXPECT_EQ( record[ "sha256" ].get<std::string>().size(), 64u );
        EXPECT_EQ( diskState( record ), "same" );

        write( "notes.txt", "other" );

        // The same length, so only a new time or the bytes tell them apart; a time the file system rounds away is
        // put a second on to be sure the time differs.
        std::filesystem::last_write_time( file, std::filesystem::last_write_time( file ) + std::chrono::seconds( 1 ) );

        EXPECT_EQ( diskState( record ), "changed" );

        write( "notes.txt", "longer text" );

        EXPECT_EQ( diskState( record ), "changed" );

        std::filesystem::remove( file );

        EXPECT_EQ( diskState( record ), "missing" );
    }

    TEST_F( CreationsTest, AFileOnlyTouchedIsTheSame )
    {
        const std::filesystem::path file = write( "notes.txt", "first" );
        const nlohmann::json record = fileRecord( file );

        std::filesystem::last_write_time( file, std::filesystem::last_write_time( file ) + std::chrono::seconds( 5 ) );

        EXPECT_EQ( diskState( record ), "same" );
    }

    TEST_F( CreationsTest, AWriteStoredBeforeRecordsHasNoState )
    {
        EXPECT_TRUE( diskState( nullptr ).is_null() );
        EXPECT_TRUE( fileRecord( folder_ / "absent.txt" ).is_null() );
    }

    TEST_F( CreationsTest, AFileCreationSaysEveryWriteAndWhatWasReadBeforeTheFirst )
    {
        nlohmann::json first = call( "write_file", "notes.txt", "2026-11-19T10:00:02.000Z" );
        first[ "model" ] = "gemma-4-12b";
        first[ "step" ] = "Write the notes";
        first[ "written" ] = { { "location", "elsewhere/notes.txt" }, { "sha256", "aa" }, { "size", 1 }, { "modified", 1 } };

        nlohmann::json denied = call( "read_file", "secret.txt", "2026-11-19T10:00:01.500Z" );
        denied[ "decision" ] = "deny";

        const nlohmann::json conversation = chat( nlohmann::json::array( {
            reply( nlohmann::json::array( { call( "read_file", "source.txt", "2026-11-19T10:00:00.000Z" ),
                call( "propose_plan", "", "2026-11-19T10:00:01.000Z", false ), denied, first } ) ),
            reply( nlohmann::json::array( { call( "read_file", "later.txt", "2026-11-19T10:01:00.000Z" ),
                call( "edit_file", "notes.txt", "2026-11-19T10:01:01.000Z" ), call( "edit_file", "other.txt", "2026-11-19T10:01:02.000Z" ) } ) ) } ) );

        const std::optional<nlohmann::json> creation = fileCreation( conversation, "notes.txt" );

        ASSERT_TRUE( creation.has_value() );
        EXPECT_EQ( ( *creation )[ "kind" ], "file" );
        EXPECT_EQ( ( *creation )[ "conversation_id" ], "conversation-1" );
        EXPECT_EQ( ( *creation )[ "project" ], "project-1" );
        EXPECT_EQ( ( *creation )[ "location" ], "elsewhere/notes.txt" );

        const nlohmann::json& writes = ( *creation )[ "writes" ];
        ASSERT_EQ( writes.size(), 2u );
        EXPECT_EQ( writes[ 0 ][ "model" ], "gemma-4-12b" );
        EXPECT_EQ( writes[ 0 ][ "step" ], "Write the notes" );
        EXPECT_EQ( writes[ 0 ][ "sha256" ], "aa" );
        EXPECT_TRUE( writes[ 1 ][ "model" ].is_null() );
        EXPECT_EQ( ( *creation )[ "made" ], writes[ 0 ] );

        // Only the read before the first write: not the plan, not the declined read, not what came after.
        const nlohmann::json& reads = ( *creation )[ "read_before" ];
        ASSERT_EQ( reads.size(), 1u );
        EXPECT_EQ( reads[ 0 ][ "tool" ], "read_file" );
        EXPECT_EQ( reads[ 0 ][ "what" ], "read_file source.txt" );
        EXPECT_EQ( ( *creation )[ "read_before_total" ], 1 );
        EXPECT_EQ( ( *creation )[ "exposed" ], true );

        EXPECT_FALSE( fileCreation( conversation, "never.txt" ).has_value() );
    }

    TEST_F( CreationsTest, AWriteStoredBeforeRecordsIsFoundInTheChatsFolder )
    {
        const nlohmann::json conversation = chat( nlohmann::json::array( {
            reply( nlohmann::json::array( { call( "write_file", "notes.txt", "2026-11-19T10:00:00.000Z" ) } ) ) } ), folderText() );

        const std::optional<nlohmann::json> creation = fileCreation( conversation, "notes.txt" );

        ASSERT_TRUE( creation.has_value() );

        const std::string location = ( *creation )[ "location" ].get<std::string>();
        EXPECT_EQ( std::filesystem::path( std::u8string( location.begin(), location.end() ) ), folder_ / "notes.txt" );
        EXPECT_TRUE( ( *creation )[ "writes" ][ 0 ][ "sha256" ].is_null() );
        EXPECT_EQ( ( *creation )[ "exposed" ], false );
    }

    TEST_F( CreationsTest, OnlyTheLatestReadsAreKeptAndAllAreCounted )
    {
        nlohmann::json calls = nlohmann::json::array();

        for ( int index = 0; index < 25; ++index )
        {
            calls.push_back( call( "read_file", std::format( "file{}.txt", index ), std::format( "2026-11-19T10:00:{:02}.000Z", index ), false ) );
        }

        calls.push_back( call( "write_file", "notes.txt", "2026-11-19T10:01:00.000Z" ) );

        const std::optional<nlohmann::json> creation = fileCreation( chat( nlohmann::json::array( { reply( calls ) } ) ), "notes.txt" );

        ASSERT_TRUE( creation.has_value() );
        ASSERT_EQ( ( *creation )[ "read_before" ].size(), 20u );
        EXPECT_EQ( ( *creation )[ "read_before" ][ 0 ][ "what" ], "read_file file5.txt" );
        EXPECT_EQ( ( *creation )[ "read_before_total" ], 25 );
        EXPECT_EQ( ( *creation )[ "exposed" ], false );
    }

    TEST_F( CreationsTest, AKeptReplyIsACreationAndOneNotKeptIsNot )
    {
        nlohmann::json kept = reply( nlohmann::json::array( { call( "read_file", "source.txt", "2026-11-19T10:00:00.000Z" ) } ), "The answer." );
        kept[ "kept" ] = "2026-11-19T10:05:00.000Z";
        kept[ "model" ] = "gemma-4-12b";

        const nlohmann::json conversation = chat( nlohmann::json::array( {
            { { "role", "user" }, { "content", "Question" } }, kept, reply( nullptr, "Not kept." ) } ) );

        const std::optional<nlohmann::json> creation = replyCreation( conversation, 1 );

        ASSERT_TRUE( creation.has_value() );
        EXPECT_EQ( ( *creation )[ "kind" ], "reply" );
        EXPECT_EQ( ( *creation )[ "text" ], "The answer." );
        EXPECT_EQ( ( *creation )[ "model" ], "gemma-4-12b" );
        EXPECT_EQ( ( *creation )[ "read_before" ].size(), 1u );

        EXPECT_FALSE( replyCreation( conversation, 0 ).has_value() );
        EXPECT_FALSE( replyCreation( conversation, 2 ).has_value() );
        EXPECT_FALSE( replyCreation( conversation, 9 ).has_value() );

        const nlohmann::json summary = summarizeConversation( conversation );
        ASSERT_EQ( summary[ "kept" ].size(), 1u );
        EXPECT_EQ( summary[ "kept" ][ 0 ][ "message" ], 1 );
        EXPECT_EQ( summary[ "kept" ][ 0 ][ "start" ], "The answer." );
    }

    TEST_F( CreationsTest, TheSummaryCarriesTheLastWritesTimeAndRecord )
    {
        nlohmann::json second = call( "edit_file", "notes.txt", "2026-11-19T10:01:00.000Z" );
        second[ "written" ] = { { "location", "notes.txt" }, { "sha256", "bb" }, { "size", 2 }, { "modified", 2 } };

        const nlohmann::json conversation = chat( nlohmann::json::array( { reply( nlohmann::json::array( {
            call( "write_file", "notes.txt", "2026-11-19T10:00:00.000Z" ), second } ) ) } ) );

        const nlohmann::json wrote = summarizeConversation( conversation )[ "wrote" ];

        ASSERT_EQ( wrote.size(), 1u );
        EXPECT_EQ( wrote[ 0 ][ "time" ], "2026-11-19T10:01:00.000Z" );
        EXPECT_EQ( wrote[ 0 ][ "written" ][ "sha256" ], "bb" );
    }

    TEST_F( CreationsTest, TextIsShownAndAnythingElseIsNot )
    {
        const nlohmann::json markdown = showableContent( write( "Notes.MD", "# Title\n\nText \xE2\x80\x94 more." ) );
        EXPECT_EQ( markdown[ "content_kind" ], "markdown" );
        EXPECT_EQ( markdown[ "content" ], "# Title\n\nText \xE2\x80\x94 more." );

        EXPECT_EQ( showableContent( write( "a.py", "print(1)\n" ) )[ "content_kind" ], "text" );
        EXPECT_TRUE( showableContent( write( "binary.bin", std::string( "a\0b", 3 ) ) )[ "content" ].is_null() );
        EXPECT_TRUE( showableContent( write( "latin.txt", "caf\xE9" ) )[ "content" ].is_null() );
        EXPECT_TRUE( showableContent( write( "overlong.txt", "\xC0\xAF" ) )[ "content" ].is_null() );
        EXPECT_TRUE( showableContent( write( "large.txt", std::string( kMostBytesShown + 1, 'a' ) ) )[ "content" ].is_null() );
        EXPECT_TRUE( showableContent( folder_ / "absent.txt" )[ "content" ].is_null() );
    }
}
