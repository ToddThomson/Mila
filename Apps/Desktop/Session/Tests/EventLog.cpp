/**
 * @file EventLog.cpp
 * @brief The workspace's events are appended and read back, and a deletion forgets the name of what went while the
 *        event keeps its kind and time (AssistantInterface.md 8, *Deleting is designed in*).
 */

#include <gtest/gtest.h>
#include <atomic>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <system_error>

import nlohmann.json;
import Desktop.EventLog;

namespace Mila::Tests::Apps::Desktop
{
    using namespace Mila::Desktop;

    namespace
    {
        class EventLogTest : public ::testing::Test
        {
        protected:

            void SetUp() override
            {
                static std::atomic<int> counter{ 0 };

                folder_ = std::filesystem::temp_directory_path() / std::format( "mila_event_log_test_{}", counter++ );
                std::filesystem::remove_all( folder_ );
            }

            void TearDown() override
            {
                std::error_code ignored;
                std::filesystem::remove_all( folder_, ignored );
            }

            EventLog log() const
            {
                return EventLog( folder_ / "history.jsonl" );
            }

            std::filesystem::path folder_;
        };

        const nlohmann::json kMade = { { "time", "2026-11-19T10:00:00.000Z" }, { "who", "you" }, { "kind", "project.made" },
            { "project", "project-a" }, { "name", "Kyoto in November" } };

        const nlohmann::json kFolder = { { "time", "2026-11-19T10:01:00.000Z" }, { "who", "you" },
            { "kind", "project.folder_added" }, { "project", "project-a" }, { "path", "D:\\Trips" }, { "access", "read" } };

        const nlohmann::json kMoved = { { "time", "2026-11-19T10:02:00.000Z" }, { "who", "you" }, { "kind", "chat.moved" },
            { "conversation", "conversation-c" }, { "name", "Plan a three-day trip" }, { "project", "project-a" },
            { "from", "project-b" } };
    }

    TEST_F( EventLogTest, EventsAreReadInTheOrderTheyHappened )
    {
        log().append( kMade );
        log().append( kMoved );

        const auto events = log().read();

        ASSERT_EQ( events.size(), 2u );
        EXPECT_EQ( events[ 0 ], kMade );
        EXPECT_EQ( events[ 1 ], kMoved );
    }

    TEST_F( EventLogTest, ALineCutShortIsSkippedAndTheRestAreRead )
    {
        log().append( kMade );
        std::ofstream( log().path(), std::ios::binary | std::ios::app ) << "{\"time\":\"2026-11";

        const auto events = log().read();

        ASSERT_EQ( events.size(), 1u );
        EXPECT_EQ( events[ 0 ], kMade );
    }

    TEST_F( EventLogTest, ADeletedChatLeavesItsEventsWithoutItsIdOrTitle )
    {
        log().append( kMoved );
        log().forget( "conversation-c" );

        const nlohmann::json event = log().read().at( 0 );

        EXPECT_EQ( event[ "kind" ], "chat.moved" );
        EXPECT_EQ( event[ "time" ], kMoved[ "time" ] );
        EXPECT_FALSE( event.contains( "conversation" ) );
        EXPECT_FALSE( event.contains( "name" ) );
        EXPECT_EQ( event[ "project" ], "project-a" );
    }

    TEST_F( EventLogTest, ADeletedProjectLeavesItsEventsWithoutItsNameOrFolder )
    {
        log().append( kMade );
        log().append( kFolder );
        log().append( kMoved );
        log().forget( "project-a" );

        const auto events = log().read();

        ASSERT_EQ( events.size(), 3u );
        EXPECT_FALSE( events[ 0 ].contains( "project" ) );
        EXPECT_FALSE( events[ 0 ].contains( "name" ) );
        EXPECT_FALSE( events[ 1 ].contains( "path" ) );
        EXPECT_EQ( events[ 1 ][ "kind" ], "project.folder_added" );

        // The chat that moved is still there to name; only the project it moved into is forgotten.
        EXPECT_FALSE( events[ 2 ].contains( "project" ) );
        EXPECT_EQ( events[ 2 ][ "name" ], "Plan a three-day trip" );
        EXPECT_EQ( events[ 2 ][ "from" ], "project-b" );
    }

    TEST_F( EventLogTest, AProjectAChatCameFromIsForgottenAsWell )
    {
        log().append( kMoved );
        log().forget( "project-b" );

        const nlohmann::json event = log().read().at( 0 );

        EXPECT_FALSE( event.contains( "from" ) );
        EXPECT_EQ( event[ "project" ], "project-a" );
        EXPECT_EQ( event[ "name" ], "Plan a three-day trip" );
    }

    TEST_F( EventLogTest, ForgettingWhatNoEventNamesLeavesTheFileAsItWas )
    {
        log().append( kMade );

        const auto written = std::filesystem::last_write_time( log().path() );

        log().forget( "conversation-other" );

        EXPECT_EQ( std::filesystem::last_write_time( log().path() ), written );
        EXPECT_EQ( log().read().at( 0 ), kMade );
    }
}
