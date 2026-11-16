/**
 * @file ConversationSummary.cpp
 * @brief The home screen reads a stored conversation's plan, last reply and written files from what the store holds
 *        (AssistantInterface.md 4.2).
 */

#include <gtest/gtest.h>
#include <string>

import nlohmann.json;
import Desktop.ConversationSummary;

namespace Mila::Tests::Apps::Desktop
{
    using namespace Mila::Desktop;

    namespace
    {
        nlohmann::json call( const std::string& name, const std::string& path, bool failed = false )
        {
            const nlohmann::json arguments = { { "path", path }, { "content", "x" } };

            return { { "call_id", "c" }, { "name", name }, { "summary", "Writing " + path },
                { "arguments", arguments.dump() }, { "decision", nullptr }, { "result", "Wrote it." },
                { "is_error", failed } };
        }

        nlohmann::json reply( nlohmann::json calls, const std::string& finish )
        {
            return { { "role", "assistant" }, { "content", "Done." }, { "tool_calls", std::move( calls ) },
                { "stats", { { "finish", finish } } } };
        }
    }

    TEST( ConversationSummaryTest, APlanSaysHowFarItCameAndWhereItStopped )
    {
        const nlohmann::json conversation = { { "id", "conversation-1" }, { "plan", nlohmann::json::array( {
            { { "text", "Read the tests" }, { "status", "done" } },
            { { "text", "Fix the parser" }, { "status", "skipped" } },
            { { "text", "Add a case" }, { "status", "pending" } },
            { { "text", "Run them" }, { "status", "pending" } } } ) } };

        const nlohmann::json summary = summarizeConversation( conversation );

        EXPECT_EQ( summary[ "progress" ][ "steps" ], 4 );
        EXPECT_EQ( summary[ "progress" ][ "finished" ], 2 );
        EXPECT_EQ( summary[ "progress" ][ "current" ], "Add a case" );
    }

    TEST( ConversationSummaryTest, AStepInProgressIsWhereThePlanIs )
    {
        const nlohmann::json conversation = { { "id", "conversation-1" }, { "plan", nlohmann::json::array( {
            { { "text", "One" }, { "status", "pending" } },
            { { "text", "Two" }, { "status", "in_progress" } } } ) } };

        EXPECT_EQ( summarizeConversation( conversation )[ "progress" ][ "current" ], "Two" );
    }

    TEST( ConversationSummaryTest, NoPlanAndNoReplyAreNull )
    {
        const nlohmann::json summary = summarizeConversation( { { "id", "conversation-1" }, { "plan", nullptr } } );

        EXPECT_TRUE( summary[ "progress" ].is_null() );
        EXPECT_TRUE( summary[ "finish" ].is_null() );
        EXPECT_TRUE( summary[ "wrote" ].empty() );
    }

    TEST( ConversationSummaryTest, TheLastReplySaysHowItEnded )
    {
        const nlohmann::json conversation = { { "id", "conversation-1" }, { "messages", nlohmann::json::array( {
            reply( nullptr, "stop" ),
            { { "role", "user" }, { "content", "More" } },
            reply( nullptr, "length" ) } ) } };

        EXPECT_EQ( summarizeConversation( conversation )[ "finish" ], "length" );
    }

    TEST( ConversationSummaryTest, TheLastReplyStartsOnOneLine )
    {
        nlohmann::json earlier = reply( nullptr, "stop" );
        nlohmann::json last = reply( nullptr, "stop" );
        last[ "content" ] = "  Keys and values\n\nnever change,\tso they are kept.  ";

        const nlohmann::json conversation = { { "id", "conversation-1" }, { "messages", nlohmann::json::array( {
            earlier, { { "role", "user" }, { "content", "Why?" } }, last } ) } };

        EXPECT_EQ( summarizeConversation( conversation )[ "reply_start" ], "Keys and values never change, so they are kept." );
    }

    TEST( ConversationSummaryTest, ALongReplyIsCutAtACharacterNeverInsideOne )
    {
        // 159 ASCII bytes, then a three-byte character the 160-byte cut would land inside.
        nlohmann::json last = reply( nullptr, "stop" );
        last[ "content" ] = std::string( 159, 'a' ) + "\xE2\x80\x94" + "more";

        const nlohmann::json conversation = { { "id", "conversation-1" }, { "messages", nlohmann::json::array( { last } ) } };
        const std::string start = summarizeConversation( conversation )[ "reply_start" ].get<std::string>();

        EXPECT_EQ( start, std::string( 159, 'a' ) );
    }

    TEST( ConversationSummaryTest, NoReplyOrAnEmptyAnswerHasNoStart )
    {
        nlohmann::json empty = reply( nullptr, "cancelled" );
        empty[ "content" ] = " \n ";

        const nlohmann::json asked = { { "id", "conversation-1" }, { "messages", nlohmann::json::array( {
            { { "role", "user" }, { "content", "Hello" } } } ) } };
        const nlohmann::json stopped = { { "id", "conversation-2" }, { "messages", nlohmann::json::array( { empty } ) } };

        EXPECT_TRUE( summarizeConversation( asked )[ "reply_start" ].is_null() );
        EXPECT_TRUE( summarizeConversation( stopped )[ "reply_start" ].is_null() );
    }

    TEST( ConversationSummaryTest, FilesWrittenAreListedOnceTheLatestLastAndFailuresLeftOut )
    {
        const nlohmann::json conversation = { { "id", "conversation-1" }, { "messages", nlohmann::json::array( {
            reply( nlohmann::json::array( { call( "write_file", "src/a.py" ), call( "read_file", "src/b.py" ) } ), "stop" ),
            reply( nlohmann::json::array( { call( "write_file", "src/c.py", true ), call( "edit_file", "README.md" ),
                call( "edit_file", "src/a.py" ) } ), "stop" ) } ) } };

        const nlohmann::json wrote = summarizeConversation( conversation )[ "wrote" ];

        ASSERT_EQ( wrote.size(), 2u );
        EXPECT_EQ( wrote[ 0 ][ "path" ], "README.md" );
        EXPECT_EQ( wrote[ 1 ][ "path" ], "src/a.py" );
        EXPECT_EQ( wrote[ 1 ][ "tool" ], "edit_file" );
    }

    TEST( ConversationSummaryTest, ACallThatDidNotRunOrHasNoPathIsNoFileWritten )
    {
        nlohmann::json stopped = call( "write_file", "a.txt" );
        stopped[ "result" ] = nullptr;

        nlohmann::json unreadable = call( "write_file", "b.txt" );
        unreadable[ "arguments" ] = "{ not json";

        const nlohmann::json conversation = { { "id", "conversation-1" }, { "messages", nlohmann::json::array( {
            reply( nlohmann::json::array( { stopped, unreadable } ), "cancelled" ) } ) } };

        EXPECT_TRUE( summarizeConversation( conversation )[ "wrote" ].empty() );
    }
}
