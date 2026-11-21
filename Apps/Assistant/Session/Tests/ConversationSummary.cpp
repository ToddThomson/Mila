/**
 * @file ConversationSummary.cpp
 * @brief The home screen reads a stored conversation's plan, last reply and written files from what the store holds
 *        (AssistantInterface.md 4.2).
 */

#include <gtest/gtest.h>
#include <string>

import nlohmann.json;
import Assistant.ConversationSummary;

namespace Mila::Tests::Apps::Assistant
{
    using namespace Mila::Assistant;

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

    TEST( ConversationSummaryTest, WhatLeftTheComputerIsListedOldestFirstAndADeclinedCallIsNot )
    {
        const auto sending = []( const std::string& name, const nlohmann::json& arguments, const nlohmann::json& decision,
            const std::string& time, bool failed )
            {
                return nlohmann::json{ { "call_id", "c" }, { "name", name }, { "arguments", arguments.dump() },
                    { "decision", decision }, { "result", "text" }, { "is_error", failed }, { "time", time } };
            };

        const nlohmann::json calls = nlohmann::json::array( {
            sending( "read_web_page", { { "url", "https://example.test/a?b=1" } }, "allow_once", "2026-11-18T10:00:00.000Z", true ),
            sending( "web_search", { { "query", "my words" } }, "deny", "2026-11-18T10:01:00.000Z", false ),
            sending( "web_search", { { "query", "other words" } }, "allow_once", "2026-11-18T10:02:00.000Z", false ),
            sending( "run_command", { { "command", "dir" } }, "allow_once", "2026-11-18T10:03:00.000Z", false ),
            sending( "run_command", { { "command", "npm i" }, { "network", true } }, "allow_once", "2026-11-18T10:04:00.000Z", false ),
            sending( "read_file", { { "path", "a.txt" } }, "allow_once", "2026-11-18T10:05:00.000Z", false ) } );

        const nlohmann::json conversation = { { "messages", nlohmann::json::array( { reply( calls, "stop" ) } ) },
            { "links_opened", nlohmann::json::array( { { { "url", "https://me@link.test:8/x" }, { "time", "2026-11-18T10:01:30.000Z" } } } ) } };

        const nlohmann::json sent = summarizeConversation( conversation )[ "sent" ];

        ASSERT_EQ( sent.size(), 4u );
        EXPECT_EQ( sent[ 0 ][ "kind" ], "page" );
        EXPECT_EQ( sent[ 0 ][ "what" ], "https://example.test/a?b=1" );
        EXPECT_EQ( sent[ 0 ][ "where" ], "example.test" );
        EXPECT_EQ( sent[ 1 ][ "kind" ], "link" );
        EXPECT_EQ( sent[ 1 ][ "where" ], "link.test:8" );
        EXPECT_EQ( sent[ 2 ][ "what" ], "other words" );
        EXPECT_EQ( sent[ 3 ][ "kind" ], "command" );
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
