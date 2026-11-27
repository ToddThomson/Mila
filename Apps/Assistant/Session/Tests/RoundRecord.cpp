/**
 * @file RoundRecord.cpp
 * @brief A conversation's round record: a line per agent event in order, a round's prompt whole or only what it adds,
 *        token ids exact, and the record gone with its conversation (ToolCalling.md 4.1).
 */

#include <gtest/gtest.h>
#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <vector>

import nlohmann.json;
import Mila;
import Assistant.RoundRecord;
import Assistant.ConversationStore;

namespace Mila::Tests::Apps::Assistant
{
    using namespace Mila::Assistant;
    using Mila::Agents::AgentEvent;
    using Kind = Mila::Agents::AgentEvent::Kind;

    namespace
    {
        class RoundRecordTest : public ::testing::Test
        {
        protected:

            void SetUp() override
            {
                static int counter = 0;
                folder_ = std::filesystem::temp_directory_path() / std::format( "mila_round_record_test_{}", counter++ );
                std::filesystem::remove_all( folder_ );
            }

            void TearDown() override
            {
                std::error_code ignored;
                std::filesystem::remove_all( folder_, ignored );
            }

            std::vector<nlohmann::json> lines( std::string_view id ) const
            {
                std::ifstream file( RoundRecord::pathFor( folder_, id ), std::ios::binary );
                std::vector<nlohmann::json> read;
                std::string line;

                while ( std::getline( file, line ) )
                {
                    read.push_back( nlohmann::json::parse( line ) );
                }

                return read;
            }

            static AgentEvent began( std::string_view prompt )
            {
                AgentEvent event{ Kind::RoundBegan };
                event.prompt = prompt;
                event.prompt_tokens = 12;
                event.sampling = Mila::Dnn::SamplingParams{ .temperature = 1.0f, .top_k = 64, .top_p = 0.95f };

                return event;
            }

            static AgentEvent token( std::int32_t id, std::string_view text )
            {
                AgentEvent event{ Kind::Token, text };
                event.token = id;

                return event;
            }

            std::filesystem::path folder_;
        };
    }

    TEST_F( RoundRecordTest, EachEventIsALineInTheOrderReported )
    {
        const Mila::Dnn::Conversation::ToolCall call{ "list_folder", R"({"path":"."})" };

        {
            RoundRecord record( RoundRecord::pathFor( folder_, "conversation-1" ) );
            ASSERT_TRUE( record.good() );

            record.record( began( "<|turn>user\nHi<turn|>\n<|turn>model\n" ), "turn-1", 1, "2026-11-27T17:00:00.000Z" );
            record.record( token( 100, "<|channel>" ), "turn-1", 1, "" );
            record.record( { Kind::ReasoningOpened }, "turn-1", 1, "" );
            record.record( token( 101, "Hello" ), "turn-1", 1, "" );
            record.record( { Kind::Answer, "Hello" }, "turn-1", 1, "" );

            AgentEvent called{ Kind::ToolCalled, "The project folder\n(empty)\n" };
            called.call = &call;
            called.decision = Mila::Agents::ToolDecision::AllowedByPerson;

            record.record( called, "turn-1", 1, "" );
            record.record( { Kind::RoundEnded }, "turn-1", 1, "" );
        }

        const std::vector<nlohmann::json> read = lines( "conversation-1" );

        ASSERT_EQ( read.size(), 7u );
        EXPECT_EQ( read[ 0 ][ "event" ], "round_began" );
        EXPECT_EQ( read[ 0 ][ "turn" ], "turn-1" );
        EXPECT_EQ( read[ 0 ][ "time" ], "2026-11-27T17:00:00.000Z" );
        EXPECT_EQ( read[ 0 ][ "prompt" ], "<|turn>user\nHi<turn|>\n<|turn>model\n" );
        EXPECT_EQ( read[ 0 ][ "prompt_tokens" ], 12 );
        EXPECT_EQ( read[ 0 ][ "sampling" ][ "top_k" ], 64 );

        // The marker the reader hides is in the token lines, with its id.
        EXPECT_EQ( read[ 1 ][ "event" ], "token" );
        EXPECT_EQ( read[ 1 ][ "id" ], 100 );
        EXPECT_EQ( read[ 1 ][ "text" ], "<|channel>" );
        EXPECT_EQ( read[ 2 ][ "event" ], "reasoning_opened" );
        EXPECT_EQ( read[ 4 ][ "event" ], "answer" );
        EXPECT_EQ( read[ 4 ][ "text" ], "Hello" );

        EXPECT_EQ( read[ 5 ][ "event" ], "tool_called" );
        EXPECT_EQ( read[ 5 ][ "name" ], "list_folder" );
        EXPECT_EQ( read[ 5 ][ "arguments" ], R"({"path":"."})" );
        EXPECT_EQ( read[ 5 ][ "decision" ], "allowed_by_person" );
        EXPECT_EQ( read[ 5 ][ "result" ], "The project folder\n(empty)\n" );
        EXPECT_EQ( read[ 6 ][ "event" ], "round_ended" );
        EXPECT_EQ( read[ 6 ][ "status" ], "stop" );

        for ( const nlohmann::json& line : read )
        {
            EXPECT_EQ( line[ "round" ], 1 );
            EXPECT_TRUE( line.contains( "ms" ) );
        }
    }

    TEST_F( RoundRecordTest, ARoundThatContinuesTheLastIsWrittenAsWhatItAdds )
    {
        const std::string first = "<|turn>user\nWeather?<turn|>\n<|turn>model\n";
        const std::string reply = "<|tool_call>call:get_weather{}<tool_call|>";
        const std::string continued = first + reply + "<|tool_response>sunny<tool_response|>";
        const std::string rendered = "<|turn>user\nWeather?<turn|>\n<|turn>model\nIt is sunny.<turn|>\n<|turn>user\nThanks<turn|>\n";

        {
            RoundRecord record( RoundRecord::pathFor( folder_, "conversation-2" ) );

            record.record( began( first ), "turn-1", 1, "t" );
            record.record( token( 7, reply ), "turn-1", 1, "" );
            record.record( began( continued ), "turn-1", 2, "t" );
            record.record( began( rendered ), "turn-2", 1, "t" );
        }

        const std::vector<nlohmann::json> read = lines( "conversation-2" );

        ASSERT_EQ( read.size(), 4u );
        EXPECT_EQ( read[ 0 ][ "prompt" ], first );
        EXPECT_FALSE( read[ 2 ].contains( "prompt" ) );
        EXPECT_EQ( read[ 2 ][ "prompt_added" ], "<|tool_response>sunny<tool_response|>" );

        // A rendering that is not the last round's prompt and reply is written whole.
        EXPECT_EQ( read[ 3 ][ "prompt" ], rendered );
        EXPECT_FALSE( read[ 3 ].contains( "prompt_added" ) );
    }

    TEST_F( RoundRecordTest, ATokenThatEndsPartWayThroughACharacterKeepsItsId )
    {
        {
            RoundRecord record( RoundRecord::pathFor( folder_, "conversation-3" ) );

            record.record( began( "p" ), "turn-1", 1, "t" );

            // The first two bytes of a three-byte character.
            record.record( token( 4242, "\xE2\x80" ), "turn-1", 1, "" );
        }

        const std::vector<nlohmann::json> read = lines( "conversation-3" );

        ASSERT_EQ( read.size(), 2u );
        EXPECT_EQ( read[ 1 ][ "id" ], 4242 );
    }

    TEST_F( RoundRecordTest, DeletingAConversationDeletesItsRecord )
    {
        const ConversationStore store( folder_ );
        const std::string id = store.newIdentifier();

        store.save( { { "id", id }, { "title", "Hi" } } );

        {
            RoundRecord record( RoundRecord::pathFor( folder_, id ) );
            record.record( began( "p" ), "turn-1", 1, "t" );
        }

        ASSERT_TRUE( std::filesystem::exists( RoundRecord::pathFor( folder_, id ) ) );

        // Not read as a conversation of its own.
        EXPECT_EQ( store.list().size(), 1u );

        EXPECT_TRUE( store.remove( id ) );
        EXPECT_FALSE( std::filesystem::exists( RoundRecord::pathFor( folder_, id ) ) );
    }
}
