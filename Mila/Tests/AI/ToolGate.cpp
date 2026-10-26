/**
 * @file ToolGate.cpp
 * @brief Every outcome of the gate a tool call passes: refused before anyone is asked, decided by a rule, asked,
 *        stopped or out of time, and the question built from the tool as registered (ToolCalling.md 3.2 and 3.3).
 */

#include <gtest/gtest.h>
#include <chrono>
#include <condition_variable>
#include <format>
#include <functional>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

import Mila;

namespace Mila::Tests::AI
{
    using namespace Mila::AI;

    namespace
    {
        constexpr const char* kParameters =
            R"({"type":"object","properties":{"path":{"type":"string"},"summary":{"type":"string"}},"required":["path"]})";

        constexpr const char* kArguments = R"({"path":"D:/Project/notes.md"})";

        /// Waits until asked to stop, or a minute passes; the minute is far past any time a test sets.
        std::string waitForStop( std::stop_token stop )
        {
            std::mutex mutex;
            std::condition_variable_any never;
            std::unique_lock<std::mutex> lock( mutex );

            never.wait_for( lock, stop, std::chrono::minutes( 1 ), [] { return false; } );

            return stop.stop_requested() ? "partial" : "finished";
        }

        class ToolGateTest : public ::testing::Test
        {
        protected:

            void SetUp() override
            {
                gate_.add( deleteTool() );
            }

            Tool deleteTool()
            {
                Tool tool{
                    "delete_file",
                    "Delete one file. Use only when the user asks for a file to be removed.",
                    kParameters,
                    [this]( std::string_view arguments, std::stop_token stop )
                    {
                        ran_.emplace_back( arguments );

                        if ( body_ )
                        {
                            return body_( stop );
                        }

                        return std::string( "deleted" );
                    } };

                tool.source = "files";
                tool.summary = "Deletes a file in the project.";
                tool.needs.writes_files = true;
                tool.needs.areas = { "D:/Project" };

                return tool;
            }

            /// Asks by answering with each answer in turn, keeping every question.
            ToolGate::Asker answering( std::vector<ToolAnswer> answers )
            {
                return [this, answers = std::move( answers ), next = std::size_t{ 0 }]( const ToolQuestion& question ) mutable
                    {
                        asked_.push_back( { std::string( question.name ), std::string( question.source ),
                            std::string( question.summary ), std::string( question.arguments ), question.needs } );

                        return answers.at( next++ );
                    };
            }

            ToolOutcome answer( const ToolGate::Asker& ask, std::string_view arguments = kArguments,
                std::stop_token halted = {}, const AgentLimits& limits = {} )
            {
                return gate_.answer( "delete_file", arguments, ask, halted, limits );
            }

            struct Asked
            {
                std::string name;
                std::string source;
                std::string summary;
                std::string arguments;
                ToolNeeds needs;
            };

            ToolGate gate_;
            std::vector<std::string> ran_;
            std::vector<Asked> asked_;
            std::function<std::string( std::stop_token )> body_;
        };
    }

    // ================================================================
    // Refused before anyone is asked
    // ================================================================

    TEST_F( ToolGateTest, ACallOfAToolNotOfferedIsNeverAskedAboutAndTheModelReadsTheTools )
    {
        const ToolOutcome outcome = gate_.answer( "get_capital", R"({"country":"France"})", answering( { ToolAnswer::Once } ), {} );

        EXPECT_EQ( outcome.decision, ToolDecision::NotOffered );
        EXPECT_TRUE( outcome.is_error );
        EXPECT_FALSE( outcome.ran );
        EXPECT_EQ( outcome.result, "Error: there is no tool named 'get_capital'. The tools are: delete_file." );
        EXPECT_TRUE( asked_.empty() );
    }

    TEST_F( ToolGateTest, ArgumentsThatDoNotMatchAreNeverAskedAboutAndNeverRun )
    {
        // The rules would allow it, so only the check stands between the call and the tool.
        gate_.setRules( ToolRules{}.add( { "*", ToolPermission::Allow } ) );

        const ToolOutcome outcome = answer( answering( { ToolAnswer::Once } ), "{}" );

        EXPECT_EQ( outcome.decision, ToolDecision::InvalidArguments );
        EXPECT_TRUE( outcome.is_error );
        EXPECT_FALSE( outcome.ran );
        EXPECT_EQ( outcome.result, "Error: the arguments do not match the parameters of 'delete_file': 'path' is required." );
        EXPECT_TRUE( asked_.empty() );
        EXPECT_TRUE( ran_.empty() );
    }

    // ================================================================
    // Decided by the rules
    // ================================================================

    TEST_F( ToolGateTest, ACallARuleAllowsRunsWithoutAsking )
    {
        gate_.setRules( ToolRules{}.add(
            { "delete_file", ToolPermission::Allow, { { "/path", ArgumentCondition::Test::Matches, { "D:/Project/*" } } } } ) );

        const ToolOutcome outcome = answer( {} );

        EXPECT_EQ( outcome.decision, ToolDecision::AllowedByRule );
        EXPECT_FALSE( outcome.is_error );
        EXPECT_TRUE( outcome.ran );
        EXPECT_EQ( outcome.result, "deleted" );
        ASSERT_EQ( ran_.size(), 1u );
        EXPECT_EQ( ran_.front(), kArguments );
        EXPECT_TRUE( asked_.empty() );

        // Outside the folder the rule does not apply, and with no one to ask nothing runs.
        EXPECT_EQ( answer( {}, R"({"path":"C:/Windows/win.ini"})" ).decision, ToolDecision::NoOneToAsk );
        EXPECT_EQ( ran_.size(), 1u );
    }

    TEST_F( ToolGateTest, ACallARuleDeniesNeverRunsEvenWhenThePersonAllowedTheToolForTheConversation )
    {
        EXPECT_EQ( answer( answering( { ToolAnswer::ForConversation } ) ).decision,
            ToolDecision::AllowedByPersonForConversation );

        gate_.setRules( ToolRules{}.add(
            { "delete_file", ToolPermission::Deny, { { "/path", ArgumentCondition::Test::Matches, { "*.md" } } } } ) );

        const ToolOutcome outcome = answer( answering( { ToolAnswer::Once } ) );

        EXPECT_EQ( outcome.decision, ToolDecision::DeniedByRule );
        EXPECT_TRUE( outcome.is_error );
        EXPECT_FALSE( outcome.ran );
        EXPECT_EQ( outcome.result, "The user's rules do not allow this tool call." );
        EXPECT_EQ( asked_.size(), 1u ) << "a denied call was asked about";
        EXPECT_EQ( ran_.size(), 1u );
    }

    // ================================================================
    // Asked
    // ================================================================

    TEST_F( ToolGateTest, ACallAllowedOnceRunsAndTheNextIsAskedAgain )
    {
        const auto ask = answering( { ToolAnswer::Once, ToolAnswer::Once } );

        EXPECT_EQ( answer( ask ).decision, ToolDecision::AllowedByPerson );
        EXPECT_EQ( answer( ask ).decision, ToolDecision::AllowedByPerson );
        EXPECT_EQ( asked_.size(), 2u );
        EXPECT_EQ( ran_.size(), 2u );
    }

    TEST_F( ToolGateTest, AToolAllowedForTheConversationIsNotAskedAboutAgainUntilItEnds )
    {
        const auto ask = answering( { ToolAnswer::ForConversation, ToolAnswer::Deny } );

        EXPECT_EQ( answer( ask ).decision, ToolDecision::AllowedByPersonForConversation );
        EXPECT_EQ( answer( ask, R"({"path":"D:/Project/other.md"})" ).decision, ToolDecision::AllowedForConversation );
        EXPECT_EQ( asked_.size(), 1u );

        gate_.newConversation();

        EXPECT_EQ( answer( ask ).decision, ToolDecision::DeclinedByPerson );
        EXPECT_EQ( asked_.size(), 2u );
        EXPECT_EQ( ran_.size(), 2u );
    }

    TEST_F( ToolGateTest, ADeclinedCallDoesNotRunAndTheModelReadsSo )
    {
        const ToolOutcome outcome = answer( answering( { ToolAnswer::Deny } ) );

        EXPECT_EQ( outcome.decision, ToolDecision::DeclinedByPerson );
        EXPECT_TRUE( outcome.is_error );
        EXPECT_FALSE( outcome.ran );
        EXPECT_EQ( outcome.result, "The user declined to run this tool call." );
        EXPECT_TRUE( ran_.empty() );
    }

    TEST_F( ToolGateTest, WithNoOneToAskACallTheRulesAskAboutDoesNotRun )
    {
        const ToolOutcome outcome = answer( {} );

        EXPECT_EQ( outcome.decision, ToolDecision::NoOneToAsk );
        EXPECT_TRUE( outcome.is_error );
        EXPECT_EQ( outcome.result, "The user declined to run this tool call." );
        EXPECT_TRUE( ran_.empty() );
    }

    TEST_F( ToolGateTest, TheQuestionIsBuiltFromTheToolAsRegisteredNeverFromTheModelsText )
    {
        // The model describes its own call as harmless; the person is shown what the tool does.
        const char* arguments = R"({"path":"D:/Project/notes.md","summary":"Only reads the weather for Paris."})";

        answer( answering( { ToolAnswer::Deny } ), arguments );

        ASSERT_EQ( asked_.size(), 1u );
        EXPECT_EQ( asked_.front().name, "delete_file" );
        EXPECT_EQ( asked_.front().source, "files" );
        EXPECT_EQ( asked_.front().summary, "Deletes a file in the project." );
        EXPECT_EQ( asked_.front().arguments, arguments ) << "every argument is shown in full";
        EXPECT_TRUE( asked_.front().needs.writes_files );
        EXPECT_FALSE( asked_.front().needs.reads_files );
        EXPECT_EQ( asked_.front().needs.areas, std::vector<std::string>{ "D:/Project" } );
    }

    // ================================================================
    // Running
    // ================================================================

    TEST_F( ToolGateTest, ACallThatRanIsRecorded )
    {
        body_ = []( std::stop_token )
            {
                std::this_thread::sleep_for( std::chrono::milliseconds( 20 ) );

                return std::string( "twelve bytes" );
            };

        const ToolOutcome outcome = answer( answering( { ToolAnswer::Once } ) );

        EXPECT_GE( outcome.duration, std::chrono::milliseconds( 20 ) );
        EXPECT_EQ( outcome.returned_bytes, 12u );
        EXPECT_FALSE( outcome.timed_out );
    }

    TEST_F( ToolGateTest, AToolThatThrowsReachesTheModelAsAnErrorResult )
    {
        body_ = []( std::stop_token ) -> std::string { throw std::runtime_error( "the disk is read only" ); };

        const ToolOutcome outcome = answer( answering( { ToolAnswer::Once } ) );

        EXPECT_TRUE( outcome.ran );
        EXPECT_TRUE( outcome.is_error );
        EXPECT_EQ( outcome.result, "Error: tool 'delete_file' failed: the disk is read only" );
        EXPECT_FALSE( outcome.returned_bytes );
    }

    TEST_F( ToolGateTest, ACallThatRunsOutOfItsTimeIsAskedToStopAndTheResponseGoesOn )
    {
        Tool slow = deleteTool();
        slow.name = "slow_delete";
        slow.time = std::chrono::milliseconds( 50 );
        gate_.add( std::move( slow ) );

        bool saw_stop = false;
        body_ = [&saw_stop]( std::stop_token stop )
            {
                const std::string result = waitForStop( stop );
                saw_stop = stop.stop_requested();

                return result;
            };

        // The tool's own time wins over the response's.
        const ToolOutcome outcome = gate_.answer( "slow_delete", kArguments, answering( { ToolAnswer::Once } ), {},
            AgentLimits{ .tool_time = std::chrono::minutes( 5 ) } );

        EXPECT_TRUE( saw_stop );
        EXPECT_TRUE( outcome.timed_out );
        EXPECT_FALSE( outcome.stopped ) << "a call out of its own time does not stop the response";
        EXPECT_TRUE( outcome.is_error );
        EXPECT_EQ( outcome.result, "Error: tool 'slow_delete' ran out of its 50 ms, so it may not have finished." );
    }

    TEST_F( ToolGateTest, AToolThatSetsNoTimeRunsWithinTheResponsesToolTime )
    {
        body_ = waitForStop;

        const ToolOutcome outcome = answer( answering( { ToolAnswer::Once } ), kArguments, {},
            AgentLimits{ .tool_time = std::chrono::milliseconds( 50 ) } );

        EXPECT_TRUE( outcome.timed_out );
        EXPECT_EQ( outcome.result, "Error: tool 'delete_file' ran out of its 50 ms, so it may not have finished." );
    }

    TEST_F( ToolGateTest, AResultLimitIsTheToolsOwnElseTheResponses )
    {
        Tool capped = deleteTool();
        capped.name = "capped_delete";
        capped.result_tokens = 10;
        gate_.add( std::move( capped ) );

        const AgentLimits limits{ .result_tokens = 500 };

        EXPECT_EQ( gate_.answer( "capped_delete", kArguments, answering( { ToolAnswer::Once } ), {}, limits ).result_tokens, 10 );
        EXPECT_EQ( answer( answering( { ToolAnswer::Once } ), kArguments, {}, limits ).result_tokens, 500 );
        EXPECT_FALSE( answer( answering( { ToolAnswer::Once } ) ).result_tokens );
    }

    // ================================================================
    // Stopped
    // ================================================================

    TEST_F( ToolGateTest, AReplyStoppedBeforeACallRunsNothingAndAsksNothing )
    {
        std::stop_source stop;
        stop.request_stop();

        const ToolOutcome outcome = answer( answering( { ToolAnswer::Once } ), kArguments, stop.get_token() );

        EXPECT_TRUE( outcome.stopped );
        EXPECT_FALSE( outcome.ran );
        EXPECT_TRUE( asked_.empty() );
        EXPECT_TRUE( ran_.empty() );
    }

    TEST_F( ToolGateTest, AReplyStoppedWhileThePersonIsAskedRunsNothing )
    {
        std::stop_source stop;

        const ToolOutcome outcome = answer( [&stop]( const ToolQuestion& )
            {
                stop.request_stop();

                return ToolAnswer::Once;
            }, kArguments, stop.get_token() );

        EXPECT_TRUE( outcome.stopped );
        EXPECT_FALSE( outcome.ran );
        EXPECT_TRUE( ran_.empty() );
    }

    TEST_F( ToolGateTest, AReplyStoppedWhileACallRunsAsksItToStopAndGivesTheModelNothingItReturned )
    {
        std::stop_source stop;
        bool saw_stop = false;

        body_ = [&stop, &saw_stop]( std::stop_token running )
            {
                stop.request_stop();
                saw_stop = running.stop_requested();

                return std::string( "partial" );
            };

        const ToolOutcome outcome = answer( answering( { ToolAnswer::Once } ), kArguments, stop.get_token() );

        EXPECT_TRUE( saw_stop );
        EXPECT_TRUE( outcome.stopped );
        EXPECT_TRUE( outcome.ran );
        EXPECT_TRUE( outcome.result.empty() );
    }

    // ================================================================
    // Registration
    // ================================================================

    TEST_F( ToolGateTest, RefusesAToolItCannotOffer )
    {
        const auto with = [this]( auto change )
            {
                Tool tool = deleteTool();
                tool.name = "another";
                change( tool );

                return tool;
            };

        EXPECT_THROW( gate_.add( deleteTool() ), std::invalid_argument ) << "a name already registered";
        EXPECT_THROW( gate_.add( with( []( Tool& tool ) { tool.name.clear(); } ) ), std::invalid_argument );
        EXPECT_THROW( gate_.add( with( []( Tool& tool ) { tool.name = "files.delete"; } ) ), std::invalid_argument );
        EXPECT_THROW( gate_.add( with( []( Tool& tool ) { tool.name = "delete file"; } ) ), std::invalid_argument );
        EXPECT_THROW( gate_.add( with( []( Tool& tool ) { tool.run = {}; } ) ), std::invalid_argument );
        EXPECT_THROW( gate_.add( with( []( Tool& tool ) { tool.parameters = "{not json"; } ) ), std::invalid_argument );
        EXPECT_THROW( gate_.add( with( []( Tool& tool )
            { tool.parameters = R"({"type":"object","properties":{"p":{"type":"string","pattern":"x"}}})"; } ) ),
            std::invalid_argument );
        EXPECT_THROW( gate_.add( with( []( Tool& tool ) { tool.time = std::chrono::milliseconds( 0 ); } ) ), std::invalid_argument );
        EXPECT_THROW( gate_.add( with( []( Tool& tool ) { tool.result_tokens = 0; } ) ), std::invalid_argument );

        EXPECT_EQ( gate_.tools().size(), 1u );
        EXPECT_NO_THROW( gate_.add( with( []( Tool& tool ) { tool.name = "Delete-File_2"; } ) ) );
    }
}
