/**
 * @file BrokerClient.Windows.cpp
 * @brief Mila.AI's broker client run against a real broker, over its named pipe, with no application involved
 *        (the gate of BACKLOG's "An application cannot use the tool service without writing its own client").
 */

#define NOMINMAX
#include <windows.h>
#include <gtest/gtest.h>

import std;
import Mila.AI;
import ToolBroker.Broker;
import ToolBroker.PipeServer;
import ToolBroker.Policy;
import ToolBroker.ToolRule;
import ToolBroker.Worker;
import ToolBroker.WorkerProgram;

namespace Mila::Tests::AI
{
    using Mila::AI::ApprovalChoice;
    using Mila::AI::ApprovalRequest;
    using Mila::AI::Approver;
    using Mila::AI::BrokerClient;
    using Mila::ToolBroker::Policy;
    using Mila::ToolBroker::ToolRule;

    namespace
    {
        Mila::ToolBroker::Worker echoWorker( const wchar_t* label )
        {
            return Mila::ToolBroker::launchWorker(
                Mila::ToolBroker::WorkerProgram{ std::filesystem::path( TOOL_BROKER_ECHO_WORKER ), { L"--json", label } }, {} );
        }

        std::wstring uniquePipeName()
        {
            static std::atomic<int> next{ 0 };

            return std::format( L"\\\\.\\pipe\\mila-ai-broker-client-tests-{}-{}", ::GetCurrentProcessId(), next++ );
        }

        /**
         * A broker over `alpha` and `beta` echo workers, serving one client on its pipe from a thread of its own.
         * The client must connect, and be gone before the broker is destroyed, so the thread ends.
         */
        struct RunningBroker
        {
            explicit RunningBroker( Policy policy )
                : name( uniquePipeName() ), server( name ), broker( std::move( policy ) )
            {
                broker.addServer( "alpha", echoWorker( L"alpha" ) );
                broker.addServer( "beta", echoWorker( L"beta" ) );

                thread = std::jthread( [this]
                    {
                        auto client = server.accept();
                        broker.serve( client );
                    } );
            }

            std::wstring name;
            Mila::ToolBroker::PipeServer server;
            Mila::ToolBroker::Broker broker;
            std::jthread thread;
        };

        /** Answers every approval with `choice` and keeps what it was shown. */
        struct RecordingApprover
        {
            explicit RecordingApprover( ApprovalChoice answer )
                : choice( answer )
            {
            }

            Approver approver()
            {
                return [this]( const ApprovalRequest& request )
                    {
                        shown.push_back( request );

                        return choice;
                    };
            }

            ApprovalChoice choice;
            std::vector<ApprovalRequest> shown;
        };
    }

    TEST( BrokerClientTests, ListsTheBrokersTools )
    {
        RunningBroker running{ Policy{} };
        BrokerClient client( running.name, "agent", "MilaAITests" );

        const auto tools = client.listTools();

        ASSERT_EQ( tools.size(), 2u );
        EXPECT_EQ( tools[ 0 ].name, "alpha.echo" );
        EXPECT_EQ( tools[ 1 ].name, "beta.echo" );
        EXPECT_EQ( tools[ 0 ].description, "Repeats its text" );
        EXPECT_NE( tools[ 0 ].input_schema.find( R"("text")" ), std::string::npos );
    }

    TEST( BrokerClientTests, RunsAnAllowedToolWithoutAsking )
    {
        Policy policy;
        policy.setRule( "agent", "alpha.echo", ToolRule::Allow );
        RunningBroker running{ std::move( policy ) };
        BrokerClient client( running.name, "agent", "MilaAITests" );
        RecordingApprover person( ApprovalChoice::Deny );

        const auto result = client.call( "alpha.echo", R"({"text":"hi"})", {}, person.approver() );

        EXPECT_FALSE( result.is_error );
        EXPECT_EQ( result.text, "alpha:echo:hi" );
        EXPECT_TRUE( person.shown.empty() );
    }

    TEST( BrokerClientTests, AsksThePersonAndRunsTheCallWhenAllowed )
    {
        RunningBroker running{ Policy{} };
        BrokerClient client( running.name, "agent", "MilaAITests" );
        client.listTools();
        RecordingApprover person( ApprovalChoice::AllowOnce );

        const auto result = client.call( "beta.echo", R"({ "text": "hi" })", {}, person.approver() );

        EXPECT_FALSE( result.is_error );
        EXPECT_EQ( result.text, "beta:echo:hi" );
        ASSERT_EQ( person.shown.size(), 1u );
        EXPECT_EQ( person.shown[ 0 ].tool, "beta.echo" );
        EXPECT_EQ( person.shown[ 0 ].description, "Repeats its text" );
        EXPECT_EQ( person.shown[ 0 ].arguments, R"({"text":"hi"})" );
        EXPECT_FALSE( person.shown[ 0 ].offers_conversation );
    }

    TEST( BrokerClientTests, TellsTheModelWhenThePersonDeclines )
    {
        RunningBroker running{ Policy{} };
        BrokerClient client( running.name, "agent", "MilaAITests" );
        RecordingApprover person( ApprovalChoice::Deny );

        const auto result = client.call( "alpha.echo", R"({"text":"hi"})", {}, person.approver() );

        EXPECT_TRUE( result.is_error );
        EXPECT_NE( result.text.find( "declined" ), std::string::npos );
        EXPECT_EQ( person.shown.size(), 1u );
    }

    TEST( BrokerClientTests, RefusesWhatTheRulesWouldAskWhenNobodyCanBeAsked )
    {
        RunningBroker running{ Policy{} };
        BrokerClient client( running.name, "agent", "MilaAITests" );

        const auto result = client.call( "alpha.echo", R"({"text":"hi"})", {}, Approver{} );

        EXPECT_TRUE( result.is_error );
        EXPECT_NE( result.text.find( "Nobody was present" ), std::string::npos );
    }

    TEST( BrokerClientTests, AllowsAToolForTheRestOfTheConversation )
    {
        RunningBroker running{ Policy{} };
        BrokerClient client( running.name, "agent", "MilaAITests" );
        RecordingApprover person( ApprovalChoice::AllowForConversation );

        EXPECT_EQ( client.call( "alpha.echo", R"({"text":"one"})", "first", person.approver() ).text, "alpha:echo:one" );
        EXPECT_EQ( client.call( "alpha.echo", R"({"text":"two"})", "first", person.approver() ).text, "alpha:echo:two" );
        ASSERT_EQ( person.shown.size(), 1u );
        EXPECT_TRUE( person.shown[ 0 ].offers_conversation );

        EXPECT_EQ( client.call( "alpha.echo", R"({"text":"three"})", "second", person.approver() ).text, "alpha:echo:three" );
        EXPECT_EQ( person.shown.size(), 2u );
    }

    TEST( BrokerClientTests, AllowsOnceWhenTheConversationChoiceWasNotOffered )
    {
        RunningBroker running{ Policy{} };
        BrokerClient client( running.name, "agent", "MilaAITests" );
        RecordingApprover person( ApprovalChoice::AllowForConversation );

        EXPECT_EQ( client.call( "alpha.echo", R"({"text":"one"})", {}, person.approver() ).text, "alpha:echo:one" );
        EXPECT_EQ( client.call( "alpha.echo", R"({"text":"two"})", {}, person.approver() ).text, "alpha:echo:two" );
        EXPECT_EQ( person.shown.size(), 2u );
    }

    TEST( BrokerClientTests, ReportsAnUnknownToolAsAFailedResult )
    {
        RunningBroker running{ Policy{} };
        BrokerClient client( running.name, "agent", "MilaAITests" );

        const auto result = client.call( "gamma.echo", R"({"text":"hi"})", {}, Approver{} );

        EXPECT_TRUE( result.is_error );
        EXPECT_NE( result.text.find( "Unknown tool: gamma.echo" ), std::string::npos );
    }

    TEST( BrokerClientTests, ReportsArgumentsThatAreNotAJsonObjectWithoutCallingTheBroker )
    {
        Policy policy;
        policy.setRule( "agent", "alpha.echo", ToolRule::Allow );
        RunningBroker running{ std::move( policy ) };
        BrokerClient client( running.name, "agent", "MilaAITests" );

        EXPECT_TRUE( client.call( "alpha.echo", "[1, 2]", {}, Approver{} ).is_error );
        EXPECT_TRUE( client.call( "alpha.echo", "{not json", {}, Approver{} ).is_error );
        EXPECT_EQ( client.call( "alpha.echo", R"({"text":"after"})", {}, Approver{} ).text, "alpha:echo:after" );
    }

    TEST( BrokerClientTests, RefusesToStartWithoutABroker )
    {
        EXPECT_THROW( BrokerClient( L"\\\\.\\pipe\\mila-ai-no-broker-here", "agent", "MilaAITests" ), std::system_error );
    }
}
