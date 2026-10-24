#define NOMINMAX
#include <windows.h>
#include <gtest/gtest.h>

import std;
import ToolBroker.Json;
import ToolBroker.LineChannel;
import ToolBroker.ProtocolError;
import ToolBroker.Relay;
import ToolBroker.Worker;
import ToolBroker.WorkerProgram;

namespace Mila::Tests::Apps::ToolBroker
{
    using Mila::ToolBroker::launchWorker;
    using Mila::ToolBroker::LineChannel;
    using Mila::ToolBroker::ProtocolError;
    using Mila::ToolBroker::relayOne;
    using Mila::ToolBroker::WorkerProgram;

    namespace
    {
        constexpr std::string_view kMeta =
            R"("_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}})";

        std::string request( int id, std::string_view method, std::string_view arguments = "{}" )
        {
            return std::format( R"({{"jsonrpc":"2.0","id":{},"method":"{}","params":{{"name":"echo","arguments":{},{}}}}})",
                id, method, arguments, kMeta );
        }

        /** A client channel and the broker's end of it, joined by anonymous pipes, and a worker in JSON mode. */
        struct Fixture
        {
            LineChannel client;
            LineChannel broker;
            Mila::ToolBroker::Worker worker;
        };

        Fixture makeFixture()
        {
            HANDLE client_read = nullptr;
            HANDLE broker_write = nullptr;
            HANDLE broker_read = nullptr;
            HANDLE client_write = nullptr;

            if ( !::CreatePipe( &client_read, &broker_write, nullptr, 0 ) || !::CreatePipe( &broker_read, &client_write, nullptr, 0 ) )
            {
                throw std::runtime_error( "CreatePipe failed" );
            }

            return { LineChannel( client_read, client_write ), LineChannel( broker_read, broker_write ),
                launchWorker( WorkerProgram{ std::filesystem::path( TOOL_BROKER_ECHO_WORKER ), { L"--json" } }, {} ) };
        }

        nlohmann::json relayAndRead( Fixture& fixture, std::string_view message )
        {
            fixture.client.send( message );
            EXPECT_TRUE( relayOne( fixture.broker, fixture.worker ) );

            return nlohmann::json::parse( fixture.client.receive().value_or( "null" ) );
        }
    }

    TEST( RelayTests, CarriesARequestToTheWorkerAndItsResponseBack )
    {
        auto fixture = makeFixture();

        const auto response = relayAndRead( fixture, request( 1, "tools/call", R"({"bytes":5})" ) );

        EXPECT_EQ( response[ "id" ], 1 );
        EXPECT_EQ( response[ "result" ][ "content" ][ 0 ][ "text" ], "xxxxx" );
    }

    TEST( RelayTests, AnswersAMalformedMessageItselfWithoutTheWorker )
    {
        auto fixture = makeFixture();

        const auto response = relayAndRead( fixture, R"({"jsonrpc":"2.0","id":2,"method":"tools/list"})" );

        EXPECT_EQ( response[ "id" ], 2 );
        EXPECT_EQ( response[ "error" ][ "code" ], ProtocolError::kInvalidParams );

        // The worker saw nothing: the next request's answer is its own.
        const auto next = relayAndRead( fixture, request( 3, "tools/call" ) );
        EXPECT_EQ( next[ "id" ], 3 );
    }

    TEST( RelayTests, PassesOnANotificationTheWorkerSendsBeforeItsResponse )
    {
        auto fixture = makeFixture();

        fixture.client.send( request( 4, "test/notify-first" ) );
        ASSERT_TRUE( relayOne( fixture.broker, fixture.worker ) );

        const auto notification = nlohmann::json::parse( fixture.client.receive().value_or( "null" ) );
        const auto response = nlohmann::json::parse( fixture.client.receive().value_or( "null" ) );

        EXPECT_EQ( notification[ "method" ], "notifications/progress" );
        EXPECT_EQ( response[ "id" ], 4 );
        EXPECT_TRUE( response.contains( "result" ) );
    }

    TEST( RelayTests, AnswersForAWorkerThatRespondsToAnotherRequest )
    {
        auto fixture = makeFixture();

        const auto response = relayAndRead( fixture, request( 5, "test/wrong-id" ) );

        EXPECT_EQ( response[ "id" ], 5 );
        EXPECT_EQ( response[ "error" ][ "code" ], ProtocolError::kInternalError );
    }

    TEST( RelayTests, AnswersForAWorkerThatSendsARequest )
    {
        auto fixture = makeFixture();

        const auto response = relayAndRead( fixture, request( 6, "test/send-request" ) );

        EXPECT_EQ( response[ "id" ], 6 );
        EXPECT_EQ( response[ "error" ][ "code" ], ProtocolError::kInternalError );
    }

    TEST( RelayTests, AnswersForAWorkerThatWritesSomethingOtherThanJson )
    {
        auto fixture = makeFixture();

        const auto response = relayAndRead( fixture, request( 7, "test/not-json" ) );

        EXPECT_EQ( response[ "id" ], 7 );
        EXPECT_EQ( response[ "error" ][ "code" ], ProtocolError::kInternalError );
    }

    TEST( RelayTests, AnswersForAWorkerThatEnds )
    {
        auto fixture = makeFixture();

        const auto response = relayAndRead( fixture, request( 8, "test/exit" ) );

        EXPECT_EQ( response[ "id" ], 8 );
        EXPECT_EQ( response[ "error" ][ "code" ], ProtocolError::kInternalError );
    }

    TEST( RelayTests, ForwardsANotificationAndWaitsForNothing )
    {
        auto fixture = makeFixture();

        fixture.client.send( R"({"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":1}})" );
        EXPECT_TRUE( relayOne( fixture.broker, fixture.worker ) );

        const auto next = relayAndRead( fixture, request( 9, "tools/call" ) );
        EXPECT_EQ( next[ "id" ], 9 );
    }

    TEST( RelayTests, ReportsTheClientClosing )
    {
        auto fixture = makeFixture();

        fixture.client.closeSending();

        EXPECT_FALSE( relayOne( fixture.broker, fixture.worker ) );
    }
}
