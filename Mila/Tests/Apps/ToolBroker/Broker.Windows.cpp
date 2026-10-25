#define NOMINMAX
#include <windows.h>
#include <gtest/gtest.h>

import std;
import ToolBroker.Broker;
import ToolBroker.Json;
import ToolBroker.Policy;
import ToolBroker.ProtocolError;
import ToolBroker.ToolRule;
import ToolBroker.Worker;
import ToolBroker.WorkerProgram;

namespace Mila::Tests::Apps::ToolBroker
{
    using Mila::ToolBroker::Broker;
    using Mila::ToolBroker::Policy;
    using Mila::ToolBroker::ProtocolError;
    using Mila::ToolBroker::ToolRule;

    namespace
    {
        Mila::ToolBroker::Worker echoWorker( const wchar_t* label )
        {
            return Mila::ToolBroker::launchWorker(
                Mila::ToolBroker::WorkerProgram{ std::filesystem::path( TOOL_BROKER_ECHO_WORKER ), { L"--json", label } }, {} );
        }

        /** A broker over two echo workers, `alpha` and `beta`, each offering one tool, `echo`. */
        std::unique_ptr<Broker> makeBroker( Policy policy = {} )
        {
            auto broker = std::make_unique<Broker>( std::move( policy ) );
            broker->addServer( "alpha", echoWorker( L"alpha" ) );
            broker->addServer( "beta", echoWorker( L"beta" ) );

            return broker;
        }

        nlohmann::json meta( bool can_ask, std::string agent = "agent", std::string conversation = {} )
        {
            nlohmann::json capabilities = nlohmann::json::object();

            if ( can_ask )
            {
                capabilities[ "elicitation" ] = { { "form", nlohmann::json::object() } };
            }

            nlohmann::json result = { { "io.modelcontextprotocol/protocolVersion", "2026-07-28" },
                { "io.modelcontextprotocol/clientCapabilities", capabilities },
                { std::string( Mila::ToolBroker::kAgentKey ), agent } };

            if ( !conversation.empty() )
            {
                result[ std::string( Mila::ToolBroker::kConversationKey ) ] = conversation;
            }

            return result;
        }

        nlohmann::json call( Broker& broker, int id, std::string_view tool, std::string_view text, const nlohmann::json& request_meta,
            const nlohmann::json& extra = nlohmann::json::object() )
        {
            nlohmann::json params = { { "name", tool }, { "arguments", { { "text", text } } }, { "_meta", request_meta } };
            params.update( extra );

            const nlohmann::json request = { { "jsonrpc", "2.0" }, { "id", id }, { "method", "tools/call" }, { "params", params } };

            return nlohmann::json::parse( broker.handle( request.dump(), []( std::string_view ) {} ).value() );
        }

        nlohmann::json approval( std::string_view action, std::string_view decision = {} )
        {
            nlohmann::json response = { { "action", action } };

            if ( !decision.empty() )
            {
                response[ "content" ] = { { "decision", decision } };
            }

            return { { std::string( Mila::ToolBroker::kApprovalInput ), response } };
        }

        nlohmann::json retry( const nlohmann::json& asked, const nlohmann::json& responses )
        {
            return { { "requestState", asked[ "result" ][ "requestState" ] }, { "inputResponses", responses } };
        }

        std::string text( const nlohmann::json& response )
        {
            return response[ "result" ][ "content" ][ 0 ][ "text" ].get<std::string>();
        }

        std::vector<std::string> choices( const nlohmann::json& asked )
        {
            std::vector<std::string> values;
            const auto pointer = nlohmann::json::json_pointer(
                "/result/inputRequests/approval/params/requestedSchema/properties/decision/oneOf" );

            for ( const auto& choice : asked[ pointer ] )
            {
                values.push_back( choice[ "const" ].get<std::string>() );
            }

            return values;
        }
    }

    TEST( BrokerTests, ListsEveryServersToolsUnderTheServersName )
    {
        auto broker = makeBroker();
        const nlohmann::json request = { { "jsonrpc", "2.0" }, { "id", 1 }, { "method", "tools/list" },
            { "params", { { "_meta", meta( false ) } } } };

        const auto response = nlohmann::json::parse( broker->handle( request.dump(), []( std::string_view ) {} ).value() );

        ASSERT_EQ( response[ "result" ][ "tools" ].size(), 2u );
        EXPECT_EQ( response[ "result" ][ "resultType" ], "complete" );
        EXPECT_EQ( response[ "result" ][ "tools" ][ 0 ][ "name" ], "alpha.echo" );
        EXPECT_EQ( response[ "result" ][ "tools" ][ 1 ][ "name" ], "beta.echo" );
        EXPECT_EQ( response[ "result" ][ "tools" ][ 1 ][ "description" ], "Repeats its text" );
    }

    TEST( BrokerTests, RunsAnAllowedCallOnItsOwnServerUnderTheToolsOwnName )
    {
        Policy policy;
        policy.setRule( "agent", "beta.echo", ToolRule::Allow );
        auto broker = makeBroker( std::move( policy ) );

        const auto response = call( *broker, 2, "beta.echo", "hi", meta( false ) );

        EXPECT_EQ( response[ "id" ], 2 );
        EXPECT_EQ( text( response ), "beta:echo:hi" );
    }

    TEST( BrokerTests, RefusesAToolNoServerOffers )
    {
        auto broker = makeBroker();

        for ( const auto* tool : { "gamma.echo", "alpha.other", "echo", "alpha" } )
        {
            const auto response = call( *broker, 3, tool, "hi", meta( true ) );

            EXPECT_EQ( response[ "error" ][ "code" ], ProtocolError::kInvalidParams ) << tool;
        }
    }

    TEST( BrokerTests, AnswersAMethodItDoesNotServe )
    {
        auto broker = makeBroker();
        const nlohmann::json request = { { "jsonrpc", "2.0" }, { "id", 4 }, { "method", "resources/list" },
            { "params", { { "_meta", meta( false ) } } } };

        const auto response = nlohmann::json::parse( broker->handle( request.dump(), []( std::string_view ) {} ).value() );

        EXPECT_EQ( response[ "error" ][ "code" ], ProtocolError::kMethodNotFound );
    }

    TEST( BrokerTests, RefusesAServerNameThatWouldMakeToolNamesAmbiguous )
    {
        Broker broker{ Policy{} };

        EXPECT_THROW( broker.addServer( "files.web", echoWorker( L"x" ) ), std::invalid_argument );
        EXPECT_THROW( broker.addServer( "", echoWorker( L"x" ) ), std::invalid_argument );

        broker.addServer( "files", echoWorker( L"x" ) );
        EXPECT_THROW( broker.addServer( "files", echoWorker( L"x" ) ), std::invalid_argument );
    }

    TEST( BrokerTests, TellsTheModelADeniedCallDidNotRun )
    {
        Policy policy;
        policy.setRule( "agent", "alpha.echo", ToolRule::Deny );
        auto broker = makeBroker( std::move( policy ) );

        const auto response = call( *broker, 5, "alpha.echo", "hi", meta( true ) );

        EXPECT_EQ( response[ "result" ][ "resultType" ], "complete" );
        EXPECT_TRUE( response[ "result" ][ "isError" ].get<bool>() );
        EXPECT_NE( text( response ).find( "do not allow alpha.echo" ), std::string::npos );
    }

    TEST( BrokerTests, KeepsEachAgentsRulesToItself )
    {
        Policy policy;
        policy.setRule( "trusted", "alpha.echo", ToolRule::Allow );
        auto broker = makeBroker( std::move( policy ) );

        EXPECT_EQ( text( call( *broker, 6, "alpha.echo", "hi", meta( false, "trusted" ) ) ), "alpha:echo:hi" );
        EXPECT_TRUE( call( *broker, 7, "alpha.echo", "hi", meta( false, "other" ) )[ "result" ][ "isError" ].get<bool>() );
    }

    TEST( BrokerTests, RefusesWhatItWouldAskWhenNobodyCanBeAsked )
    {
        auto broker = makeBroker();

        const auto response = call( *broker, 8, "alpha.echo", "hi", meta( false ) );

        EXPECT_TRUE( response[ "result" ][ "isError" ].get<bool>() );
        EXPECT_NE( text( response ).find( "Nobody was present" ), std::string::npos );
    }

    TEST( BrokerTests, AsksWithTheToolAndItsArgumentsInFull )
    {
        auto broker = makeBroker();

        const auto asked = call( *broker, 9, "alpha.echo", "hi", meta( true ) );
        const auto& result = asked[ "result" ];

        EXPECT_EQ( result[ "resultType" ], "input_required" );
        EXPECT_TRUE( result[ "requestState" ].is_string() );
        EXPECT_EQ( result[ "inputRequests" ][ "approval" ][ "method" ], "elicitation/create" );

        const auto message = result[ "inputRequests" ][ "approval" ][ "params" ][ "message" ].get<std::string>();
        EXPECT_NE( message.find( "alpha.echo" ), std::string::npos );
        EXPECT_NE( message.find( R"({"text":"hi"})" ), std::string::npos );

        // No conversation is named, so nothing can be allowed for one.
        EXPECT_EQ( choices( asked ), ( std::vector<std::string>{ "allow_once", "deny" } ) );
    }

    TEST( BrokerTests, RunsTheCallOnceThePersonAllowsIt )
    {
        auto broker = makeBroker();
        const auto asked = call( *broker, 10, "alpha.echo", "hi", meta( true ) );

        const auto response = call( *broker, 11, "alpha.echo", "hi", meta( true ), retry( asked, approval( "accept", "allow_once" ) ) );

        EXPECT_EQ( text( response ), "alpha:echo:hi" );
    }

    TEST( BrokerTests, UsesAnApprovalOnlyOnce )
    {
        auto broker = makeBroker();
        const auto asked = call( *broker, 12, "alpha.echo", "hi", meta( true ) );
        const auto allowed = retry( asked, approval( "accept", "allow_once" ) );

        EXPECT_EQ( text( call( *broker, 13, "alpha.echo", "hi", meta( true ), allowed ) ), "alpha:echo:hi" );

        const auto replayed = call( *broker, 14, "alpha.echo", "hi", meta( true ), allowed );
        EXPECT_EQ( replayed[ "error" ][ "code" ], ProtocolError::kInvalidParams );

        // Allowing once grants nothing later.
        EXPECT_EQ( call( *broker, 15, "alpha.echo", "hi", meta( true ) )[ "result" ][ "resultType" ], "input_required" );
    }

    TEST( BrokerTests, RefusesAnApprovalPresentedForAnotherCall )
    {
        auto broker = makeBroker();
        const auto allowed = retry( call( *broker, 16, "alpha.echo", "hi", meta( true ) ), approval( "accept", "allow_once" ) );

        EXPECT_EQ( call( *broker, 17, "alpha.echo", "other", meta( true ), allowed )[ "error" ][ "code" ], ProtocolError::kInvalidParams );

        const auto for_beta = retry( call( *broker, 18, "alpha.echo", "hi", meta( true ) ), approval( "accept", "allow_once" ) );
        EXPECT_EQ( call( *broker, 19, "beta.echo", "hi", meta( true ), for_beta )[ "error" ][ "code" ], ProtocolError::kInvalidParams );

        const auto for_other_agent = retry( call( *broker, 20, "alpha.echo", "hi", meta( true ) ), approval( "accept", "allow_once" ) );
        EXPECT_EQ( call( *broker, 21, "alpha.echo", "hi", meta( true, "intruder" ), for_other_agent )[ "error" ][ "code" ],
            ProtocolError::kInvalidParams );

        const auto made_up = nlohmann::json{ { "requestState", "0123456789abcdef0123456789abcdef" },
            { "inputResponses", approval( "accept", "allow_once" ) } };
        EXPECT_EQ( call( *broker, 22, "alpha.echo", "hi", meta( true ), made_up )[ "error" ][ "code" ], ProtocolError::kInvalidParams );
    }

    TEST( BrokerTests, TellsTheModelThePersonDeclined )
    {
        auto broker = makeBroker();

        for ( const auto& answer : { approval( "decline" ), approval( "cancel" ), approval( "accept", "deny" ) } )
        {
            const auto asked = call( *broker, 23, "alpha.echo", "hi", meta( true ) );
            const auto response = call( *broker, 24, "alpha.echo", "hi", meta( true ), retry( asked, answer ) );

            EXPECT_TRUE( response[ "result" ][ "isError" ].get<bool>() );
            EXPECT_NE( text( response ).find( "declined" ), std::string::npos );
        }
    }

    TEST( BrokerTests, AsksAgainWhenTheRetryCarriesNoAnswer )
    {
        auto broker = makeBroker();
        const auto asked = call( *broker, 25, "alpha.echo", "hi", meta( true ) );

        const auto response = call( *broker, 26, "alpha.echo", "hi", meta( true ),
            { { "requestState", asked[ "result" ][ "requestState" ] } } );

        EXPECT_EQ( response[ "result" ][ "resultType" ], "input_required" );
        EXPECT_NE( response[ "result" ][ "requestState" ], asked[ "result" ][ "requestState" ] );
    }

    TEST( BrokerTests, AllowsAToolForTheRestOfOneConversationOnly )
    {
        auto broker = makeBroker();
        const auto in_first = meta( true, "agent", "first" );

        const auto asked = call( *broker, 27, "alpha.echo", "hi", in_first );
        EXPECT_EQ( choices( asked ), ( std::vector<std::string>{ "allow_once", "allow_conversation", "deny" } ) );
        EXPECT_EQ( text( call( *broker, 28, "alpha.echo", "hi", in_first, retry( asked, approval( "accept", "allow_conversation" ) ) ) ),
            "alpha:echo:hi" );

        EXPECT_EQ( text( call( *broker, 29, "alpha.echo", "again", in_first ) ), "alpha:echo:again" );
        EXPECT_EQ( call( *broker, 30, "beta.echo", "hi", in_first )[ "result" ][ "resultType" ], "input_required" );
        EXPECT_EQ( call( *broker, 31, "alpha.echo", "hi", meta( true, "agent", "second" ) )[ "result" ][ "resultType" ],
            "input_required" );
    }

    TEST( BrokerTests, NeverAllowsForAConversationTheCallDidNotName )
    {
        auto broker = makeBroker();
        const auto asked = call( *broker, 32, "alpha.echo", "hi", meta( true ) );

        const auto response = call( *broker, 33, "alpha.echo", "hi", meta( true ), retry( asked, approval( "accept", "allow_conversation" ) ) );

        EXPECT_EQ( response[ "result" ][ "resultType" ], "input_required" );
    }
}
