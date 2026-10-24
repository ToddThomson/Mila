#include <gtest/gtest.h>

import std;
import ToolBroker.ClientMessage;
import ToolBroker.Json;
import ToolBroker.ProtocolError;

namespace Mila::Tests::Apps::ToolBroker
{
    using Mila::ToolBroker::errorResponse;
    using Mila::ToolBroker::parseClientMessage;
    using Mila::ToolBroker::ProtocolError;

    namespace
    {
        constexpr std::string_view kMeta =
            R"("_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}})";

        std::string request( std::string_view id, std::string_view method = "tools/list" )
        {
            return std::format( R"({{"jsonrpc":"2.0","id":{},"method":"{}","params":{{{}}}}})", id, method, kMeta );
        }

        int errorCode( std::string_view text )
        {
            const auto parsed = parseClientMessage( text );

            return parsed ? 0 : parsed.error().code;
        }
    }

    TEST( ClientMessageTests, ReadsARequestWithAnIntegerId )
    {
        const auto parsed = parseClientMessage( request( "7" ) );

        ASSERT_TRUE( parsed.has_value() );
        EXPECT_EQ( parsed->id, 7 );
        EXPECT_EQ( parsed->method, "tools/list" );
        EXPECT_FALSE( parsed->isNotification() );
    }

    TEST( ClientMessageTests, ReadsARequestWithAStringId )
    {
        const auto parsed = parseClientMessage( request( R"("call-1")" ) );

        ASSERT_TRUE( parsed.has_value() );
        EXPECT_EQ( parsed->id, "call-1" );
    }

    TEST( ClientMessageTests, ReadsANotification )
    {
        const auto parsed = parseClientMessage( R"({"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":7}})" );

        ASSERT_TRUE( parsed.has_value() );
        EXPECT_TRUE( parsed->isNotification() );
    }

    TEST( ClientMessageTests, RefusesTextThatIsNotJson )
    {
        EXPECT_EQ( errorCode( "{not json" ), ProtocolError::kParseError );
    }

    TEST( ClientMessageTests, RefusesJsonThatIsNotAnObject )
    {
        EXPECT_EQ( errorCode( "[1,2]" ), ProtocolError::kInvalidRequest );
    }

    TEST( ClientMessageTests, RefusesAnotherJsonRpcVersion )
    {
        EXPECT_EQ( errorCode( R"({"jsonrpc":"1.0","id":1,"method":"tools/list"})" ), ProtocolError::kInvalidRequest );
        EXPECT_EQ( errorCode( R"({"id":1,"method":"tools/list"})" ), ProtocolError::kInvalidRequest );
    }

    TEST( ClientMessageTests, RefusesANullId )
    {
        EXPECT_EQ( errorCode( request( "null" ) ), ProtocolError::kInvalidRequest );
    }

    TEST( ClientMessageTests, RefusesAnIdThatIsNeitherStringNorInteger )
    {
        EXPECT_EQ( errorCode( request( "1.5" ) ), ProtocolError::kInvalidRequest );
        EXPECT_EQ( errorCode( request( "[1]" ) ), ProtocolError::kInvalidRequest );
    }

    TEST( ClientMessageTests, RefusesAMessageWithoutAMethod )
    {
        EXPECT_EQ( errorCode( R"({"jsonrpc":"2.0","id":1})" ), ProtocolError::kInvalidRequest );
    }

    TEST( ClientMessageTests, RefusesAResponseFromAClient )
    {
        EXPECT_EQ( errorCode( R"({"jsonrpc":"2.0","id":1,"result":{}})" ), ProtocolError::kInvalidRequest );
    }

    TEST( ClientMessageTests, RefusesParametersThatAreNotAnObject )
    {
        EXPECT_EQ( errorCode( R"({"jsonrpc":"2.0","id":1,"method":"tools/list","params":[1]})" ), ProtocolError::kInvalidParams );
    }

    TEST( ClientMessageTests, RefusesARequestWithoutItsProtocolMetadata )
    {
        EXPECT_EQ( errorCode( R"({"jsonrpc":"2.0","id":1,"method":"tools/list"})" ), ProtocolError::kInvalidParams );
        EXPECT_EQ( errorCode( R"({"jsonrpc":"2.0","id":1,"method":"tools/list","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28"}}})" ),
            ProtocolError::kInvalidParams );
    }

    TEST( ClientMessageTests, KeepsTheIdOfARequestItRefuses )
    {
        const auto parsed = parseClientMessage( R"({"jsonrpc":"2.0","id":"keep","method":"tools/list"})" );

        ASSERT_FALSE( parsed.has_value() );
        EXPECT_EQ( parsed.error().id, "keep" );
    }

    TEST( ClientMessageTests, WritesAnErrorResponseWithTheRequestsId )
    {
        const auto response = nlohmann::json::parse( errorResponse( { ProtocolError::kInvalidParams, "Invalid params", 3 } ) );

        EXPECT_EQ( response[ "jsonrpc" ], "2.0" );
        EXPECT_EQ( response[ "id" ], 3 );
        EXPECT_EQ( response[ "error" ][ "code" ], ProtocolError::kInvalidParams );
        EXPECT_EQ( response[ "error" ][ "message" ], "Invalid params" );
    }

    TEST( ClientMessageTests, LeavesOutAnIdItCouldNotRead )
    {
        const auto response = nlohmann::json::parse( errorResponse( { ProtocolError::kParseError, "Parse error", nullptr } ) );

        EXPECT_FALSE( response.contains( "id" ) );
    }
}
