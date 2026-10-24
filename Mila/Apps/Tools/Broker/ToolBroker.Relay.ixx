/**
 * @file ToolBroker.Relay.ixx
 * @brief Carries one client request to a worker and the worker's answer back, checking both against MCP.
 */

export module ToolBroker.Relay;

import std;
import ToolBroker.ClientMessage;
import ToolBroker.Json;
import ToolBroker.LineChannel;
import ToolBroker.ProtocolError;
import ToolBroker.Worker;

namespace Mila::ToolBroker
{
    namespace
    {
        ProtocolError workerFault( const nlohmann::json& id, std::string message )
        {
            return ProtocolError{ ProtocolError::kInternalError, std::move( message ), id };
        }
    }

    /**
     * @brief Reads one message from `client` and, if it is a well-formed request, forwards it to `worker` and the
     *        worker's response back.
     *
     * A malformed message is answered by the broker and never reaches the worker. A notification is forwarded and
     * needs no answer. While a request is in flight the worker may send notifications about it, which are passed
     * on; its response must carry the request's id and exactly one of `result` and `error`. A worker that sends a
     * request -- which an MCP server never does -- or anything else, or ends, is answered for with an internal
     * error, so the client always gets one response per request.
     *
     * @return False once the client has closed; true after each message handled.
     */
    export bool relayOne( LineChannel& client, Worker& worker )
    {
        const auto text = client.receive();

        if ( !text )
        {
            return false;
        }

        auto message = parseClientMessage( *text );

        if ( !message )
        {
            client.send( errorResponse( message.error() ) );

            return true;
        }

        worker.channel().send( *text );

        if ( message->isNotification() )
        {
            return true;
        }

        while ( true )
        {
            const auto answer = worker.channel().receive();

            if ( !answer )
            {
                client.send( errorResponse( workerFault( message->id, "The tool's worker ended" ) ) );

                return true;
            }

            const auto reply = nlohmann::json::parse( *answer, nullptr, false );

            if ( reply.is_discarded() || !reply.is_object() || reply.value( "jsonrpc", "" ) != "2.0" )
            {
                client.send( errorResponse( workerFault( message->id, "The tool's worker answered with something other than JSON-RPC" ) ) );

                return true;
            }

            const bool has_id = reply.contains( "id" );

            if ( !has_id && reply.contains( "method" ) )
            {
                client.send( *answer );
                continue;
            }

            // A request from the worker fails here too: it has no result or error, or another request's id.
            if ( !has_id || reply[ "id" ] != message->id || reply.contains( "result" ) == reply.contains( "error" ) )
            {
                client.send( errorResponse( workerFault( message->id, "The tool's worker did not answer this request" ) ) );

                return true;
            }

            client.send( *answer );

            return true;
        }
    }
}
