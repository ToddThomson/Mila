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
     * @brief Sends one request to `worker` and returns the response line to give the client.
     *
     * While the request is in flight the worker may send notifications about it, which go to `pass_on`; its
     * response must carry the request's id and exactly one of `result` and `error`. A worker that sends a request
     * -- which an MCP server never does -- or anything else, or ends, is answered for with an internal error, so
     * every request gets one response.
     */
    export std::string forwardRequest( Worker& worker, std::string_view request, const nlohmann::json& id,
        const std::function<void( std::string_view )>& pass_on )
    {
        worker.channel().send( request );

        while ( true )
        {
            const auto answer = worker.channel().receive();

            if ( !answer )
            {
                return errorResponse( workerFault( id, "The tool's worker ended" ) );
            }

            const auto reply = nlohmann::json::parse( *answer, nullptr, false );

            if ( reply.is_discarded() || !reply.is_object() || reply.value( "jsonrpc", "" ) != "2.0" )
            {
                return errorResponse( workerFault( id, "The tool's worker answered with something other than JSON-RPC" ) );
            }

            const bool has_id = reply.contains( "id" );

            if ( !has_id && reply.contains( "method" ) )
            {
                pass_on( *answer );
                continue;
            }

            // A request from the worker fails here too: it has no result or error, or another request's id.
            if ( !has_id || reply[ "id" ] != id || reply.contains( "result" ) == reply.contains( "error" ) )
            {
                return errorResponse( workerFault( id, "The tool's worker did not answer this request" ) );
            }

            return *answer;
        }
    }

    /**
     * @brief Reads one message from `client` and, if it is a well-formed request, forwards it to `worker` and the
     *        worker's response back, as `forwardRequest` does.
     *
     * A malformed message is answered by the broker and never reaches the worker. A notification is forwarded and
     * needs no answer.
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

        if ( message->isNotification() )
        {
            worker.channel().send( *text );

            return true;
        }

        client.send( forwardRequest( worker, *text, message->id,
            [&client]( std::string_view notification ) { client.send( notification ); } ) );

        return true;
    }
}
