/**
 * @file ToolBroker.ClientMessage.ixx
 * @brief A message from a client, read and checked against MCP's base protocol before anything acts on it.
 */

export module ToolBroker.ClientMessage;

import std;
import ToolBroker.Json;
import ToolBroker.ProtocolError;

namespace Mila::ToolBroker
{
    /**
     * @brief A client's request or notification, as JSON-RPC 2.0 and MCP 2026-07-28 shape it.
     */
    export struct ClientMessage
    {
        /** A string or an integer for a request; null for a notification, which has none. */
        nlohmann::json id;

        std::string method;

        /** An object, or null when the message carries no parameters. */
        nlohmann::json params;

        bool isNotification() const noexcept
        {
            return id.is_null();
        }
    };

    /**
     * @brief Reads one line from a client as a message, or the error to answer it with.
     *
     * Refused: text that is not JSON (parse error); anything but an object with `jsonrpc` "2.0" and a string
     * `method` -- so also a response, which a client never sends -- and an id that is present but neither a string
     * nor an integer, since MCP forbids a null one (invalid request); and parameters that are not an object, or a request
     * whose `_meta` lacks the protocol version or the client capabilities every MCP request carries (invalid params).
     */
    export std::expected<ClientMessage, ProtocolError> parseClientMessage( std::string_view text )
    {
        auto message = nlohmann::json::parse( text, nullptr, false );

        if ( message.is_discarded() )
        {
            return std::unexpected( ProtocolError{ ProtocolError::kParseError, "Parse error", nullptr } );
        }

        if ( !message.is_object() )
        {
            return std::unexpected( ProtocolError{ ProtocolError::kInvalidRequest, "A message must be a JSON object", nullptr } );
        }

        nlohmann::json id = nullptr;
        const auto id_field = message.find( "id" );

        if ( id_field != message.end() )
        {
            if ( !id_field->is_string() && !id_field->is_number_integer() )
            {
                return std::unexpected( ProtocolError{ ProtocolError::kInvalidRequest,
                    "A request's id must be a string or an integer", nullptr } );
            }

            id = *id_field;
        }

        const auto version = message.find( "jsonrpc" );

        if ( version == message.end() || *version != "2.0" )
        {
            return std::unexpected( ProtocolError{ ProtocolError::kInvalidRequest, "A message must be JSON-RPC 2.0", id } );
        }

        const auto method = message.find( "method" );

        if ( method == message.end() || !method->is_string() )
        {
            return std::unexpected( ProtocolError{ ProtocolError::kInvalidRequest, "A message must name its method", id } );
        }

        nlohmann::json params = nullptr;
        const auto params_field = message.find( "params" );

        if ( params_field != message.end() )
        {
            if ( !params_field->is_object() )
            {
                return std::unexpected( ProtocolError{ ProtocolError::kInvalidParams, "Parameters must be an object", id } );
            }

            params = *params_field;
        }

        if ( !id.is_null() )
        {
            const auto meta = params.is_object() ? params.find( "_meta" ) : params.end();
            const bool versioned = meta != params.end() && meta->is_object()
                && meta->contains( "io.modelcontextprotocol/protocolVersion" )
                && ( *meta )[ "io.modelcontextprotocol/protocolVersion" ].is_string();
            const bool capable = versioned
                && meta->contains( "io.modelcontextprotocol/clientCapabilities" )
                && ( *meta )[ "io.modelcontextprotocol/clientCapabilities" ].is_object();

            if ( !versioned || !capable )
            {
                return std::unexpected( ProtocolError{ ProtocolError::kInvalidParams,
                    "A request must carry its protocol version and client capabilities in _meta", id } );
            }
        }

        return ClientMessage{ std::move( id ), method->get<std::string>(), std::move( params ) };
    }
}
