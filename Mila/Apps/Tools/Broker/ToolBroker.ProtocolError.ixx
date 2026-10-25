/**
 * @file ToolBroker.ProtocolError.ixx
 * @brief A JSON-RPC error the broker answers with, and the response line that carries it.
 */

export module ToolBroker.ProtocolError;

import std;
import ToolBroker.Json;

namespace Mila::ToolBroker
{
    /**
     * @brief An error in the codes JSON-RPC 2.0 defines, which MCP uses for general protocol failures.
     */
    export struct ProtocolError
    {
        static constexpr int kParseError = -32700;
        static constexpr int kInvalidRequest = -32600;
        static constexpr int kMethodNotFound = -32601;
        static constexpr int kInvalidParams = -32602;
        static constexpr int kInternalError = -32603;

        int code = kInternalError;
        std::string message;

        /** The request's id, or null where it could not be read. */
        nlohmann::json id;
    };

    /**
     * @brief The error response for `error`, as one line.
     *
     * MCP lets an error response omit its id when the request's could not be read, so a null id is left out
     * rather than written as null.
     */
    export std::string errorResponse( const ProtocolError& error )
    {
        nlohmann::json response = {
            { "jsonrpc", "2.0" },
            { "error", { { "code", error.code }, { "message", error.message } } }
        };

        if ( !error.id.is_null() )
        {
            response[ "id" ] = error.id;
        }

        return response.dump();
    }
}
