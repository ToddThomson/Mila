/**
 * @file Conversation.ProtocolMarkers.ixx
 * @brief The text a template's grammar marks its channels and calls with, for a reader of the raw stream.
 */

export module Dnn.Models.Conversation.ProtocolMarkers;

import std;

namespace Mila::Dnn::Conversation
{
    /**
     * @brief Where a reply's reasoning and calls begin and end, and where generation should stop.
     *
     * For a consumer routing a stream as it arrives. Parsing a finished reply is Protocol::parseReply's,
     * since the markers say where a call is and not how its arguments are written.
     */
    export struct ProtocolMarkers
    {
        std::string reasoning_open;
        std::string reasoning_close;

        std::string tool_call_open;
        std::string tool_call_close;
        std::string tool_response_open;

        /**
         * @brief Text that ends a turn early.
         *
         * A closed call, since left running the model writes the result itself; and the open of a
         * result, which is the engine's to write and never the model's, as a backstop for a call
         * the model failed to close.
         */
        std::vector<std::string> stop_after;
    };
}
