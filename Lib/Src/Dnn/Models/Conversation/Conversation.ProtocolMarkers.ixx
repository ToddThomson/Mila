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
         * @brief Text that ends a turn early: the open of a result, which is the engine's to write.
         *
         * Never a closed call. A turn may hold several calls, and the model makes all of them before
         * the result marker -- Gemma 4 writes it as its own stop, Qwen ends the turn instead -- so
         * stopping at a call's close keeps only the first.
         */
        std::vector<std::string> stop_after;
    };
}
