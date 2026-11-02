/**
 * @file Conversation.ToolCall.ixx
 * @brief One call the model asked for, as a name and a JSON object of arguments.
 */

export module Dnn.Models.Conversation.ToolCall;

import std;

namespace Mila::Dnn::Conversation
{
    /**
     * @brief One call the model asked for.
     *
     * Arguments stay JSON text rather than becoming a parsed object: every template renders them
     * back out, every host hands them on, and a round trip through a typed representation can
     * only lose something the model wrote -- the order of its keys among them, which Qwen's
     * template keeps.
     *
     * No id. Ids are how one wire format finds which call a result answers; the loop that ran the
     * call resolves that, and the result's turn names its tool (Turn::tool_name).
     */
    export struct ToolCall
    {
        std::string name;
        std::string arguments;
    };
}
