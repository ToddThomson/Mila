/**
 * @file Conversation.Turn.ixx
 * @brief One turn of the conversation a chat template renders.
 */

export module Dnn.Models.Conversation.Turn;

import std;
export import Dnn.Models.Conversation.Role;
export import Dnn.Models.Conversation.ToolCall;

namespace Mila::Dnn::Conversation
{
    /**
     * @brief A single turn in a conversation.
     *
     * The union of what the templates read: each protocol renders the fields its grammar uses and
     * ignores the rest. No timestamp, display state or correlation id -- no template renders them.
     */
    export struct Turn
    {
        Role role;
        std::string content;

        /// Non-empty only on an Assistant turn in which the model issued calls.
        std::vector<ToolCall> tool_calls;

        /**
         * @brief On a Tool turn, the tool whose result this is.
         *
         * Set by whoever ran the call, which is the only party that knows which call a result
         * answers. Gemma's grammar renders it (`response:get_weather{...}`); Qwen's and Llama's
         * render the content alone. Never inferred from the turn's position: a result returned
         * out of order or alone would be misnamed.
         */
        std::string tool_name;

        /// On an Assistant turn, the text of its reasoning channel, for templates that render it
        /// back into history (Qwen 3.8's preserve_thinking). Empty when it was not kept.
        std::string reasoning;
    };
}
